"""
V15-Clean: Training on 800 cleaned samples. Cloud-ready (Paddle 3.2.0).
"""
import os, sys, json, time, random

os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

MODEL_PATH = "PaddlePaddle/PaddleOCR-VL"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints_v15")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_DIM = 384
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
GRAD_CLIP = 1.0
BASE_LR = 2e-5
WARMUP_STEPS = 100
CHECKPOINT_STEPS = 400

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_jsonl(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

# Load data
log("Loading data...")
train_data = load_jsonl(os.path.join(PROJECT_DIR, "output", "train_clean.jsonl"))
val_data = load_jsonl(os.path.join(PROJECT_DIR, "output", "val_clean.jsonl"))
log(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Load model
log(f"Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForConditionalGeneration.from_pretrained(MODEL_PATH, dtype="float32", trust_remote_code=True)
log(f"Loaded. Total params: {sum(p.numel() for p in model.parameters()):,}")

# Freeze projector
for n, p in model.named_parameters():
    if "mlp_AR" in n or "projector" in n:
        p.stop_gradient = True

# LoRA
log("Applying LoRA r=16 alpha=32...")
lora_config = LoRAConfig(
    r=16, alpha=32,
    target_modules=[".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"],
)
model = LoRAModel(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
log(f"Trainable: {trainable:,}")

# Optimizer
opt = paddle.optimizer.AdamW(
    learning_rate=BASE_LR,
    parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.01,
)

# Training
log(f"Training {EPOCHS} epochs, {len(train_data)} samples, grad_accum={GRAD_ACCUM}")
global_step = 0
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    epoch_loss = 0
    epoch_start = time.time()

    for i in range(0, len(train_data), BATCH_SIZE):
        batch_samples = train_data[i:i + BATCH_SIZE]
        messages_batch = [json.dumps(s["messages"], ensure_ascii=False) for s in batch_samples]

        # Process images
        from PIL import Image
        images = []
        for s in batch_samples:
            img = Image.open(s["images"][0]).convert("RGB")
            w, h = img.size
            scale = MAX_DIM / max(w, h)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            images.append(np.array(img))

        inputs = processor(
            text=messages_batch,
            images=images,
            return_tensors="pd",
            padding=True,
            max_length=2048,
            truncation=True,
        )

        outputs = model(**inputs)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        if (i // BATCH_SIZE + 1) % GRAD_ACCUM == 0:
            paddle.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            opt.clear_gradients()
            global_step += 1

        epoch_loss += loss.item() * GRAD_ACCUM

        if global_step % 50 == 0 and global_step > 0:
            avg_loss = epoch_loss / max(1, i // BATCH_SIZE + 1)
            elapsed = time.time() - epoch_start
            eta = elapsed / max(1, i+1) * (len(train_data) - i) / 60
            log(f"Epoch {epoch+1}/{EPOCHS} step {global_step}: loss={avg_loss:.4f} ETA={eta:.1f}min")

        if global_step % CHECKPOINT_STEPS == 0 and global_step > 0:
            model.eval()
            val_loss = 0
            n_val = 0
            with paddle.no_grad():
                for j in range(min(30, len(val_data))):
                    s = val_data[j]
                    try:
                        img = Image.open(s["images"][0]).convert("RGB")
                        w, h = img.size
                        scale = MAX_DIM / max(w, h)
                        if scale < 1:
                            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                        msgs = [json.dumps(s["messages"], ensure_ascii=False)]
                        vinputs = processor(text=msgs, images=[np.array(img)], return_tensors="pd",
                                           padding=True, max_length=2048, truncation=True)
                        vout = model(**vinputs)
                        val_loss += vout.loss.item()
                        n_val += 1
                    except Exception as e:
                        pass
            val_loss /= max(1, n_val)
            log(f"  Val loss: {val_loss:.4f}")

            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_s{global_step}")
            model.save_pretrained(ckpt_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(OUTPUT_DIR, "checkpoint_best")
                model.save_pretrained(best_path)
                log(f"  Best: {best_path} (loss={best_val_loss:.4f})")
            model.train()

    elapsed = time.time() - epoch_start
    log(f"Epoch {epoch+1} done: {elapsed/60:.1f}min")

final_path = os.path.join(OUTPUT_DIR, "checkpoint_final")
model.save_pretrained(final_path)
log(f"Done. Best val loss: {best_val_loss:.4f}")
