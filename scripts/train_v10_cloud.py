"""V10-style training — apply_chat_template + separate tokenization + manual CE loss."""
import os, sys, json, time, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

# Standard patches
sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu, 'get_one_valid_tokenizer_file'):
    mu.get_one_valid_tokenizer_file = lambda d, e: list(mu._filter_valid_tokenizer_files(d, e))

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

import numpy as np; from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms, torch
import transformers.utils.import_utils as tiu
tiu.is_torch_available = lambda: (True, '')
tiu.is_torchvision_available = lambda: (True, '')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import compute_all

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"
PROJECT_DIR = "/root/circuit_ocr"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── Config ──
MAX_DIM = 384; EPOCHS = 2; GRAD_ACCUM = 4; GRAD_CLIP = 1.0
CHECKPOINT_STEPS = 400
BASE_LR = 2e-5; WARMUP_STEPS = 100; ETA_MIN = 2e-6
REPETITION_PENALTY = 1.1

TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]

log(f"V10-STYLE: max_dim={MAX_DIM}, epochs={EPOCHS}, LR={BASE_LR:.0e}, grad_accum={GRAD_ACCUM}")

# ── Load Model ──
log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, load_checkpoint_format="safetensors", dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS, lora_dropout=0.05)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

tp = [p for p in model.parameters() if not p.stop_gradient]
log(f"Trainable: {sum(p.numel() for p in tp):,}")

# ── Data ──
with open(f"{PROJECT_DIR}/output/train_clean.jsonl", encoding="utf-8") as f:
    all_data = [json.loads(l) for l in f if l.strip()]
random.shuffle(all_data)
split = int(len(all_data) * 0.9)
train_data = all_data[:split]
val_data = all_data[split:]
total_steps = EPOCHS * len(train_data) // GRAD_ACCUM
log(f"Train: {len(train_data)}, Val: {len(val_data)}, Steps: {total_steps}")

# ── Optimizer ──
cosine = paddle.optimizer.lr.CosineAnnealingDecay(BASE_LR, T_max=total_steps - WARMUP_STEPS, eta_min=ETA_MIN)
lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=WARMUP_STEPS, start_lr=ETA_MIN, end_lr=BASE_LR)
opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

# ── Quick Inference (V10-style manual greedy decode) ──
def quick_inference(samples, max_tokens=128):
    preds = []
    for s in samples:
        try:
            img_path = s['images'][0]
            if not os.path.exists(img_path): img_path = img_path.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            msgs = [{"role":"user","content":[{"type":"image","image":img_path},{"type":"text","text":"OCR:"}]}]
            inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
            input_ids = inp["input_ids"]
            attn = inp["attention_mask"]
            pv = inp.get("pixel_values")
            igt = inp.get("image_grid_thw")

            gen = []
            with paddle.no_grad():
                for _ in range(max_tokens):
                    out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pv, image_grid_thw=igt)
                    logits = out[0] if isinstance(out, (list, tuple)) else out.logits
                    ntl = logits[:, -1, :]
                    if REPETITION_PENALTY != 1.0 and gen:
                        for tid in set(gen):
                            sc = float(ntl[0, tid])
                            ntl[0, tid] = sc * REPETITION_PENALTY if sc < 0 else sc / REPETITION_PENALTY
                    nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
                    if nt == processor.tokenizer.eos_token_id: break
                    gen.append(nt)
                    input_ids = paddle.concat([input_ids, paddle.to_tensor([[nt]])], axis=1)
                    attn = paddle.concat([attn, paddle.ones([1, 1], dtype=attn.dtype)], axis=1)
            preds.append(processor.tokenizer.decode(gen, skip_special_tokens=True))
            img.close()
        except Exception as e:
            preds.append(f"[ERR:{str(e)[:40]}]")
    return preds

# ── Train ──
model.train()
t0 = time.time()
gs = 0; el = 0.0; opt.clear_grad()
best_loss = float('inf')

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    for idx, s in enumerate(train_data):
        try:
            img_path = s['images'][0]
            if not os.path.exists(img_path): img_path = img_path.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            query = "OCR:"
            label = s["messages"][1]["content"]

            # V10: Separate tokenization with image PATH
            msgs = [{"role":"user","content":[{"type":"image","image":img_path},{"type":"text","text":query}]}]
            pinp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
            prompt_ids = pinp["input_ids"][0]
            prompt_len = prompt_ids.shape[0]

            lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=512)
            label_ids = lt["input_ids"][0]
            eos_t = paddle.to_tensor([processor.tokenizer.eos_token_id], dtype=label_ids.dtype)
            label_ids = paddle.concat([label_ids, eos_t], axis=0)
            label_len = label_ids.shape[0]

            # Concatenate
            full_ids = paddle.concat([prompt_ids, label_ids], axis=0).unsqueeze(0)
            full_mask = paddle.concat([pinp["attention_mask"][0], paddle.ones([label_len], dtype="int64")], axis=0).unsqueeze(0)
            labels = paddle.full([1, prompt_len + label_len], -100, dtype="int64")
            labels[0, prompt_len:] = label_ids

            # Forward
            out = model(input_ids=full_ids, attention_mask=full_mask,
                       pixel_values=pinp["pixel_values"], image_grid_thw=pinp.get("image_grid_thw"))
            logits = out[0] if isinstance(out, (tuple, list)) else out.logits

            # Manual CE loss with shift
            shift_logits = paddle.cast(logits[:, :-1, :], "float32")
            shift_labels = labels[:, 1:]
            mask = paddle.cast(shift_labels != -100, "float32")
            shift_labels_clean = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
            ce = paddle.nn.functional.cross_entropy(
                shift_logits.reshape([-1, shift_logits.shape[-1]]),
                shift_labels_clean.reshape([-1]), reduction="none").reshape(shift_labels.shape)
            loss = (ce * mask).sum() / mask.sum().clip(min=1)

            (loss / GRAD_ACCUM).backward()
            el += loss.item()
            img.close()

            if (idx + 1) % GRAD_ACCUM == 0 or idx == len(train_data) - 1:
                paddle.nn.utils.clip_grad_norm_(tp, GRAD_CLIP)
                opt.step(); lrs.step(); opt.clear_grad()
                gs += 1

                if gs % 20 == 0:
                    eta = (time.time()-t0)/max(1,gs) * (total_steps-gs)/60
                    log(f"E{epoch+1}/{EPOCHS} S{gs}/{total_steps} loss={el/max(1,idx+1):.4f} ETA={eta:.0f}m")

                if gs % CHECKPOINT_STEPS == 0:
                    log(f"Checkpoint S{gs}...")
                    model.eval()
                    lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                    paddle.save(lora_dict, f"/root/circuit_ocr/checkpoints/v10/checkpoint_s{gs}.pdparams")
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        paddle.save(lora_dict, f"/root/circuit_ocr/checkpoints/v10/best.pdparams")
                        log(f"  BEST loss={best_loss:.4f}")

                    # Quick val
                    m_samples = val_data[:5]
                    preds = quick_inference(m_samples)
                    refs = [ms["messages"][1]["content"] for ms in m_samples]
                    m = compute_all(preds, refs, label=f"s{gs}")
                    log(f"  Val: jf1={m['joint_f1']:.4f} CompF1={m['component_f1']:.4f} RepRate={m['repetition_rate']:.2%}")
                    for pi in range(min(2, len(preds))):
                        log(f"  [{pi}] PRED: {preds[pi][:80]}")
                    model.train()
        except Exception as e:
            log(f"  SKIP sample {idx}: {e}")
            opt.clear_grad()
            continue

tt = (time.time()-t0)/60
log(f"DONE {tt:.1f}min. Best loss={best_loss:.4f}")
