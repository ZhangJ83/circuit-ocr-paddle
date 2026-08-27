"""Local training V3 — PIL Images for processor, proven manual approach."""
import os, sys, json, time, random, numpy as np
from types import ModuleType
_dummy = ModuleType('dummy_flex_checkpoint')
_dummy.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy)

from safetensors import safe_open as _orig_so
def _patched_so(*args, **kwargs):
    result = _orig_so(*args, **kwargs)
    if len(result.keys()) > 0:
        sl = result.get_slice(list(result.keys())[0])
        if not hasattr(type(sl), 'shape'): type(sl).shape = property(lambda self: self.get_shape())
    return result
import safetensors; safetensors.safe_open = _patched_so

import paddle; paddle.set_device("gpu")
if not hasattr(paddle, "LongTensor"): paddle.LongTensor = paddle.Tensor
import paddle.nn.functional as F
if not hasattr(F, "swiglu"): F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.generation import GenerationConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'circuit-ocr-dataset', 'scripts'))
from eval_metrics import compute_all

DATASET_DIR = r"g:/mimo_project/circuit_ocr"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = os.path.join(DATASET_DIR, "checkpoints", "local_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_DIM = 384; EPOCHS = 2; GRAD_ACCUM = 4; GRAD_CLIP = 1.0
CHECKPOINT_STEPS = 400; BASE_LR = 2e-5; WARMUP_STEPS = 100; ETA_MIN = 2e-6
TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log(f"LOCAL-V3: max_dim={MAX_DIM} epochs={EPOCHS} lr={BASE_LR:.0e}")

# Load model
log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS, lora_dropout=0.05)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'): model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)

tp = [p for p in model.parameters() if not p.stop_gradient]
log(f"Trainable: {sum(p.numel() for p in tp):,}")

# Data
with open(os.path.join(DATASET_DIR, "output", "train_local.jsonl"), encoding="utf-8") as f:
    all_data = [json.loads(l) for l in f if l.strip()]
random.shuffle(all_data)
split = int(len(all_data) * 0.9)
train_data = all_data[:split]; val_data = all_data[split:]
total_steps = EPOCHS * len(train_data) // GRAD_ACCUM
log(f"Train: {len(train_data)} Val: {len(val_data)} Steps: {total_steps}")

cosine = paddle.optimizer.lr.CosineAnnealingDecay(BASE_LR, T_max=max(1, total_steps - WARMUP_STEPS), eta_min=ETA_MIN)
lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=WARMUP_STEPS, start_lr=ETA_MIN, end_lr=BASE_LR)
opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

gc = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False)
val_fixed = val_data[:10]

model.train(); t0 = time.time(); gs = 0; el = 0.0; opt.clear_grad()
best_loss = float('inf')

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    log(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    for idx, s in enumerate(train_data):
        try:
            ip = s['images'][0]
            if not os.path.exists(ip): ip = ip.replace("/root/circuit_ocr/", DATASET_DIR + "/")
            img = Image.open(ip).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

            label = s["messages"][1]["content"]
            label_ids = processor.tokenizer.encode(label) + [processor.tokenizer.eos_token_id or 2]
            label_tensor = paddle.to_tensor(label_ids, dtype="int64")

            # Process image as PIL (not numpy!)
            img_inputs = processor.image_processor(images=[img], return_tensors="np")
            igt = img_inputs["image_grid_thw"][0]
            n_copies = max(1, int(igt[1]) * int(igt[2]) // 4)
            prompt = ('<' + '|placeholder|' + '>') * n_copies + 'OCR:'
            # Text processor also needs PIL
            inp = processor(text=[prompt], images=[img], return_tensors="np", padding=True, max_length=2048, truncation=True)

            full_input_ids = paddle.to_tensor(np.concatenate([inp["input_ids"][0], label_ids])).unsqueeze(0)
            prompt_len = inp["input_ids"].shape[1]
            full_mask = paddle.ones([1, full_input_ids.shape[1]], dtype="int64")
            labels_t = paddle.full([1, full_input_ids.shape[1]], -100, dtype="int64")
            labels_t[0, prompt_len:] = label_tensor

            out = model(input_ids=full_input_ids, attention_mask=full_mask,
                       pixel_values=paddle.to_tensor(img_inputs["pixel_values"]),
                       image_grid_thw=paddle.to_tensor(img_inputs["image_grid_thw"]))

            logits = out[0] if isinstance(out, (tuple, list)) else out.logits
            shift_logits = paddle.cast(logits[:, :-1, :], "float32"); shift_labels = labels_t[:, 1:]
            mask = paddle.cast(shift_labels != -100, "float32")
            shift_labels_clean = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
            ce = F.cross_entropy(shift_logits.reshape([-1, shift_logits.shape[-1]]),
                                 shift_labels_clean.reshape([-1]), reduction="none").reshape(shift_labels.shape)
            loss = (ce * mask).sum() / mask.sum().clip(min=1)
            (loss / GRAD_ACCUM).backward(); el += loss.item()
            img.close()

            if (idx + 1) % GRAD_ACCUM == 0 or idx == len(train_data) - 1:
                paddle.nn.utils.clip_grad_norm_(tp, GRAD_CLIP)
                opt.step(); lrs.step(); opt.clear_grad(); gs += 1

                if gs % 20 == 0:
                    eta = (time.time()-t0)/max(1,gs)*(total_steps-gs)/60
                    log(f"E{epoch+1}/{EPOCHS} S{gs}/{total_steps} loss={el/max(1,idx+1):.4f} ETA={eta:.0f}m")

                if gs % CHECKPOINT_STEPS == 0:
                    log(f"Checkpoint S{gs}...")
                    model.eval()
                    lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                    ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_s{gs}.pdparams")
                    paddle.save(lora_dict, ckpt_path)
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        paddle.save(lora_dict, os.path.join(OUTPUT_DIR, "best.pdparams"))
                    log(f"  Saved loss={loss.item():.4f} best={best_loss:.4f}")
                    model.train()
        except Exception as e:
            log(f"  SKIP {idx}: {str(e)[:60]}")
            opt.clear_grad()
            continue

tt = (time.time()-t0)/60
log(f"DONE {tt:.1f}m. Best loss={best_loss:.4f}. Output: {OUTPUT_DIR}")
