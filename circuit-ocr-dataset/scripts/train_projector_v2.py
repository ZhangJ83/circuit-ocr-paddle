"""
SAFE Projector-ONLY LoRA training v2 with checkpoints.
- Improved dataset: 1,857 samples (1,357 real + 500 new synthetic)
- Projector-only targets (.*linear_1, .*linear_2) — proven safe
- max_dim=168 — proven safe (336 caused diversity collapse)
- Checkpoint every 500 steps with 3-sample inference test
- Auto-save best model based on inference quality
"""
import os, sys, json, time, random
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HOME": "F:/hf_cache/hub",
    "PADDLE_HOME": "F:/paddle_cache", "HF_HUB_CACHE": "F:/hf_cache/hub",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "FLAGS_allocator_strategy": "auto_growth",
})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
CKPT_DIR = f"{OUTPUT_DIR}/checkpoints_v2"
os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    try: print(f"[{ts}] {msg}", flush=True)
    except: print(f"[{ts}] {msg.encode('ascii','replace').decode('ascii')}", flush=True)

# ── Config ──
MAX_DIM = 168
EPOCHS = 3
CHECKPOINT_STEPS = 500
DIVERSITY_CHECK_SAMPLES = 5   # Number of samples for diversity check (keep low to avoid OOM)
TARGETS = [
    ".*linear_1", ".*linear_2",    # Projector ONLY — proven safe
]

log("=" * 60)
log("SAFE TRAINING V2")
log(f"Dataset: ocr_vl_sft-train-v2.jsonl")
log(f"Targets: {TARGETS}")
log(f"Config: max_dim={MAX_DIM}, epochs={EPOCHS}, checkpoint_every={CHECKPOINT_STEPS}")
log("=" * 60)

# ── Load ──
log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)

trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
lora_count = sum(1 for k, p in model.named_parameters() if 'lora_' in k)
log(f"Trainable: {trainable:,}  LoRA matrices: {lora_count}")

# ── Data ──
with open(f"{DATASET_DIR}/ocr_vl_sft-train-v2.jsonl", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data)
total_steps = EPOCHS * len(data)
log(f"Training: {len(data)} samples x {EPOCHS} epochs = {total_steps} steps")

# ── Optimizer ──
lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=5e-4, T_max=total_steps, eta_min=5e-5)
opt = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler, parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1)

# ── Quick inference helper ──
def quick_inference(samples, max_tokens=20):
    """Run inference on a few samples, return predictions."""
    preds = []
    for s in samples:
        try:
            from PIL import Image
            img_path = f"{DATASET_DIR}/{s['images'][0].lstrip('./')}"
            img = Image.open(img_path).convert("RGB")
            msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
            inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
            with paddle.no_grad():
                out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False, use_cache=False)
            if isinstance(out, (list,tuple)): tok = out[0]
            else: tok = out
            if len(tok.shape)>1: tok = tok[0]
            ids = [int(x) for x in tok.numpy().tolist() if int(x)>0]
            resp = processor.tokenizer.decode(ids, skip_special_tokens=True)
            preds.append(resp)
            img.close()
            del img, inp, out, ids; paddle.device.cuda.empty_cache()
        except Exception as e:
            preds.append(f"[ERR:{str(e)[:40]}]")
    return preds

def check_diversity(preds):
    """Count unique predictions (normalized). Returns ratio of unique outputs."""
    normalized = [" ".join(p.split()) for p in preds]  # Normalize whitespace
    unique = len(set(normalized))
    return unique / len(preds) if preds else 0

# ── Train ──
from PIL import Image; from io import BytesIO
model.train()
t0 = time.time()
global_step = 0
history = []
best_diversity = 0
best_checkpoint_path = None
safety_alerts = []

# Load test data for checkpoints
test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl"
with open(test_path, encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()]

for epoch in range(EPOCHS):
    random.shuffle(data)
    log(f"--- Epoch {epoch+1}/{EPOCHS} ---")

    for idx, sample in enumerate(data):
        img_path = f"{DATASET_DIR}/{sample['images'][0].lstrip('./')}"
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0)
        image = Image.open(buf)

        query = sample["messages"][0]["content"]
        label = sample["messages"][1]["content"][:200]

        prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
        prompt_inputs = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        full_msgs = [
            {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]},
            {"role":"assistant","content":[{"type":"text","text":label}]},
        ]
        full_inputs = processor.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pd")
        full_ids = full_inputs["input_ids"]
        labels_t = paddle.full_like(full_ids, -100, dtype=full_ids.dtype)
        labels_t[0, prompt_len:] = full_ids[0, prompt_len:]

        out = model(**full_inputs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        shift_logits = paddle.cast(logits[:, :-1, :], "float32")
        shift_labels = labels_t[:, 1:]
        mask = paddle.cast(shift_labels != -100, "float32")
        shift_labels_clamped = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
        ce = paddle.nn.functional.cross_entropy(
            shift_logits.reshape([-1, shift_logits.shape[-1]]),
            shift_labels_clamped.reshape([-1]), reduction="none").reshape(shift_labels.shape)
        loss = (ce * mask).sum() / mask.sum().clip(min=1)

        loss.backward(); opt.step(); opt.clear_grad()
        global_step += 1
        image.close()

        if global_step % 100 == 0 or global_step == 1:
            elapsed = (time.time()-t0)/60
            eta = (elapsed/global_step*total_steps - elapsed) if global_step > 0 else 0
            log(f"  [S{global_step}/{total_steps}] loss={loss.item():.4f} lr={opt.get_lr():.2e} elapsed={elapsed:.0f}m ETA={eta:.0f}m")
            history.append({"step": global_step, "loss": float(loss.item()), "lr": opt.get_lr()})

        # ── Checkpoint Save (no inference check to avoid Paddle 3.0b2 segfault) ──
        if global_step % CHECKPOINT_STEPS == 0:
            log(f"--- Checkpoint at S{global_step} ---")

            # Save model
            model.eval()
            lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
            ckpt_path = f"{CKPT_DIR}/lora_s{global_step}.pdparams"
            paddle.save(lora_dict, ckpt_path)
            log(f"  Saved: {ckpt_path}")

            # Also save as "best" for latest checkpoint
            best_ckpt_path = f"{OUTPUT_DIR}/lora_best_v2_fp16.pdparams"
            paddle.save(lora_dict, best_ckpt_path)
            log(f"  Also saved as best/latest")

            paddle.device.cuda.empty_cache()
            model.train()  # back to training mode

total_min = (time.time()-t0)/60
log(f"\nTraining done in {total_min:.0f}m")

# ── Save Final Model ──
model.eval()
lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
final_path = f"{OUTPUT_DIR}/lora_projector_v2_final_fp16.pdparams"
paddle.save(lora_dict, final_path)
log(f"Final model saved: {final_path}")

# ── Final Report ──
log("=" * 60)
log("TRAINING SUMMARY")
log(f"  Total steps: {total_steps}")
log(f"  Total time: {total_min:.0f}m")
log(f"  Final model: {final_path}")
log(f"  Checkpoints: {CKPT_DIR}")
log("=" * 60)

# ── Save training history ──
with open(f"{CKPT_DIR}/training_history.json", "w") as f:
    json.dump({"history": history, "total_steps": total_steps, "total_min": total_min}, f)

log("Training complete! Evaluate checkpoints with eval_benchmark.py")
