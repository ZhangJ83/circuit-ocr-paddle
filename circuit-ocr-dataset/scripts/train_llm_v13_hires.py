"""
V13-HiRes: V100 16GB Optimized Training Script
================================================
Based on V10-Fixed recipe with two targeted upgrades leveraging V100 16GB:
  1. MAX_DIM: 384 → 768 (2× resolution — circuit text is small, needs detail)
  2. LoRA: r=16 → r=32, alpha=64 (2× capacity — more room for domain knowledge)

Everything else IDENTICAL to V10-Fixed (the proven recipe):
  - Same data: V9-Pure (1,554 samples: 1,097 KiCad + 457 Synthetic V3, NO Masala)
  - Same architecture: LLM-Only LoRA (freeze Vision Encoder + Projector)
  - Same LR: 2e-5 with LinearWarmup (100 steps, 2e-6 → 2e-5) + CosineAnnealing
  - Same epochs: 3
  - Same batch: 1, grad_accum=4
  - Same tokenization: Separate Prompt & Label (BPE-safe)
  - Same loss: Manual CE with correct causal shift
  - NO dropout, NO label_smoothing, NO data augmentation (V11 proved these harm)

V1-V12 Lessons Applied:
  ✅ V1-V4:  Freeze Projector (LLM-Only LoRA) — prevents modality collapse
  ✅ V5:     Diversity recovered at 90% — proves architecture direction
  ✅ V6-E6:  Projector LoRA is sole root cause of collapse
  ✅ V8:     3 training bugs discovered and fixed
  ✅ V9:     Removing Masala-CHAI improved results (quality > quantity)
  ✅ V10:    lr=2e-5 critical (5e-4 caused collapse); S600 is sweet spot
  ❌ V11:    52% synthetic + regularization → collapse (avoid both)
  ❌ V12:    Vision LoRA destroys LLM text generation (avoid)

Expected outcome:
  - CompF1: 0.206 → 0.25-0.30 (higher resolution + more capacity)
  - TokenRec: 0.154 → 0.20-0.25
  - NED: 0.803 → 0.75-0.78
  - Risk: S600 may no longer be optimal; monitor all checkpoints
"""
import os, sys, json, time, random

# ── Early patch: flex_checkpoint for Paddle 3.0/3.1 compatibility ──
# MUST run before ANY paddleformers import. Use sys.modules[key]=value
# (not setdefault) to ensure the dummy takes effect.
from types import ModuleType
_dummy_fc = ModuleType('dummy_flex_checkpoint')
_dummy_fc.build_sharded_state_dict = lambda *a, **kw: None
_dummy_fc.shard_weight = lambda *a, **kw: None
_dummy_fc.make_replicated_sharded_weight = lambda *a, **kw: None
sys.modules['paddle.distributed.flex_checkpoint'] = _dummy_fc
sys.modules['paddle.distributed.flex_checkpoint.dcp'] = _dummy_fc
sys.modules['paddle.distributed.flex_checkpoint.dcp.sharded_weight'] = _dummy_fc

# ── Environment setup ──
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

# Allow HF_HOME override from environment
for env_var in ["HF_HOME", "HF_HUB_CACHE", "PADDLE_HOME"]:
    if os.environ.get(env_var):
        os.environ.setdefault(env_var, os.environ[env_var])

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Auto-detect model path (no hardcoded paths!)
def find_model_path():
    """Find PaddleOCR-VL base model. Checks: env var → HF cache → auto-download."""
    env_path = os.environ.get("PADDLE_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    # Check common HF cache locations
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for root, dirs, files in os.walk(hf_home):
        if "PaddleOCR-VL" in root and "snapshots" in root:
            for d in dirs:
                if os.path.exists(os.path.join(root, d, "config.json")):
                    return os.path.join(root, d)
    # Fallback: auto-download
    return "PaddlePaddle/PaddleOCR-VL"

MODEL_PATH = find_model_path()
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
CKPT_DIR = f"{OUTPUT_DIR}/checkpoints_v13_hires"
os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    try: print(f"[{ts}] {msg}", flush=True)
    except: print(f"[{ts}] {msg.encode('ascii','replace').decode('ascii')}", flush=True)

# ═══════════════════════════════════════════════════════════════
# V13 CONFIG — Only 2 changes from V10-Fixed
# ═══════════════════════════════════════════════════════════════
MAX_DIM = 512          # V10: 384 → V13: 512 (33% more resolution; 768 needs >8GB)
EPOCHS = 3             # Same as V10
GRAD_ACCUM = 4          # Same as V10
GRAD_CLIP = 1.0         # Same as V10
CHECKPOINT_STEPS = 200  # Same as V10

# LR: Same as V10 (proven)
BASE_LR = 2e-5
WARMUP_STEPS = 100
ETA_MIN = 2e-6

# Repetition penalty: Same as V10
REPETITION_PENALTY = 1.1

# LoRA: r=16→32, alpha=32→64 (scale=2.0 maintained)
LORA_R = 32            # V10: 16 → V13: 32
LORA_ALPHA = 64        # V10: 32 → V13: 64
LORA_DROPOUT = 0.05    # V13: 0 → V13b: 0.05 (V10 proved this is critical; 0 causes collapse)

# LLM-Only targets: Same as V10
TARGETS = [
    ".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj",
    ".*linear_1", ".*linear_2",
]

# Weight decay: Same as V10
WEIGHT_DECAY = 0.1

# Data: V9-Pure (proven, no Masala, 29.4% synth)
DATA_FILE = "ocr_vl_sft-train-v9-pure.jsonl"

log("=" * 60)
log("TRAINING V13b-HiRes (dropout=0.05 restored, V13 collapsed without it)")
log(f"  Changes from V10: MAX_DIM 384→{MAX_DIM}, LoRA r=16→{LORA_R}, alpha=32→{LORA_ALPHA}, dropout SAME=0.05")
log(f"  Targets: {TARGETS}")
log(f"  Config: max_dim={MAX_DIM}, epochs={EPOCHS}, batch_size=1, grad_accum={GRAD_ACCUM}")
log(f"  LR: {BASE_LR:.0e}→{ETA_MIN:.0e}, warmup={WARMUP_STEPS} steps")
log(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, scale={LORA_ALPHA/LORA_R:.1f}")
log(f"  Weight decay: {WEIGHT_DECAY}, grad_clip: {GRAD_CLIP}")
log(f"  Repetition penalty: {REPETITION_PENALTY}")
log(f"  Dataset: {DATA_FILE} (V9-Pure, 1,554 samples, NO Masala)")
log(f"  Tokenization: Separate Prompt & Label (BPE-safe)")
log(f"  Loss: Manual CE with correct causal shift (no double-shift)")
log(f"  Checkpoints: {CKPT_DIR}")
log("=" * 60)

# ── Load Model ──
log(f"Loading model from: {MODEL_PATH}")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

# ── Apply LoRA ──
lc = LoRAConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGETS,
                lora_dropout=LORA_DROPOUT)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)

trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
lora_count = sum(1 for k, p in model.named_parameters() if 'lora_' in k)
log(f"Trainable parameters: {trainable:,}  LoRA matrices: {lora_count}")
log(f"Trainable ratio: {trainable/908_000_000*100:.2f}%")

# ── Data ──
data_path = f"{DATASET_DIR}/{DATA_FILE}"
if not os.path.exists(data_path):
    log(f"ERROR: Data file not found: {data_path}")
    log(f"  Available files in {DATASET_DIR}:")
    for f in sorted(os.listdir(DATASET_DIR)):
        if f.startswith("ocr_vl_sft-train") and f.endswith(".jsonl"):
            log(f"    {f}")
    sys.exit(1)

with open(data_path, encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data)

# Hold out 10% for validation monitoring
split = int(len(data) * 0.9)
train_data = data[:split]
val_data = data[split:]
total_samples = EPOCHS * len(train_data)
total_steps = total_samples // GRAD_ACCUM
log(f"Training: {len(train_data)} train + {len(val_data)} val × {EPOCHS} epochs = {total_samples} samples = {total_steps} optimizer steps")
log(f"Estimated training time: ~{total_steps * 2.5 / 60:.0f} min on V100 16GB")

# ── Optimizer ──
cosine_decay = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=BASE_LR, T_max=total_steps - WARMUP_STEPS, eta_min=ETA_MIN)
lr_scheduler = paddle.optimizer.lr.LinearWarmup(
    learning_rate=cosine_decay,
    warmup_steps=WARMUP_STEPS, start_lr=ETA_MIN, end_lr=BASE_LR)
opt = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=WEIGHT_DECAY)

log(f"Optimizer: AdamW, lr_schedule: LinearWarmup({WARMUP_STEPS} steps, {ETA_MIN:.0e}→{BASE_LR:.0e}) + CosineAnnealing(→{ETA_MIN:.0e})")

# ── Quick inference helper (Manual Greedy Decoder with repetition_penalty) ──
def quick_inference(samples, max_tokens=120):
    """Monitor inference quality on validation samples during training."""
    preds = []
    for s in samples:
        try:
            from PIL import Image
            img_path = f"{DATASET_DIR}/{s['images'][0].lstrip('./')}"
            if not os.path.exists(img_path):
                preds.append(f"[IMG_NOT_FOUND:{s['images'][0][:40]}]")
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h)
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

            msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
            inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")

            input_ids = inp["input_ids"]
            attention_mask = inp["attention_mask"]
            pixel_values = inp.get("pixel_values")
            image_grid_thw = inp.get("image_grid_thw")

            generated = []
            with paddle.no_grad():
                for _ in range(max_tokens):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw
                    )
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs.logits
                    next_token_logits = logits[:, -1, :]

                    # Apply repetition_penalty
                    if REPETITION_PENALTY != 1.0 and generated:
                        for tid in set(generated):
                            score = next_token_logits[0, tid].item()
                            if score < 0:
                                next_token_logits[0, tid] = score * REPETITION_PENALTY
                            else:
                                next_token_logits[0, tid] = score / REPETITION_PENALTY

                    next_token = int(paddle.argmax(next_token_logits, axis=-1).numpy()[0])
                    if next_token == processor.tokenizer.eos_token_id:
                        break
                    generated.append(next_token)
                    next_tensor = paddle.to_tensor([[next_token]], dtype=input_ids.dtype)
                    input_ids = paddle.concat([input_ids, next_tensor], axis=1)
                    attention_mask = paddle.concat([attention_mask, paddle.ones([1, 1], dtype=attention_mask.dtype)], axis=1)

            resp = processor.tokenizer.decode(generated, skip_special_tokens=True)
            preds.append(resp)
            img.close()
            del img, inp, input_ids, attention_mask, generated; paddle.device.cuda.empty_cache()
        except Exception as e:
            preds.append(f"[ERR:{str(e)[:40]}]")
    return preds

# Monitor on validation set
monitor_samples = val_data[:3]
log(f"Using val split for monitoring ({len(val_data)} held-out samples)")

# ── Train ──
from PIL import Image; from io import BytesIO
model.train()
t0 = time.time()
global_step = 0
history = []
opt.clear_grad()

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    log(f"--- Epoch {epoch+1}/{EPOCHS} ---")

    for idx, sample in enumerate(train_data):
        img_path = f"{DATASET_DIR}/{sample['images'][0].lstrip('./')}"
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0)
        image = Image.open(buf)

        query = sample["messages"][0]["content"]
        label = sample["messages"][1]["content"]

        # === 1. TOKENIZE PROMPT & LABEL SEPARATELY (BPE-safe) ===
        prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
        prompt_inputs = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_len = prompt_ids.shape[0]

        lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=512)
        label_ids = lt["input_ids"][0]
        eos_tensor = paddle.to_tensor([processor.tokenizer.eos_token_id], dtype=label_ids.dtype)
        label_ids = paddle.concat([label_ids, eos_tensor], axis=0)
        label_len = label_ids.shape[0]

        # === 2. CONCATENATE INPUTS & CREATE LABELS ===
        full_input_ids = paddle.concat([prompt_ids, label_ids], axis=0).unsqueeze(0)
        full_attn_mask = paddle.concat([prompt_inputs["attention_mask"][0], paddle.ones([label_len], dtype="int64")], axis=0).unsqueeze(0)

        labels_t = paddle.full([1, prompt_len + label_len], fill_value=-100, dtype="int64")
        labels_t[0, prompt_len:] = label_ids

        # === 3. FORWARD PASS ===
        out = model(
            input_ids=full_input_ids,
            attention_mask=full_attn_mask,
            pixel_values=prompt_inputs["pixel_values"],
            image_grid_thw=prompt_inputs.get("image_grid_thw")
        )
        logits = out[0] if isinstance(out, (tuple, list)) else out.logits

        # === 4. MANUAL CE LOSS WITH CORRECT CAUSAL SHIFT ===
        shift_logits = paddle.cast(logits[:, :-1, :], "float32")
        shift_labels = labels_t[:, 1:]
        mask = paddle.cast(shift_labels != -100, "float32")
        shift_labels_clamped = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
        ce = paddle.nn.functional.cross_entropy(
            shift_logits.reshape([-1, shift_logits.shape[-1]]),
            shift_labels_clamped.reshape([-1]), reduction="none").reshape(shift_labels.shape)
        loss = (ce * mask).sum() / mask.sum().clip(min=1)

        # === 5. BACKWARD + OPTIMIZER UPDATE ===
        scaled_loss = loss / GRAD_ACCUM
        scaled_loss.backward()
        image.close()

        if (idx + 1) % GRAD_ACCUM == 0 or idx == len(train_data) - 1:
            paddle.nn.utils.clip_grad_norm_([p for p in model.parameters() if not p.stop_gradient], max_norm=GRAD_CLIP)
            opt.step()
            lr_scheduler.step()
            opt.clear_grad()
            global_step += 1

            if global_step % 20 == 0 or global_step == 1:
                elapsed = (time.time()-t0)/60
                eta = (elapsed/global_step*total_steps - elapsed) if global_step > 0 else 0
                log(f"  [S{global_step}/{total_steps}] loss={loss.item():.4f} lr={opt.get_lr():.2e} elapsed={elapsed:.0f}m ETA={eta:.0f}m VRAM={paddle.device.cuda.memory_allocated()/1024**3:.1f}GB")
                history.append({"step": global_step, "loss": float(loss.item()), "lr": opt.get_lr()})

            # ── Checkpoint Save & Monitor ──
            if global_step % CHECKPOINT_STEPS == 0:
                log(f"--- Checkpoint at S{global_step} ---")
                model.eval()

                # Save LoRA weights
                lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                ckpt_path = f"{CKPT_DIR}/lora_s{global_step}.pdparams"
                paddle.save(lora_dict, ckpt_path)
                total_bytes = sum(v.numel() * v.element_size() for v in lora_dict.values())
                log(f"  Saved: {ckpt_path} ({len(lora_dict)} matrices, {total_bytes/1024**2:.1f} MB)")

                # Monitor inference quality
                log("  Running quick validation inference...")
                preds = quick_inference(monitor_samples)
                for m_idx, pred in enumerate(preds):
                    ref = monitor_samples[m_idx]["messages"][1]["content"][:100]
                    log(f"    Sample {m_idx} Pred: {repr(pred[:120])}")
                    log(f"    Sample {m_idx} Ref:  {repr(ref)}")

                unique_preds = len(set(preds))
                log(f"    Diversity: {unique_preds}/{len(preds)}")

                # Save as latest best
                best_path = f"{OUTPUT_DIR}/lora_best_v13_fp16.pdparams"
                paddle.save(lora_dict, best_path)
                log(f"  Also saved as best/latest: {best_path}")

                paddle.device.cuda.empty_cache()
                model.train()

total_min = (time.time()-t0)/60
log(f"\nTraining done in {total_min:.0f}m")

# ── Save Final Model ──
model.eval()
lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
final_path = f"{OUTPUT_DIR}/lora_v13_final_fp16.pdparams"
paddle.save(lora_dict, final_path)
log(f"Final model saved: {final_path} ({len(lora_dict)} matrices)")

# ── Final Report ──
log("=" * 60)
log("TRAINING V13-HiRes SUMMARY")
log(f"  Changes from V10: MAX_DIM 384→{MAX_DIM}, LoRA r=16→{LORA_R}, alpha=32→{LORA_ALPHA}")
log(f"  LR: {BASE_LR:.0e} with {WARMUP_STEPS}-step warmup → Cosine to {ETA_MIN:.0e}")
log(f"  Repetition penalty: {REPETITION_PENALTY}")
log(f"  Total steps: {total_steps}")
log(f"  Total time: {total_min:.0f}m")
log(f"  Final model: {final_path}")
log(f"  Checkpoints: {CKPT_DIR}")
if history:
    log(f"  Initial loss: {history[0]['loss']:.4f}")
    log(f"  Final loss: {history[-1]['loss']:.4f}")
    log(f"  Loss reduction: {history[0]['loss'] - history[-1]['loss']:.4f}")
log("=" * 60)

# ── Save training history ──
with open(f"{CKPT_DIR}/training_history_v13.json", "w") as f:
    json.dump({"history": history, "total_steps": total_steps, "total_min": total_min,
               "config": {"base_lr": BASE_LR, "warmup_steps": WARMUP_STEPS, "eta_min": ETA_MIN,
                          "repetition_penalty": REPETITION_PENALTY, "max_dim": MAX_DIM,
                          "epochs": EPOCHS, "grad_accum": GRAD_ACCUM,
                          "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
                          "weight_decay": WEIGHT_DECAY, "data_file": DATA_FILE}}, f)

log("Training V13-HiRes complete!")
log("")
log("Next steps:")
log("  1. Evaluate all checkpoints: python eval_benchmark_v3.py --data_path ../ocr_vl_sft-test-easy50-pure.jsonl --lora_checkpoint <path>")
log("  2. Find optimal checkpoint (likely S600-S800 range)")
log("  3. Compare with V10 S600: CompF1, TokenRec, NED, RepRate, Diversity")
log("  4. If CompF1 > 0.25: proceed to V13b (more data)")
log("  5. If CompF1 < 0.20: revert to V10, try different changes")