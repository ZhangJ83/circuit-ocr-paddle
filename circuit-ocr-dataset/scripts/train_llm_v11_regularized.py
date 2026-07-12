"""
V11-Regularized: Phase 2 Training Script
=========================================
Based on V10-Fixed, with regularization additions:
  1. LoRA dropout=0.1 (prevents overfitting that killed S800)
  2. Label smoothing=0.05 (softens targets, reduces overconfidence)
  3. Data augmentation: random rotation ±3°, brightness ±10%, contrast ±10%
  4. Early stopping: monitor CompF1 on easy50 validation every 200 steps
  5. Expanded dataset: V9-Pure (1554) + Synthetic-V4 (~1500)

Key unchanged from V10-Fixed:
  - SEPARATE tokenization (BPE-safe)
  - Manual CE loss with correct causal shift
  - MAX_DIM=384, EPOCHS=3, GRAD_ACCUM=4, GRAD_CLIP=1.0
  - LR=2e-5 with LinearWarmup(100) + CosineAnnealing
  - LoRA r=16, alpha=32, LLM attention + Projector
"""
import os, sys, json, time, random, re
from PIL import Image, ImageEnhance
from io import BytesIO

# ── Early patch: flex_checkpoint for Paddle 3.1.0 compatibility ──
from types import ModuleType
_dummy_fc = ModuleType('dummy_flex_checkpoint')
_dummy_fc.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy_fc)

local_hf_cache = "F:/hf_cache/hub"
local_paddle_cache = "F:/paddle_cache"
if os.path.exists(local_hf_cache):
    os.environ.setdefault("HF_HOME", local_hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", local_hf_cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if os.path.exists(local_paddle_cache):
    os.environ.setdefault("PADDLE_HOME", local_paddle_cache)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = os.environ.get("PADDLE_MODEL_PATH", "PaddlePaddle/PaddleOCR-VL")

OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
CKPT_DIR = f"{OUTPUT_DIR}/checkpoints_v11_regularized"
os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    try: print(f"[{ts}] {msg}", flush=True)
    except: print(f"[{ts}] {msg.encode('ascii','replace').decode('ascii')}", flush=True)

# ── Config ──
MAX_DIM = 384
EPOCHS = 3
GRAD_ACCUM = 4
GRAD_CLIP = 1.0
CHECKPOINT_STEPS = 200

BASE_LR = 2e-5
WARMUP_STEPS = 100
ETA_MIN = 2e-6

REPETITION_PENALTY = 1.1

# ── NEW: Regularization ──
LORA_DROPOUT = 0.1          # LoRA dropout (was 0 in V10)
LABEL_SMOOTHING = 0.05      # Label smoothing for CE loss
AUG_ROTATION_DEG = 3        # Random rotation ±3°
AUG_BRIGHTNESS = 0.10       # Random brightness ±10%
AUG_CONTRAST = 0.10         # Random contrast ±10%

# ── NEW: Early stopping ──
EARLY_STOP_PATIENCE = 3     # Stop if no CompF1 improvement for 3 checks
EARLY_STOP_MIN_DELTA = 0.005  # Minimum improvement to count

# WIDE targets: LLM attention + Projector (same as V10-Fixed)
TARGETS = [
    ".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj",
    ".*linear_1", ".*linear_2",
]

log("=" * 60)
log("TRAINING V11-REGULARIZED (Phase 2: Data Diversity + Regularization)")
log(f"  Targets: {TARGETS}")
log(f"  Config: max_dim={MAX_DIM}, epochs={EPOCHS}, LR={BASE_LR:.0e}→{ETA_MIN:.0e}")
log(f"  NEW: lora_dropout={LORA_DROPOUT}, label_smoothing={LABEL_SMOOTHING}")
log(f"  NEW: augmentation rotation=±{AUG_ROTATION_DEG}°, brightness=±{AUG_BRIGHTNESS}, contrast=±{AUG_CONTRAST}")
log(f"  NEW: early_stopping patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
log(f"  Dataset: V9-Pure (1554) + Synthetic-V4 (~1500)")
log("=" * 60)

# ── Load Model ──
log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

# LoRA with dropout
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS, lora_dropout=LORA_DROPOUT)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)

trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
lora_count = sum(1 for k, p in model.named_parameters() if 'lora_' in k)
log(f"Trainable parameters: {trainable:,}  LoRA matrices: {lora_count}")

# ── Data: V9-Pure + Synthetic-V4 ──
data = []
for jsonl_name in ["ocr_vl_sft-train-v9-pure.jsonl", "ocr_vl_sft-synthetic-v4.jsonl"]:
    path = f"{DATASET_DIR}/{jsonl_name}"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            chunk = [json.loads(l) for l in f if l.strip()]
        data.extend(chunk)
        log(f"  Loaded {len(chunk)} samples from {jsonl_name}")
    else:
        log(f"  WARNING: {jsonl_name} not found, skipping")

random.shuffle(data)
total_samples = EPOCHS * len(data)
total_steps = total_samples // GRAD_ACCUM
log(f"Training: {len(data)} samples x {EPOCHS} epochs = {total_samples} samples = {total_steps} optimizer steps")

# ── Optimizer ──
cosine_decay = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=BASE_LR, T_max=total_steps - WARMUP_STEPS, eta_min=ETA_MIN)
lr_scheduler = paddle.optimizer.lr.LinearWarmup(
    learning_rate=cosine_decay,
    warmup_steps=WARMUP_STEPS, start_lr=ETA_MIN, end_lr=BASE_LR)
opt = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler, parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1)

log(f"Optimizer: AdamW, lr_schedule: LinearWarmup({WARMUP_STEPS}, {ETA_MIN:.0e}→{BASE_LR:.0e}) + Cosine(→{ETA_MIN:.0e})")

# ── Data augmentation ──
def augment_image(image):
    """Apply random rotation ±3°, brightness ±10%, contrast ±10%."""
    if random.random() < 0.5:
        angle = random.uniform(-AUG_ROTATION_DEG, AUG_ROTATION_DEG)
        image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-AUG_CONTRAST, AUG_CONTRAST)
        image = ImageEnhance.Contrast(image).enhance(factor)
    return image

# ── Component extraction for early stopping ──
RE_COMPONENT = re.compile(r'\b([A-Z]+)\d+\b')

def compute_comp_f1(pred_text, ref_text):
    """Simple component F1 for validation monitoring."""
    pred_comps = set(RE_COMPONENT.findall(pred_text))
    ref_comps = set(RE_COMPONENT.findall(ref_text))
    if not pred_comps and not ref_comps:
        return 1.0
    if not pred_comps or not ref_comps:
        return 0.0
    intersection = pred_comps & ref_comps
    precision = len(intersection) / len(pred_comps)
    recall = len(intersection) / len(ref_comps)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ── Quick inference helper (Manual Greedy Decoder with repetition_penalty) ──
def quick_inference(samples, max_tokens=60):
    preds = []
    for s in samples:
        try:
            img_path = f"{DATASET_DIR}/{s['images'][0].lstrip('./')}"
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

# ── Load validation data for early stopping ──
val_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50-pure.jsonl"
if os.path.exists(val_path):
    with open(val_path, encoding="utf-8") as f:
        val_data = [json.loads(l) for l in f if l.strip()]
    log(f"Validation: {len(val_data)} samples from easy50-pure")
    monitor_samples = val_data[:3]
else:
    val_path = f"{DATASET_DIR}/ocr_vl_sft-val-v9-pure.jsonl"
    with open(val_path, encoding="utf-8") as f:
        val_data = [json.loads(l) for l in f if l.strip()]
    log(f"Validation: {len(val_data)} samples from val-v9-pure")
    monitor_samples = val_data[:3]

# ── Train ──
model.train()
t0 = time.time()
global_step = 0
history = []
opt.clear_grad()

# Early stopping state
best_comp_f1 = -1.0
patience_counter = 0
best_checkpoint_path = None

for epoch in range(EPOCHS):
    random.shuffle(data)
    log(f"--- Epoch {epoch+1}/{EPOCHS} ---")

    for idx, sample in enumerate(data):
        img_path = f"{DATASET_DIR}/{sample['images'][0].lstrip('./')}"
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")

        # ── NEW: Data augmentation ──
        image = augment_image(image)

        w, h = image.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0)
        image = Image.open(buf)

        query = sample["messages"][0]["content"]
        label = sample["messages"][1]["content"]

        # === 1. TOKENIZE PROMPT & LABEL SEPARATELY ===
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

        # === 4. MANUAL CE LOSS WITH CORRECT SHIFT + LABEL SMOOTHING ===
        shift_logits = paddle.cast(logits[:, :-1, :], "float32")
        shift_labels = labels_t[:, 1:]
        mask = paddle.cast(shift_labels != -100, "float32")

        if LABEL_SMOOTHING > 0:
            vocab_size = shift_logits.shape[-1]
            shift_labels_clamped = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
            # Standard CE
            ce = paddle.nn.functional.cross_entropy(
                shift_logits.reshape([-1, vocab_size]),
                shift_labels_clamped.reshape([-1]), reduction="none").reshape(shift_labels.shape)
            # Smooth CE (uniform prior)
            log_probs = paddle.nn.functional.log_softmax(shift_logits, axis=-1)
            smooth_ce = -log_probs.mean(axis=-1)
            loss = ((1 - LABEL_SMOOTHING) * ce + LABEL_SMOOTHING * smooth_ce)
            loss = (loss * mask).sum() / mask.sum().clip(min=1)
        else:
            shift_labels_clamped = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
            ce = paddle.nn.functional.cross_entropy(
                shift_logits.reshape([-1, shift_logits.shape[-1]]),
                shift_labels_clamped.reshape([-1]), reduction="none").reshape(shift_labels.shape)
            loss = (ce * mask).sum() / mask.sum().clip(min=1)

        # === 5. BACKWARD + OPTIMIZER UPDATE ===
        scaled_loss = loss / GRAD_ACCUM
        scaled_loss.backward()
        image.close()

        if (idx + 1) % GRAD_ACCUM == 0 or idx == len(data) - 1:
            paddle.nn.utils.clip_grad_norm_([p for p in model.parameters() if not p.stop_gradient], max_norm=GRAD_CLIP)
            opt.step()
            lr_scheduler.step()
            opt.clear_grad()
            global_step += 1

            if global_step % 20 == 0 or global_step == 1:
                elapsed = (time.time()-t0)/60
                eta = (elapsed/global_step*total_steps - elapsed) if global_step > 0 else 0
                log(f"  [S{global_step}/{total_steps}] loss={loss.item():.4f} lr={opt.get_lr():.2e} elapsed={elapsed:.0f}m ETA={eta:.0f}m")
                history.append({"step": global_step, "loss": float(loss.item()), "lr": opt.get_lr()})

            # ── Checkpoint Save & Early Stopping Monitor ──
            if global_step % CHECKPOINT_STEPS == 0:
                log(f"--- Checkpoint at S{global_step} ---")
                model.eval()

                # Save LoRA weights
                lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                ckpt_path = f"{CKPT_DIR}/lora_s{global_step}.pdparams"
                paddle.save(lora_dict, ckpt_path)
                log(f"  Saved: {ckpt_path} ({len(lora_dict)} matrices)")

                # Monitor inference quality
                log("  Running validation inference...")
                preds = quick_inference(monitor_samples)
                for m_idx, pred in enumerate(preds):
                    ref = monitor_samples[m_idx]["messages"][1]["content"][:80]
                    log(f"    Sample {m_idx} Pred: {repr(pred[:100])}")
                    log(f"    Sample {m_idx} Ref:  {repr(ref)}")

                unique_preds = len(set(preds))
                log(f"    Diversity: {unique_preds}/{len(preds)}")

                # ── NEW: Full validation CompF1 for early stopping ──
                log(f"  Computing CompF1 on full validation set ({len(val_data)} samples)...")
                val_preds = quick_inference(val_data)
                val_f1s = []
                for vp, vs in zip(val_preds, val_data):
                    ref_text = vs["messages"][1]["content"]
                    f1 = compute_comp_f1(vp, ref_text)
                    val_f1s.append(f1)
                avg_val_f1 = sum(val_f1s) / len(val_f1s) if val_f1s else 0.0
                log(f"  Validation CompF1: {avg_val_f1:.4f} (best so far: {max(best_comp_f1, avg_val_f1):.4f})")

                if avg_val_f1 > best_comp_f1 + EARLY_STOP_MIN_DELTA:
                    best_comp_f1 = avg_val_f1
                    patience_counter = 0
                    best_path = f"{OUTPUT_DIR}/lora_best_v11_regularized_fp16.pdparams"
                    paddle.save(lora_dict, best_path)
                    best_checkpoint_path = ckpt_path
                    log(f"  ★ New best CompF1={best_comp_f1:.4f}! Saved: {best_path}")
                else:
                    patience_counter += 1
                    log(f"  No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

                paddle.device.cuda.empty_cache()
                model.train()

                if patience_counter >= EARLY_STOP_PATIENCE:
                    log(f"  EARLY STOPPING triggered at S{global_step}!")
                    log(f"  Best CompF1: {best_comp_f1:.4f} at {best_checkpoint_path}")
                    break

    if patience_counter >= EARLY_STOP_PATIENCE:
        break

total_min = (time.time()-t0)/60
log(f"\nTraining done in {total_min:.0f}m")

# ── Save Final Model ──
model.eval()
lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
final_path = f"{OUTPUT_DIR}/lora_v11_regularized_final_fp16.pdparams"
paddle.save(lora_dict, final_path)
log(f"Final model saved: {final_path} ({len(lora_dict)} matrices)")

# ── Final Report ──
log("=" * 60)
log("TRAINING V11-REGULARIZED (Phase 2) SUMMARY")
log(f"  LR: {BASE_LR:.0e} with {WARMUP_STEPS}-step warmup → Cosine to {ETA_MIN:.0e}")
log(f"  lora_dropout: {LORA_DROPOUT}, label_smoothing: {LABEL_SMOOTHING}")
log(f"  augmentation: rotation=±{AUG_ROTATION_DEG}°, brightness=±{AUG_BRIGHTNESS}, contrast=±{AUG_CONTRAST}")
log(f"  Total steps: {global_step}")
log(f"  Total time: {total_min:.0f}m")
log(f"  Best validation CompF1: {best_comp_f1:.4f}")
log(f"  Best checkpoint: {best_checkpoint_path}")
log(f"  Final model: {final_path}")
if history:
    log(f"  Initial loss: {history[0]['loss']:.4f}")
    log(f"  Final loss: {history[-1]['loss']:.4f}")
log("=" * 60)

# ── Save training history ──
with open(f"{CKPT_DIR}/training_history_v11_regularized.json", "w") as f:
    json.dump({"history": history, "total_steps": global_step, "total_min": total_min,
               "best_val_comp_f1": best_comp_f1, "best_checkpoint": best_checkpoint_path,
               "config": {"base_lr": BASE_LR, "warmup_steps": WARMUP_STEPS, "eta_min": ETA_MIN,
                          "lora_dropout": LORA_DROPOUT, "label_smoothing": LABEL_SMOOTHING,
                          "aug_rotation_deg": AUG_ROTATION_DEG, "aug_brightness": AUG_BRIGHTNESS,
                          "aug_contrast": AUG_CONTRAST,
                          "max_dim": MAX_DIM, "epochs": EPOCHS, "grad_accum": GRAD_ACCUM}}, f)

log("Training V11-Regularized complete!")
