"""
V12-Stage2: Phase 3 Vision LoRA Retraining Script
===================================================
Hypothesis: V10-Fixed's vision LoRA (r=16) was undertrained because LLM LoRA
adapted around poor vision features, creating competing gradients. By freezing
the already-trained LLM LoRA and retraining vision LoRA from scratch with a
clean gradient signal, the vision encoder should learn better features.

Strategy:
  1. Load base model + apply LoRA with SAME targets as V10-Fixed (r=16, alpha=32)
  2. Load V10-Fixed S600 checkpoint → all LoRA weights initialized
  3. Re-initialize VISION LoRA weights to random (discard V10-Fixed vision LoRA)
  4. Freeze LLM LoRA + Projector LoRA (stop_gradient=True)
  5. Train ONLY vision LoRA at higher LR (1e-4) for 5 epochs
  6. Same data: V9-Pure (1554 samples)

Key differences from V10-Fixed:
  - Starts FROM V10-Fixed S600 (not from scratch)
  - Vision LoRA: random init (not V10-Fixed weights)
  - LLM LoRA + Projector: FROZEN (not trainable)
  - LR: 1e-4 (5x higher than V10-Fixed's 2e-5)
  - Epochs: 5 (vs 3)
  - Optimizer: only vision LoRA params (~162 params vs ~310)

Expected outcome:
  - Success: CompF1 > 0.30 → proceed to Phase 4
  - Marginal: CompF1 0.20-0.30 → try vision MLP LoRA or increase rank
  - Stagnation: CompF1 ≈ 0.20 → Phase 3B (data-centric)
  - Degradation: CompF1 < 0.15 → approach failing

Usage:
  python train_llm_v12_stage2_vision_lora.py                  # default 384px
  python train_llm_v12_stage2_vision_lora.py --max_dim 448    # 448px (higher VRAM)
  python train_llm_v12_stage2_vision_lora.py --vision_rank 8  # r=8 instead of r=16
"""

import os, sys, json, time, random, argparse

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
CKPT_DIR = f"{OUTPUT_DIR}/checkpoints_v12_stage2"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Path to V10-Fixed S600 (best checkpoint, frozen LLM + Projector source) ──
V10_S600_PATH = f"{OUTPUT_DIR}/checkpoints_v10_fixed/lora_s600.pdparams"


def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except:
        print(f"[{ts}] {msg.encode('ascii','replace').decode('ascii')}", flush=True)


# ── Config ──
parser = argparse.ArgumentParser(description="V12 Stage2: Vision LoRA Retraining")
parser.add_argument("--max_dim", type=int, default=384,
                    help="Max image dimension (384=known safe, 448=higher VRAM)")
parser.add_argument("--vision_rank", type=int, default=None,
                    help="Vision LoRA rank override (default: same as V10-Fixed r=16)")
parser.add_argument("--vision_lr", type=float, default=1e-4,
                    help="Learning rate for vision LoRA")
parser.add_argument("--epochs", type=int, default=5,
                    help="Number of training epochs")
parser.add_argument("--v10_checkpoint", type=str, default=V10_S600_PATH,
                    help="Path to V10-Fixed checkpoint for LLM+Projector weights")
args = parser.parse_args()

MAX_DIM = args.max_dim
VISION_LR = args.vision_lr
EPOCHS = args.epochs
GRAD_ACCUM = 4
GRAD_CLIP = 1.0
CHECKPOINT_STEPS = 200

# LR schedule: LinearWarmup(100) + CosineAnnealing
WARMUP_STEPS = 100
ETA_MIN = VISION_LR * 0.1  # 1e-5 when VISION_LR=1e-4

REPETITION_PENALTY = 1.1

# V10-Fixed LoRA config (must match for checkpoint loading)
LLM_RANK = 16
LLM_ALPHA = 32

# Vision LoRA rank: default same as LLM (r=16), can override to r=4/r=8
VISION_RANK = args.vision_rank if args.vision_rank else LLM_RANK
VISION_ALPHA = VISION_RANK * 2  # Keep alpha/r = 2.0 ratio

# Same TARGETS as V10-Fixed (must match for checkpoint loading)
TARGETS = [
    ".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj",
    ".*linear_1", ".*linear_2",
]

# Vision key prefix for identification
VISION_KEY_PREFIX = "model.visual.vision_model.encoder.layers."

log("=" * 70)
log("TRAINING V12-STAGE2: Vision LoRA Retraining (Phase 3)")
log(f"  Hypothesis: Competing gradients degraded V10-Fixed vision LoRA")
log(f"  Strategy: Freeze LLM+Projector, retrain vision LoRA from scratch")
log(f"  V10-Fixed checkpoint: {args.v10_checkpoint}")
log(f"  Vision rank: {VISION_RANK} (LLM rank: {LLM_RANK})")
log(f"  Vision LR: {VISION_LR:.0e} (5x V10-Fixed LLM LR)")
log(f"  Resolution: {MAX_DIM}px")
log(f"  Epochs: {EPOCHS}")
log(f"  Data: V9-Pure (1554 samples)")
log("=" * 70)

# ── Load Base Model ──
log("Loading base model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

# Apply LoRA with V10-Fixed config (must match for checkpoint loading)
lc = LoRAConfig(r=LLM_RANK, lora_alpha=LLM_ALPHA, target_modules=TARGETS)
model = LoRAModel(model, lc)
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ── Load V10-Fixed S600 Checkpoint ──
log(f"Loading V10-Fixed checkpoint: {args.v10_checkpoint}")
if not os.path.exists(args.v10_checkpoint):
    raise FileNotFoundError(f"V10-Fixed checkpoint not found: {args.v10_checkpoint}")

v10_state = paddle.load(args.v10_checkpoint)
log(f"  Checkpoint has {len(v10_state)} keys")

model_lora_params = {k: p for k, p in model.named_parameters() if 'lora_' in k}
log(f"  Model has {len(model_lora_params)} LoRA params")

loaded = 0
skipped = 0
for ckpt_key, ckpt_value in v10_state.items():
    if ckpt_key in model_lora_params:
        p = model_lora_params[ckpt_key]
        ckpt_tensor = paddle.cast(ckpt_value, p.dtype)
        p.set_value(ckpt_tensor)
        loaded += 1
    else:
        skipped += 1
        if skipped <= 5:
            log(f"  SKIP (no match): {ckpt_key}")

log(f"  Loaded {loaded}/{len(v10_state)} LoRA params (skipped={skipped})")

# ── Identify Vision vs LLM vs Projector LoRA params ──
vision_lora_params = {}
llm_lora_params = {}
projector_lora_params = {}

for name, param in model.named_parameters():
    if 'lora_' not in name:
        continue
    if VISION_KEY_PREFIX in name:
        vision_lora_params[name] = param
    elif 'mlp_AR' in name:
        projector_lora_params[name] = param
    else:
        llm_lora_params[name] = param

log(f"\nLoRA parameter breakdown:")
log(f"  Vision:    {len(vision_lora_params)} params")
log(f"  LLM:       {len(llm_lora_params)} params")
log(f"  Projector: {len(projector_lora_params)} params")

# ── Re-initialize Vision LoRA weights ──
log("\nRe-initializing vision LoRA weights...")
reinit_count = 0
for name, param in vision_lora_params.items():
    if 'lora_A' in name:
        # Kaiming uniform init (standard for LoRA A)
        fan_in = param.shape[0]
        bound = np.sqrt(6.0 / fan_in)
        new_val = paddle.uniform(param.shape, min=-bound, max=bound, dtype=param.dtype)
        param.set_value(new_val)
        reinit_count += 1
    elif 'lora_B' in name:
        # Zero init (standard for LoRA B → ΔW = BA = 0 at start)
        param.set_value(paddle.zeros(param.shape, dtype=param.dtype))
        reinit_count += 1

log(f"  Re-initialized {reinit_count} vision LoRA params")

# ── Freeze LLM + Projector, unfreeze Vision ──
log("\nSetting trainable flags...")
for name, param in model.named_parameters():
    if 'lora_' not in name:
        continue
    if VISION_KEY_PREFIX in name:
        param.stop_gradient = False  # Trainable: vision LoRA
    else:
        param.stop_gradient = True   # Frozen: LLM + Projector LoRA

trainable_params = [p for p in model.parameters() if not p.stop_gradient]
trainable_count = sum(p.size for p in trainable_params)
log(f"  Trainable: {trainable_count:,} params ({len(trainable_params)} tensors)")
log(f"  Frozen:    {sum(p.size for p in model.parameters()) - trainable_count:,} params")

# ── Data ──
data_path = f"{DATASET_DIR}/ocr_vl_sft-train-v9-pure.jsonl"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Training data not found: {data_path}")

with open(data_path, encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data)
total_samples = EPOCHS * len(data)
total_steps = total_samples // GRAD_ACCUM
log(f"Training: {len(data)} samples x {EPOCHS} epochs = {total_samples} samples = {total_steps} optimizer steps")

# ── Optimizer (Vision LoRA only, higher LR) ──
cosine_decay = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=VISION_LR, T_max=total_steps - WARMUP_STEPS, eta_min=ETA_MIN)
lr_scheduler = paddle.optimizer.lr.LinearWarmup(
    learning_rate=cosine_decay,
    warmup_steps=WARMUP_STEPS, start_lr=ETA_MIN, end_lr=VISION_LR)
opt = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler, parameters=trainable_params,
    weight_decay=0.1)

log(f"Optimizer: AdamW({len(trainable_params)} vision-only params)")
log(f"  LR schedule: LinearWarmup({WARMUP_STEPS}, {ETA_MIN:.0e}→{VISION_LR:.0e}) + Cosine(→{ETA_MIN:.0e})")

# ── Quick inference helper (Manual Greedy Decoder with repetition_penalty) ──
def quick_inference(samples, max_tokens=60):
    preds = []
    for s in samples:
        try:
            from PIL import Image
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


# ── Load test data for checkpoint monitoring ──
test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50-pure.jsonl"
if os.path.exists(test_path):
    with open(test_path, encoding="utf-8") as f:
        test_data = [json.loads(l) for l in f if l.strip()]
    monitor_samples = test_data[:3]
    log(f"Monitor: easy50-pure ({len(test_data)} samples)")
else:
    val_path = f"{DATASET_DIR}/ocr_vl_sft-val-v9-pure.jsonl"
    with open(val_path, encoding="utf-8") as f:
        val_data = [json.loads(l) for l in f if l.strip()]
    monitor_samples = val_data[:3]
    log(f"Monitor: val-v9-pure ({len(val_data)} samples)")

# ── Train ──
from PIL import Image; from io import BytesIO
model.train()
t0 = time.time()
global_step = 0
history = []
opt.clear_grad()

log(f"\n{'='*70}")
log(f"STARTING TRAINING")
log(f"{'='*70}")

for epoch in range(EPOCHS):
    random.shuffle(data)
    log(f"--- Epoch {epoch+1}/{EPOCHS} ---")

    for idx, sample in enumerate(data):
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

        if (idx + 1) % GRAD_ACCUM == 0 or idx == len(data) - 1:
            paddle.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP)
            opt.step()
            lr_scheduler.step()
            opt.clear_grad()
            global_step += 1

            if global_step % 20 == 0 or global_step == 1:
                elapsed = (time.time()-t0)/60
                eta = (elapsed/global_step*total_steps - elapsed) if global_step > 0 else 0
                log(f"  [S{global_step}/{total_steps}] loss={loss.item():.4f} lr={opt.get_lr():.2e} elapsed={elapsed:.0f}m ETA={eta:.0f}m")
                history.append({"step": global_step, "loss": float(loss.item()), "lr": opt.get_lr()})

            # ── Checkpoint Save & Monitor ──
            if global_step % CHECKPOINT_STEPS == 0:
                log(f"--- Checkpoint at S{global_step} ---")
                model.eval()

                # Save LoRA weights (vision + frozen LLM + projector for full model)
                lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                ckpt_path = f"{CKPT_DIR}/lora_s{global_step}.pdparams"
                paddle.save(lora_dict, ckpt_path)
                log(f"  Saved: {ckpt_path} ({len(lora_dict)} matrices)")

                # Also save vision-only LoRA weights (for analysis)
                vision_only = {k: v for k, v in lora_dict.items() if VISION_KEY_PREFIX in k}
                vision_ckpt = f"{CKPT_DIR}/vision_only_s{global_step}.pdparams"
                paddle.save(vision_only, vision_ckpt)
                log(f"  Vision-only saved: {vision_ckpt} ({len(vision_only)} matrices)")

                # Monitor inference quality
                log("  Running validation inference...")
                preds = quick_inference(monitor_samples)
                for m_idx, pred in enumerate(preds):
                    ref = monitor_samples[m_idx]["messages"][1]["content"][:80]
                    log(f"    Sample {m_idx} Pred: {repr(pred[:100])}")
                    log(f"    Sample {m_idx} Ref:  {repr(ref)}")

                unique_preds = len(set(preds))
                log(f"    Diversity: {unique_preds}/{len(preds)}")

                # Save as latest best
                best_path = f"{OUTPUT_DIR}/lora_best_v12_stage2_fp16.pdparams"
                paddle.save(lora_dict, best_path)
                log(f"  Also saved as best/latest: {best_path}")

                paddle.device.cuda.empty_cache()
                model.train()

total_min = (time.time()-t0)/60
log(f"\nTraining done in {total_min:.0f}m")

# ── Save Final Model ──
model.eval()
lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
final_path = f"{OUTPUT_DIR}/lora_v12_stage2_final_fp16.pdparams"
paddle.save(lora_dict, final_path)
log(f"Final model saved: {final_path} ({len(lora_dict)} matrices)")

# Save vision-only final
vision_only_final = {k: v for k, v in lora_dict.items() if VISION_KEY_PREFIX in k}
vision_final_path = f"{OUTPUT_DIR}/lora_v12_stage2_vision_only_fp16.pdparams"
paddle.save(vision_only_final, vision_final_path)
log(f"Vision-only final saved: {vision_final_path} ({len(vision_only_final)} matrices)")

# ── Final Report ──
log("=" * 70)
log("TRAINING V12-STAGE2 (Phase 3) SUMMARY")
log(f"  V10-Fixed base: {args.v10_checkpoint}")
log(f"  Vision rank: {VISION_RANK}, Vision LR: {VISION_LR:.0e}")
log(f"  Resolution: {MAX_DIM}px")
log(f"  Epochs: {EPOCHS}")
log(f"  Total steps: {global_step}")
log(f"  Total time: {total_min:.0f}m")
log(f"  Trainable params: {trainable_count:,}")
log(f"  Final model: {final_path}")
log(f"  Vision-only: {vision_final_path}")
log(f"  Checkpoints: {CKPT_DIR}")
if history:
    log(f"  Initial loss: {history[0]['loss']:.4f}")
    log(f"  Final loss: {history[-1]['loss']:.4f}")
log("=" * 70)

# ── Save training history ──
with open(f"{CKPT_DIR}/training_history_v12_stage2.json", "w") as f:
    json.dump({
        "phase": "Phase 3 Stage 2: Vision LoRA Retraining",
        "hypothesis": "Competing gradients degraded V10-Fixed vision LoRA; freeze LLM, retrain vision",
        "history": history,
        "total_steps": global_step,
        "total_min": total_min,
        "config": {
            "v10_checkpoint": args.v10_checkpoint,
            "vision_rank": VISION_RANK,
            "vision_alpha": VISION_ALPHA,
            "llm_rank": LLM_RANK,
            "llm_alpha": LLM_ALPHA,
            "vision_lr": VISION_LR,
            "warmup_steps": WARMUP_STEPS,
            "eta_min": ETA_MIN,
            "max_dim": MAX_DIM,
            "epochs": EPOCHS,
            "grad_accum": GRAD_ACCUM,
            "grad_clip": GRAD_CLIP,
            "repetition_penalty": REPETITION_PENALTY,
            "trainable_params": trainable_count,
            "dataset": "ocr_vl_sft-train-v9-pure.jsonl",
            "num_samples": len(data),
        }
    }, f, ensure_ascii=False, indent=2)

log("Training V12-Stage2 complete!")
