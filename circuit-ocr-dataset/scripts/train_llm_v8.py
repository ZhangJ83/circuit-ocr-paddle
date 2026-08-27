"""
V8: Return to the PROVEN architecture from train_r16_v100.py
============================================================
Key changes from V7:
  1. MANUAL CE loss with correct shift (logits[:,:-1] vs labels[:,1:])
     - Avoids double-shift bug when model also does internal shift
  2. WIDE LoRA targets: .*q_proj, .*k_proj, .*v_proj, .*o_proj, .*linear_1, .*linear_2
     - Covers Vision Encoder + LLM + Projector (~306 LoRA matrices, ~5.7M params)
  3. HIGH learning rate: 5e-4 -> 5e-5 (Cosine), matching the successful baseline
  4. 3 epochs on V5-Golden dataset (2,299 samples)
  5. MAX_DIM=512 for better text readability

New additions over train_r16_v100.py:
  - Gradient clipping (1.0) for stability
  - Checkpoint saving every 500 steps with inference monitoring
  - Manual greedy decode for monitoring (bypasses buggy LoRAModel.generate())
"""
import os, sys, json, time, random
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HOME": "F:/hf_cache/hub",
    "PADDLE_HOME": "F:/paddle_cache", "HF_HUB_CACHE": "F:/hf_cache/hub",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "FLAGS_allocator_strategy": "auto_growth",
})
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
CKPT_DIR = f"{OUTPUT_DIR}/checkpoints_v8"
os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    try: print(f"[{ts}] {msg}", flush=True)
    except: print(f"[{ts}] {msg.encode('ascii','replace').decode('ascii')}", flush=True)

# ── Config ──
MAX_DIM = 512
EPOCHS = 3
GRAD_CLIP = 1.0
CHECKPOINT_STEPS = 500

# WIDE targets: Vision Encoder + LLM + Projector (same as successful train_r16_v100.py)
TARGETS = [
    ".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj",   # all attention layers
    ".*linear_1", ".*linear_2",                         # Projector (mlp_AR)
]

log("=" * 60)
log("TRAINING V8 (PROVEN ARCHITECTURE + V5-GOLDEN DATASET)")
log(f"  Targets: {TARGETS}")
log(f"  Config: max_dim={MAX_DIM}, epochs={EPOCHS}, LR=5e-4->5e-5, grad_clip={GRAD_CLIP}")
log(f"  Dataset: ocr_vl_sft-train-v5-golden.jsonl")
log(f"  Loss: MANUAL CE with correct shift (no double-shift)")
log("=" * 60)

# ── Load Model ──
log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

# LoRA with r=16, alpha=32 (scale = 2.0)
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)

trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
lora_count = sum(1 for k, p in model.named_parameters() if 'lora_' in k)
log(f"Trainable parameters: {trainable:,}  LoRA matrices: {lora_count}")

# ── Data ──
with open(f"{DATASET_DIR}/ocr_vl_sft-train-v5-golden.jsonl", encoding="utf-8") as f:
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

# ── Quick inference helper (Manual Greedy Decoder) ──
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
                    next_token = int(paddle.argmax(next_token_logits, axis=-1).numpy()[0])
                    if next_token == processor.tokenizer.eos_token_id:
                        break
                    generated.append(next_token)
                    next_tensor = paddle.to_tensor([[next_token]], dtype=input_ids.dtype)
                    input_ids = paddle.concat([input_ids, next_tensor], axis=1)
                    attention_mask = paddle.concat([attention_mask, paddle.ones([1, 1], dtype=attention_mask.dtype)], axis=1)
                    # Only need pixel_values on first step
                    pixel_values = None
                    image_grid_thw = None

            resp = processor.tokenizer.decode(generated, skip_special_tokens=True)
            preds.append(resp)
            img.close()
            del img, inp, input_ids, attention_mask, generated; paddle.device.cuda.empty_cache()
        except Exception as e:
            preds.append(f"[ERR:{str(e)[:40]}]")
    return preds

# Load test data for checkpoint monitoring
test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl"
with open(test_path, encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()]
monitor_samples = test_data[:3]

# ── Train ──
from PIL import Image; from io import BytesIO
model.train()
t0 = time.time()
global_step = 0
history = []

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
        label = sample["messages"][1]["content"][:200]

        # === TOKENIZATION (same as train_r16_v100.py) ===
        # Step 1: Tokenize prompt only (with generation prompt)
        prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
        prompt_inputs = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Step 2: Tokenize full conversation (prompt + assistant response)
        full_msgs = [
            {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]},
            {"role":"assistant","content":[{"type":"text","text":label}]},
        ]
        full_inputs = processor.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pd")
        full_ids = full_inputs["input_ids"]

        # Step 3: Create labels — mask prompt, keep only assistant tokens
        labels_t = paddle.full_like(full_ids, -100, dtype=full_ids.dtype)
        labels_t[0, prompt_len:] = full_ids[0, prompt_len:]

        # === FORWARD PASS (raw logits, NO labels to model) ===
        out = model(**full_inputs)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        # === MANUAL CE LOSS with CORRECT causal shift ===
        shift_logits = paddle.cast(logits[:, :-1, :], "float32")
        shift_labels = labels_t[:, 1:]
        mask = paddle.cast(shift_labels != -100, "float32")
        shift_labels_clamped = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
        ce = paddle.nn.functional.cross_entropy(
            shift_logits.reshape([-1, shift_logits.shape[-1]]),
            shift_labels_clamped.reshape([-1]), reduction="none").reshape(shift_labels.shape)
        loss = (ce * mask).sum() / mask.sum().clip(min=1)

        # === BACKWARD + UPDATE ===
        loss.backward()
        paddle.nn.utils.clip_grad_norm_([p for p in model.parameters() if not p.stop_gradient], max_norm=GRAD_CLIP)
        opt.step(); opt.clear_grad()
        lr_scheduler.step()
        global_step += 1
        image.close()

        if global_step % 100 == 0 or global_step == 1:
            elapsed = (time.time()-t0)/60
            eta = (elapsed/global_step*total_steps - elapsed) if global_step > 0 else 0
            log(f"  [S{global_step}/{total_steps}] loss={loss.item():.4f} lr={opt.get_lr():.2e} elapsed={elapsed:.0f}m ETA={eta:.0f}m")
            history.append({"step": global_step, "loss": float(loss.item()), "lr": opt.get_lr()})

        # ── Checkpoint Save & Monitor ──
        if global_step % CHECKPOINT_STEPS == 0:
            log(f"--- Checkpoint at S{global_step} ---")
            model.eval()

            # Save LoRA weights
            lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
            ckpt_path = f"{CKPT_DIR}/lora_s{global_step}.pdparams"
            paddle.save(lora_dict, ckpt_path)
            log(f"  Saved: {ckpt_path} ({len(lora_dict)} matrices)")

            # Monitor inference quality
            log("  Running quick validation inference...")
            preds = quick_inference(monitor_samples)
            for m_idx, pred in enumerate(preds):
                ref = monitor_samples[m_idx]["messages"][1]["content"][:80]
                log(f"    Sample {m_idx} Pred: {repr(pred[:100])}")
                log(f"    Sample {m_idx} Ref:  {repr(ref)}")

            unique_preds = len(set(preds))
            log(f"    Diversity: {unique_preds}/{len(preds)}")

            # Also save as latest best
            best_path = f"{OUTPUT_DIR}/lora_best_v8_fp16.pdparams"
            paddle.save(lora_dict, best_path)
            log(f"  Also saved as best/latest: {best_path}")

            paddle.device.cuda.empty_cache()
            model.train()

total_min = (time.time()-t0)/60
log(f"\nTraining done in {total_min:.0f}m")

# ── Save Final Model ──
model.eval()
lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
final_path = f"{OUTPUT_DIR}/lora_v8_final_fp16.pdparams"
paddle.save(lora_dict, final_path)
log(f"Final model saved: {final_path} ({len(lora_dict)} matrices)")

# ── Final Report ──
log("=" * 60)
log("TRAINING V8 SUMMARY")
log(f"  Total steps: {total_steps}")
log(f"  Total time: {total_min:.0f}m")
log(f"  Final model: {final_path}")
log(f"  Checkpoints: {CKPT_DIR}")
if history:
    log(f"  Initial loss: {history[0]['loss']:.4f}")
    log(f"  Final loss: {history[-1]['loss']:.4f}")
log("=" * 60)

# ── Save training history ──
with open(f"{CKPT_DIR}/training_history_v8.json", "w") as f:
    json.dump({"history": history, "total_steps": total_steps, "total_min": total_min}, f)

log("Training V8 complete!")
