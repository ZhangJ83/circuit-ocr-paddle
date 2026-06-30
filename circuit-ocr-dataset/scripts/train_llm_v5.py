"""
SAFE LLM-ONLY LoRA SFT training v5 with checkpoints and correct token alignment.
- Targets ONLY the LLM decoder self-attention layers: model.layers.X.self_attn.{q,k,v,o}_proj
- Freezes visual encoder and projector (mlp_AR) to preserve pre-trained VL alignment and prevent collapse
- Improved dataset: 1,857 samples (1,357 real + 500 new synthetic)
- max_dim=384 (keep text readable)
- Gradient accumulation (4 steps)
- Gradient clipping (1.0)
- Learning rate warmup (100 steps)
- Safe learning rate: 2e-5 (Cosine decay to 2e-6)
- Manual token concatenation: 100% alignment between prompt and label
"""
import os, sys, json, time, random
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HOME": "F:/hf_cache/hub",
    "PADDLE_HOME": "F:/paddle_cache", "HF_HUB_CACHE": "F:/hf_cache/hub",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "FLAGS_allocator_strategy": "auto_growth",
})
sys.stdout.reconfigure(encoding='utf-8') # Fix Windows console encoding issues

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
CKPT_DIR = f"{OUTPUT_DIR}/checkpoints_v5"
os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    try: print(f"[{ts}] {msg}", flush=True)
    except: print(f"[{ts}] {msg.encode('ascii','replace').decode('ascii')}", flush=True)

# ── Config ──
MAX_DIM = 384
EPOCHS = 2
GRAD_ACCUM = 4
GRAD_CLIP = 1.0
CHECKPOINT_STEPS = 200

# Targets: ONLY LLM decoder layers, no projector or vision encoder
TARGETS = [
    "model\\.layers\\..*q_proj",
    "model\\.layers\\..*k_proj",
    "model\\.layers\\..*v_proj",
    "model\\.layers\\..*o_proj",
]

log("=" * 60)
log("TRAINING V5 (LLM-ONLY LoRA r=8 alpha=16, RESOLUTION 384, LR 2e-5)")
log(f"Dataset: ocr_vl_sft-train-v2.jsonl")
log(f"Targets: {TARGETS}")
log(f"Config: max_dim={MAX_DIM}, epochs={EPOCHS}, checkpoint_every={CHECKPOINT_STEPS}, grad_accum={GRAD_ACCUM}")
log("=" * 60)

# ── Load Model ──
log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

# LoRA with r=8, alpha=16 (scale = 2.0)
lc = LoRAConfig(r=8, lora_alpha=16, target_modules=TARGETS)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)

trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
lora_count = sum(1 for k, p in model.named_parameters() if 'lora_' in k)
log(f"Trainable parameters: {trainable:,}  LoRA matrices: {lora_count}")

# ── Data ──
with open(f"{DATASET_DIR}/ocr_vl_sft-train-v2.jsonl", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data)
total_samples = EPOCHS * len(data)
total_steps = total_samples // GRAD_ACCUM
log(f"Training: {len(data)} samples x {EPOCHS} epochs = {total_samples} samples = {total_steps} optimizer steps")

# ── Optimizer & Warmup Scheduler ──
WARMUP_STEPS = 100
base_lr_decay = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=2e-5, T_max=total_steps - WARMUP_STEPS, eta_min=2e-6)
lr_scheduler = paddle.optimizer.lr.LinearWarmup(
    learning_rate=base_lr_decay, warmup_steps=WARMUP_STEPS, start_lr=2e-6, end_lr=2e-5)

opt = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler, parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1)

# ── Quick inference helper ──
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

# Load test data for checkpoints monitoring
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
opt.clear_grad()

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

        # Tokenize prompt (includes generation prompt, ending with token 93919)
        prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
        prompt_inputs = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Tokenize label separately
        lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=512)
        label_ids = lt["input_ids"][0]
        # Append EOS token ID (2 = </s>)
        eos_tensor = paddle.to_tensor([processor.tokenizer.eos_token_id], dtype=label_ids.dtype)
        label_ids = paddle.concat([label_ids, eos_tensor], axis=0)
        label_len = label_ids.shape[0]

        # Manually concatenate inputs and labels to guarantee 100% boundary alignment
        full_input_ids = paddle.concat([prompt_inputs["input_ids"][0], label_ids], axis=0).unsqueeze(0)
        full_attn_mask = paddle.concat([prompt_inputs["attention_mask"][0], paddle.ones([label_len], dtype="int64")], axis=0).unsqueeze(0)
        
        # Labels for cross entropy calculation
        labels_t = paddle.full([1, prompt_len + label_len], fill_value=-100, dtype="int64")
        labels_t[0, prompt_len:] = label_ids

        # Forward pass
        out = model(
            input_ids=full_input_ids,
            attention_mask=full_attn_mask,
            pixel_values=prompt_inputs["pixel_values"],
            image_grid_thw=prompt_inputs.get("image_grid_thw"),
            labels=labels_t
        )
        loss = out.loss if hasattr(out, "loss") else out[0]

        # Scale loss for gradient accumulation
        scaled_loss = loss / GRAD_ACCUM
        scaled_loss.backward()
        image.close()

        # Update weights every GRAD_ACCUM steps
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

            # ── Checkpoint Save & Monitor ──
            if global_step % CHECKPOINT_STEPS == 0:
                log(f"--- Checkpoint at S{global_step} ---")
                
                # Save model
                model.eval()
                lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                ckpt_path = f"{CKPT_DIR}/lora_s{global_step}.pdparams"
                paddle.save(lora_dict, ckpt_path)
                log(f"  Saved: {ckpt_path}")

                # Monitor sample quality
                log("  Running quick validation inference...")
                preds = quick_inference(monitor_samples)
                for m_idx, pred in enumerate(preds):
                    log(f"    Sample {m_idx}: {repr(pred)}")
                
                # Check for diversity/collapse
                unique_preds = len(set(preds))
                log(f"    Monitor Diversity: {unique_preds}/{len(preds)}")
                
                best_ckpt_path = f"{OUTPUT_DIR}/lora_best_v5_fp16.pdparams"
                paddle.save(lora_dict, best_ckpt_path)
                log(f"  Also saved as best/latest")

                paddle.device.cuda.empty_cache()
                model.train()  # Back to training mode

total_min = (time.time()-t0)/60
log(f"\nTraining done in {total_min:.0f}m")

# ── Save Final Model ──
model.eval()
lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
final_path = f"{OUTPUT_DIR}/lora_projector_v5_final_fp16.pdparams"
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

log("Training complete!")
