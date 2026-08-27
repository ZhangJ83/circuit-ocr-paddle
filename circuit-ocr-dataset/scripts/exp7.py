"""exp7: Larger LoRA r=32 + Two-Stage Training"""
import os, sys, json, time, random
from types import ModuleType
_d = ModuleType('dummy'); _d.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _d)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

D = r"g:/mimo_project/circuit_ocr"
M = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUT = os.path.join(D, "PaddleOCR-VL-LoRA-circuit-ocr", "checkpoints_exp7")
os.makedirs(OUT, exist_ok=True)
RANK = 32; ALPHA = 64; MAX_DIM = 384; GRAD_ACCUM = 4; GRAD_CLIP = 1.0

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("=== EXP7: LoRA r=32 + Two-Stage ===")
log("Loading base model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    M, convert_from_hf=True, load_checkpoint_format="naive", low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
lc = LoRAConfig(r=RANK, lora_alpha=ALPHA, target_modules=TARGETS, lora_dropout=0.05)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'): model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(M)
tp = [p for p in model.parameters() if not p.stop_gradient]
log(f"Trainable: {sum(p.numel() for p in tp):,}")

# Load data
synth_path = os.path.join(D, "output", "train_v10fmt_synth.jsonl")
with open(synth_path, encoding="utf-8") as f:
    all_data = [json.loads(l) for l in f if l.strip()]
random.shuffle(all_data)
synth_data = [s for s in all_data if 'synth_text_images' in s['images'][0]]
circuit_data = [s for s in all_data if 'synth_text_images' not in s['images'][0]]
log(f"Synth: {len(synth_data)}, Circuit: {len(circuit_data)}")

def train_stage(data, epochs, lr, stage_name, prev_steps=0):
    random.shuffle(data)
    total_steps = epochs * len(data) // GRAD_ACCUM
    log(f"  {stage_name} | {len(data)} samples x {epochs} epochs = {total_steps} steps")

    cosine = paddle.optimizer.lr.CosineAnnealingDecay(lr, T_max=max(1, total_steps - 100), eta_min=lr/10)
    wu = min(100, total_steps // 3)
    lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=wu, start_lr=lr/10, end_lr=lr)
    opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

    model.train(); t0 = time.time(); gs = prev_steps; el_acc = 0.0; opt.clear_grad()
    best_loss = float('inf'); skipped = 0

    for epoch in range(epochs):
        random.shuffle(data)
        log(f"  {stage_name} Epoch {epoch+1}/{epochs}")
        for idx, sample in enumerate(data):
            try:
                img_path = sample['images'][0]
                if not os.path.exists(img_path): skipped += 1; continue
                image = Image.open(img_path).convert("RGB")
                w, h = image.size
                if max(w, h) > MAX_DIM:
                    scale = MAX_DIM / max(w, h); image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0); image = Image.open(buf)

                content_raw = sample["messages"][0]["content"]
                if isinstance(content_raw, list):
                    tp2 = [item["text"] for item in content_raw if item.get("type") == "text"]
                    query = tp2[0] if tp2 else "<image>OCR:"
                else:
                    query = content_raw
                label = sample["messages"][1]["content"]

                prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
                pi = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
                prompt_ids = pi["input_ids"][0]; prompt_len = prompt_ids.shape[0]

                lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=512)
                label_ids = lt["input_ids"][0]
                eos_t = paddle.to_tensor([processor.tokenizer.eos_token_id], dtype=label_ids.dtype)
                label_ids = paddle.concat([label_ids, eos_t], axis=0); label_len = label_ids.shape[0]

                full_ids = paddle.concat([prompt_ids, label_ids], axis=0).unsqueeze(0)
                full_mask = paddle.concat([pi["attention_mask"][0], paddle.ones([label_len], dtype="int64")], axis=0).unsqueeze(0)
                labels_t = paddle.full([1, prompt_len + label_len], -100, dtype="int64")
                labels_t[0, prompt_len:] = label_ids

                out = model(input_ids=full_ids, attention_mask=full_mask,
                           pixel_values=pi["pixel_values"], image_grid_thw=pi.get("image_grid_thw"))
                logits = out[0] if isinstance(out, (tuple, list)) else out.logits
                slogits = paddle.cast(logits[:, :-1, :], "float32"); slabs = labels_t[:, 1:]
                mask = paddle.cast(slabs != -100, "float32")
                sl_clean = paddle.where(slabs != -100, slabs, paddle.zeros_like(slabs))
                ce = paddle.nn.functional.cross_entropy(
                    slogits.reshape([-1, slogits.shape[-1]]), sl_clean.reshape([-1]), reduction="none").reshape(slabs.shape)
                loss = (ce * mask).sum() / mask.sum().clip(min=1)
                (loss / GRAD_ACCUM).backward(); el_acc += loss.item()
                image.close()

                if (idx + 1) % GRAD_ACCUM == 0 or idx == len(data) - 1:
                    paddle.nn.utils.clip_grad_norm_(tp, GRAD_CLIP)
                    opt.step(); lrs.step(); opt.clear_grad(); gs += 1
                    if gs % 10 == 0:
                        eta = (time.time()-t0)/max(1, gs-prev_steps) * (total_steps-(gs-prev_steps))/60
                        avg_loss = el_acc/max(1, idx+1)
                        log(f"  S{gs}/{prev_steps+total_steps} loss={avg_loss:.4f} lr={opt.get_lr():.2e} ETA={eta:.0f}m")
            except Exception as e:
                skipped += 1
                if skipped <= 3: log(f"  SKIP: {str(e)[:60]}")
                try: opt.clear_grad()
                except: pass; continue
    return gs, best_loss

# Stage 1: Synth pre-train
log("\n>>> STAGE 1: Synth pre-training <<<")
gs, _ = train_stage(synth_data, epochs=2, lr=2e-5, stage_name="Synth")
lora_s1 = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
paddle.save(lora_s1, os.path.join(OUT, "stage1_synth.pdparams"))
log(f"Stage 1 saved. Steps: {gs}")

# Stage 2: Circuit fine-tune
log("\n>>> STAGE 2: Circuit fine-tuning <<<")
gs, best_loss = train_stage(circuit_data, epochs=2, lr=1e-5, stage_name="Circuit", prev_steps=gs)

lora_final = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
paddle.save(lora_final, os.path.join(OUT, "lora_exp7_final.pdparams"))
paddle.save(lora_final, os.path.join(OUT, "best.pdparams"))

log(f"\nDONE exp7. Steps: {gs}. Output: {OUT}")
