"""exp8: SPICE-format fine-tuning on top of exp6 best weights."""
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
OUT = os.path.join(D, "PaddleOCR-VL-LoRA-circuit-ocr", "checkpoints_exp8")
os.makedirs(OUT, exist_ok=True)
RANK = 16; ALPHA = 32; MAX_DIM = 384; GRAD_ACCUM = 4; GRAD_CLIP = 1.0

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("=== EXP8: SPICE Fine-tune on exp6 weights ===")
log("Loading model...")
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

# Load exp6 best weights
exp6_path = os.path.join(D, "PaddleOCR-VL-LoRA-circuit-ocr", "lora_exp6_best.pdparams")
state = paddle.load(exp6_path)
loaded = 0
for k, p in model.named_parameters():
    if k in state:
        v = state[k]
        if p.dtype != v.dtype: v = paddle.cast(v, p.dtype)
        if list(p.shape) == list(v.shape):
            p.set_value(v); loaded += 1
log(f"Loaded {loaded} params from exp6")

tp = [p for p in model.parameters() if not p.stop_gradient]
log(f"Trainable: {sum(p.numel() for p in tp):,}")

# Load SPICE data
spice_path = os.path.join(D, "output", "train_spice.jsonl")
with open(spice_path, encoding="utf-8") as f:
    spice_data = [json.loads(l) for l in f if l.strip()]
random.shuffle(spice_data)

# Also mix in some original circuit data for stability (20%)
orig_path = os.path.join(D, "output", "train_v10fmt.jsonl")
with open(orig_path, encoding="utf-8") as f:
    orig_data = [json.loads(l) for l in f if l.strip()]

# Use 80% SPICE + 20% original
split_spice = int(len(spice_data) * 0.9)
train_data = spice_data[:split_spice] + random.sample(orig_data, min(len(spice_data)//5, len(orig_data)))
val_data = spice_data[split_spice:]
random.shuffle(train_data)
log(f"Train: {len(train_data)} ({len(spice_data[:split_spice])} SPICE + {min(len(spice_data)//5, len(orig_data))} orig)")

EPOCHS = 1; BASE_LR = 1e-5
total_steps = EPOCHS * len(train_data) // GRAD_ACCUM
log(f"Steps: {total_steps}")

cosine = paddle.optimizer.lr.CosineAnnealingDecay(BASE_LR, T_max=max(1, total_steps - 50), eta_min=BASE_LR/10)
lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=min(50, total_steps//3), start_lr=BASE_LR/10, end_lr=BASE_LR)
opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

model.train(); t0 = time.time(); gs = 0; el_acc = 0.0; opt.clear_grad()
best_loss = float('inf'); skipped = 0

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    log(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    for idx, sample in enumerate(train_data):
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

            if (idx + 1) % GRAD_ACCUM == 0 or idx == len(train_data) - 1:
                paddle.nn.utils.clip_grad_norm_(tp, GRAD_CLIP)
                opt.step(); lrs.step(); opt.clear_grad(); gs += 1
                if gs % 10 == 0:
                    eta = (time.time()-t0)/max(1, gs) * (total_steps-gs)/60
                    avg_loss = el_acc/max(1, idx+1)
                    log(f"  S{gs}/{total_steps} loss={avg_loss:.4f} lr={opt.get_lr():.2e} ETA={eta:.0f}m")
                    # Print sample predictions
                    if gs % 50 == 0:
                        model.eval()
                        test_samples = val_data[:2]
                        for si, ts in enumerate(test_samples):
                            try:
                                img_path2 = ts['images'][0]
                                img2 = Image.open(img_path2).convert("RGB")
                                w2, h2 = img2.size
                                if max(w2, h2) > MAX_DIM:
                                    s2 = MAX_DIM / max(w2, h2); img2 = img2.resize((int(w2*s2), int(h2*s2)), Image.LANCZOS)
                                buf2 = BytesIO(); img2.save(buf2, format="JPEG", quality=95); buf2.seek(0); img2 = Image.open(buf2)
                                cr2 = ts["messages"][0]["content"]
                                q2 = [it['text'] for it in cr2 if it.get('type')=='text'][0] if isinstance(cr2, list) else cr2
                                msgs2 = [{"role":"user","content":[{"type":"image","image":img2},{"type":"text","text":q2.replace("<image>","")}]}]
                                inp2 = processor.apply_chat_template(msgs2, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
                                ids2=inp2["input_ids"]; am2=inp2["attention_mask"]; pv2=inp2.get("pixel_values"); igt2=inp2.get("image_grid_thw")
                                gen2=[]
                                with paddle.no_grad():
                                    for _ in range(60):
                                        out2=model(input_ids=ids2,attention_mask=am2,pixel_values=pv2,image_grid_thw=igt2)
                                        lo2=(out2[0] if isinstance(out2,(list,tuple)) else out2.logits)[:,-1,:]
                                        for tid2 in set(gen2): sc2=float(lo2[0,tid2]); lo2[0,tid2]=sc2*1.1 if sc2<0 else sc2/1.1
                                        nt2=int(paddle.argmax(lo2,axis=-1).numpy()[0])
                                        if nt2==processor.tokenizer.eos_token_id: break
                                        gen2.append(nt2)
                                        ids2=paddle.concat([ids2,paddle.to_tensor([[nt2]])],axis=1)
                                        am2=paddle.concat([am2,paddle.ones([1,1],dtype=am2.dtype)],axis=1)
                                pred2 = processor.tokenizer.decode(gen2, skip_special_tokens=True)
                                ref2 = ts["messages"][1]["content"][:80]
                                log(f"    [{si}] Pred: {pred2[:80]}")
                                log(f"    [{si}] Ref:  {ref2}")
                                img2.close()
                            except Exception as e2:
                                log(f"    [{si}] ERR: {str(e2)[:40]}")
                        model.train()
        except Exception as e:
            skipped += 1
            if skipped <= 3: log(f"  SKIP: {str(e)[:60]}")
            try: opt.clear_grad()
            except: pass; continue

lora_final = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
paddle.save(lora_final, os.path.join(OUT, "lora_exp8_spice.pdparams"))
paddle.save(lora_final, os.path.join(OUT, "best.pdparams"))
total_min = (time.time()-t0)/60
log(f"\nDONE exp8. {total_min:.0f}min. Output: {OUT}")
