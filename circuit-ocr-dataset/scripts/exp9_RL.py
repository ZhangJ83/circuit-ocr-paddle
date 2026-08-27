"""exp9: RL reward-guided fine-tuning on exp6 weights.
Uses reward-weighted regression: generate, score, retrain on best outputs."""
import os, sys, json, time, random, re
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
OUT = os.path.join(D, "PaddleOCR-VL-LoRA-circuit-ocr", "checkpoints_exp9")
os.makedirs(OUT, exist_ok=True)
MAX_DIM = 384; GRAD_ACCUM = 4; GRAD_CLIP = 1.0
RANK = 16; ALPHA = 32

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("=== EXP9: RL Reward-Guided Fine-Tuning ===")
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
        if list(p.shape) == list(v.shape): p.set_value(v); loaded += 1
log(f"Loaded {loaded} params from exp6")

tp = [p for p in model.parameters() if not p.stop_gradient]
log(f"Trainable: {sum(p.numel() for p in tp):,}")

# Load data
train_path = os.path.join(D, "output", "train_v10fmt_synth.jsonl")
with open(train_path, encoding="utf-8") as f:
    all_data = [json.loads(l) for l in f if l.strip()]
# Use circuit-only for RL (no synthetic text)
circuit_data = [s for s in all_data if 'synth_text_images' not in s['images'][0]]
random.shuffle(circuit_data)
train_data = circuit_data[:200]  # 200 samples for RL
log(f"RL training: {len(train_data)} circuit samples")

# Reward function
re_comp = re.compile(r'\b((?:LED|[RCDLQUJYF])\d+)\b')

def compute_reward(pred, ref):
    """Reward based on CompF1 + JointF1. Range ~[-1, 1]."""
    # Component F1
    pc = set(re_comp.findall(pred))
    rc = set(re_comp.findall(ref))
    if not pc and not rc: cf1 = 1.0
    elif not pc or not rc: cf1 = 0.0
    else:
        tp = len(pc & rc); prec = tp/len(pc); rec = tp/len(rc)
        cf1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

    # Joint F1 (simplified)
    def parse(t):
        ps = set()
        for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))', t):
            v = m.group(2).strip().rstrip(',').replace(' ','').upper()
            if v and len(v)<50: ps.add((m.group(1), v))
        return ps
    pp = parse(pred); rp = parse(ref)
    if not pp and not rp: jf1 = 1.0
    elif not pp or not rp: jf1 = 0.0
    else:
        tp = len(pp & rp); prec = tp/len(pp); rec = tp/len(rp)
        jf1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

    # Collapse penalty: if prediction is all numbers "1\n2\n3..."
    lines = pred.strip().split('\n')
    num_lines = sum(1 for l in lines if re.match(r'^\d+$', l.strip()))
    collapse_penalty = -0.5 if num_lines > len(lines)*0.5 and len(lines)>4 else 0.0

    # Diversity bonus
    unique_lines = len(set(l.strip() for l in lines if l.strip()))
    div_bonus = 0.1 if unique_lines > 5 else -0.1

    return cf1 * 0.4 + jf1 * 0.3 + collapse_penalty + div_bonus

def generate_prediction(sample, temperature=0.8):
    """Generate a prediction from the model with sampling."""
    try:
        img_path = sample['images'][0]
        if not os.path.exists(img_path): return None
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

        msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
        inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
        ids = inp["input_ids"]; am = inp["attention_mask"]
        pv = inp.get("pixel_values"); igt = inp.get("image_grid_thw")
        gen = []
        with paddle.no_grad():
            for _ in range(80):
                out = model(input_ids=ids, attention_mask=am, pixel_values=pv, image_grid_thw=igt)
                logits = (out[0] if isinstance(out, (tuple, list)) else out.logits)[:, -1, :].astype("float32")
                # Apply temperature
                logits = logits / temperature
                # Repetition penalty
                for tid in set(gen):
                    logits[0, tid] = logits[0, tid] * 1.1 if logits[0, tid] < 0 else logits[0, tid] / 1.1
                # Sample (not argmax for RL exploration)
                probs = paddle.nn.functional.softmax(logits, axis=-1)
                nt = int(paddle.multinomial(probs, num_samples=1).numpy()[0][0])
                if nt == processor.tokenizer.eos_token_id: break
                gen.append(nt)
                ids = paddle.concat([ids, paddle.to_tensor([[nt]])], axis=1)
                am = paddle.concat([am, paddle.ones([1, 1], dtype=am.dtype)], axis=1)
        pred = processor.tokenizer.decode(gen, skip_special_tokens=True)
        image.close()
        return pred
    except Exception as e:
        return None

# RL training loop
BASE_LR = 5e-6  # Very small LR for RL
EPOCHS = 2
total_steps = EPOCHS * len(train_data) // GRAD_ACCUM

cosine = paddle.optimizer.lr.CosineAnnealingDecay(BASE_LR, T_max=max(1, total_steps), eta_min=BASE_LR/10)
lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=min(20, total_steps//4), start_lr=BASE_LR/10, end_lr=BASE_LR)
opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.01)

model.train(); t0 = time.time(); gs = 0; total_reward = 0.0; opt.clear_grad()

log(f"RL steps: {total_steps} (GRAD_ACCUM={GRAD_ACCUM})")

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    log(f"--- RL Epoch {epoch+1}/{EPOCHS} ---")
    for idx, sample in enumerate(train_data):
        # Phase 1: Generate prediction (exploration)
        model.eval()
        pred = generate_prediction(sample, temperature=0.8)
        model.train()

        if pred is None: continue

        ref = sample["messages"][1]["content"]
        reward = compute_reward(pred, ref)
        total_reward += reward

        # Phase 2: Train on ground truth with reward weighting
        # (reward-weighted SFT: high reward = learn more from GT)
        try:
            img_path = sample['images'][0]
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

            prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
            pi = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
            prompt_ids = pi["input_ids"][0]; prompt_len = prompt_ids.shape[0]

            lt = processor.tokenizer(ref, return_tensors="pd", padding=False, truncation=True, max_length=512)
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

            # Weight by reward: high reward -> learn more from GT
            rw_weight = max(0.1, reward + 0.5)  # range [0.1, 1.5]
            (loss * rw_weight / GRAD_ACCUM).backward()
            image.close()
        except Exception as e:
            continue

        if (idx + 1) % GRAD_ACCUM == 0 or idx == len(train_data) - 1:
            paddle.nn.utils.clip_grad_norm_(tp, GRAD_CLIP)
            opt.step(); lrs.step(); opt.clear_grad(); gs += 1
            if gs % 5 == 0:
                avg_r = total_reward / max(1, (epoch * len(train_data) + idx + 1))
                log(f"  RL S{gs}/{total_steps} avg_reward={avg_r:.3f} lr={opt.get_lr():.1e}")

lora_final = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
paddle.save(lora_final, os.path.join(OUT, "lora_exp9_RL.pdparams"))
paddle.save(lora_final, os.path.join(OUT, "best.pdparams"))
log(f"\nDONE exp9 RL. Steps: {gs}. Avg reward: {total_reward/max(1,epoch*len(train_data)+1):.3f}")
log(f"Output: {OUT}")
