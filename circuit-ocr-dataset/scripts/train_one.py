"""Generic single-experiment training script. CLI args for pipeline."""
import os, sys, json, time, random, argparse
from types import ModuleType
_d = ModuleType('d'); _d.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _d)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="exp")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--data", default="")
ap.add_argument("--output", default="")
ap.add_argument("--ckpt", default="")  # Pre-trained LoRA checkpoint to load
args = ap.parse_args()

D = r"g:/mimo_project/circuit_ocr"
M = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
RANK = 16; MAX_DIM = 384; GRAD_ACCUM = 4; GRAD_CLIP = 1.0
DROP = 0.05
OUT = args.output or os.path.join(D, "checkpoints", args.name)
os.makedirs(OUT, exist_ok=True)

log(f"=== {args.name} lr={args.lr:.0e} alpha={args.alpha} epochs={args.epochs} warmup={args.warmup} ===")

model = AutoModelForConditionalGeneration.from_pretrained(
    M, convert_from_hf=True, load_checkpoint_format="naive", low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
lc = LoRAConfig(r=RANK, lora_alpha=args.alpha, target_modules=TARGETS, lora_dropout=DROP)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'): model.model.full = lambda *a, **kw: iter(model.model.named_parameters())

# Phase 2 support: load pre-trained LoRA checkpoint
if args.ckpt and os.path.exists(args.ckpt):
    state = paddle.load(args.ckpt)
    n_loaded = 0
    for k, p in model.named_parameters():
        if k in state:
            v = state[k]
            if p.dtype != v.dtype: v = paddle.cast(v, p.dtype)
            if list(p.shape) == list(v.shape): p.set_value(v); n_loaded += 1
    log(f"  Loaded {n_loaded} params from {os.path.basename(args.ckpt)}")
processor = AutoProcessor.from_pretrained(M)
tp = [p for p in model.parameters() if not p.stop_gradient]
log(f"Trainable: {sum(p.numel() for p in tp):,}")

data_path = args.data or os.path.join(D, "output", "train_3k.jsonl")
with open(data_path, encoding="utf-8") as f:
    all_data = [json.loads(l) for l in f if l.strip()]
random.shuffle(all_data)
split = int(len(all_data) * 0.9)
train_data = all_data[:split]; val_data = all_data[split:]
total_steps = args.epochs * len(train_data) // GRAD_ACCUM
log(f"Train: {len(train_data)} Val: {len(val_data)} Steps: {total_steps}")

cosine = paddle.optimizer.lr.CosineAnnealingDecay(args.lr, T_max=max(1, total_steps - args.warmup), eta_min=args.lr/10)
lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=args.warmup, start_lr=args.lr/10, end_lr=args.lr)
opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

model.train(); t0 = time.time(); gs = 0; el_acc = 0.0; opt.clear_grad()
best_loss = float('inf'); skipped = 0

for epoch in range(args.epochs):
    random.shuffle(train_data)
    log(f"--- Epoch {epoch+1}/{args.epochs} ---")
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
            else: query = content_raw
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
                if gs % 20 == 0:
                    eta = (time.time()-t0)/max(1, gs) * (total_steps-gs)/60
                    log(f"  S{gs}/{total_steps} loss={el_acc/max(1,idx+1):.4f} lr={opt.get_lr():.2e} ETA={eta:.0f}m")
        except Exception as e:
            skipped += 1
            if skipped <= 3: log(f"  SKIP: {str(e)[:60]}")
            try: opt.clear_grad()
            except: pass; continue

lora_final = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
paddle.save(lora_final, os.path.join(OUT, "best.pdparams"))
log(f"DONE {args.name}. {int((time.time()-t0)/60)}min. Output: {OUT}")
