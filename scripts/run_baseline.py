"""Direct baseline run — clean, test-first, all patches in one place."""
import os, sys, json, time, random
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# === PATCH 1: fix torchvision BEFORE paddleformers ===
sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms
import torch  # needed for type checks below

# === PATCH 2: fix mistral BEFORE paddleformers ===
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu, 'get_one_valid_tokenizer_file'):
    mu.get_one_valid_tokenizer_file = lambda d, e: list(mu._filter_valid_tokenizer_files(d, e))

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

import numpy as np
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

# === PATCH 3: restore availability after paddleformers blocked them ===
sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms
import transformers.utils.import_utils as tiu
tiu.is_torch_available = lambda: (True, '')
tiu.is_torchvision_available = lambda: (True, '')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))
from eval_metrics import compute_all
MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"
CKPT_DIR = f"{PROJECT_DIR}/checkpoints/baseline"
os.makedirs(CKPT_DIR, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# Load data
log("Loading data...")
train = [json.loads(l) for l in open(f"{PROJECT_DIR}/output/train_clean.jsonl") if l.strip()]
val = [json.loads(l) for l in open(f"{PROJECT_DIR}/output/val_clean.jsonl") if l.strip()]
random.shuffle(train)
log(f"Train: {len(train)}, Val: {len(val)}")

# Load model + processor
log("Model...")
proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForConditionalGeneration.from_pretrained(MODEL_PATH, load_checkpoint_format="safetensors", dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
log(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Freeze projector
for n, p in model.named_parameters():
    if "mlp_AR" in n or "projector" in n: p.stop_gradient = True

# LoRA
lc = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=[".*q_proj",".*k_proj",".*v_proj",".*o_proj",".*linear_1",".*linear_2"])
model = LoRAModel(model, lc)
tp = [p for p in model.parameters() if not p.stop_gradient]
log(f"Trainable: {sum(p.numel() for p in tp):,}")

# Optimizer
total_steps = len(train) * 3
cd = paddle.optimizer.lr.CosineAnnealingDecay(2e-5, T_max=max(1, total_steps - 100), eta_min=2e-6)
lrs = paddle.optimizer.lr.LinearWarmup(cd, warmup_steps=100, start_lr=2e-6, end_lr=2e-5)
opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)
log(f"Steps: {total_steps}")

def to_pd(d):
    """Recursively convert numpy to Paddle tensors."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray): out[k] = paddle.to_tensor(v)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray): out[k] = paddle.to_tensor(np.array(v))
        elif isinstance(v, tuple) and len(v) > 0 and isinstance(v[0], np.ndarray): out[k] = paddle.to_tensor(np.array(v))
        elif isinstance(v, torch.Tensor): out[k] = paddle.to_tensor(v.numpy())
        else: out[k] = v
    return out

global_step = 0
t0 = time.time()

for epoch in range(3):
    random.shuffle(train)
    el = 0.0
    for i in range(0, len(train)):
        s = train[i]
        ip = s["images"][0]
        if not os.path.exists(ip):
            ip = ip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")

        img = Image.open(ip).convert("RGB")
        w, h = img.size; scale = 192 / max(w, h)
        if scale < 1: img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        # Build prompt + labels (single processor call)
        img_np = np.array(img)
        img_inputs = proc.image_processor(images=[img_np], return_tensors="np")
        igt = img_inputs["image_grid_thw"][0]
        n_patches = int(igt[1]) * int(igt[2])
        n_copies = max(1, n_patches // 4)
        label = s["messages"][1]["content"]
        label_ids = proc.tokenizer.encode(label) + [proc.tokenizer.eos_token_id or 2]
        label_tensor = paddle.to_tensor(label_ids, dtype="int64")

        # One call: get text + image tensors in one go
        inp = proc(text=[f"{'<|placeholder|>'*n_copies}OCR:"], images=[img_np],
                   return_tensors="np", padding=True, max_length=2048, truncation=True)
        inp_pd = to_pd(inp)

        # Concatenate prompt + label, mask prompt with -100
        prompt_len = inp_pd["input_ids"].shape[1]
        full_ids = paddle.concat([inp_pd["input_ids"][0], label_tensor])
        full_labels = paddle.concat([paddle.full([prompt_len], -100, dtype="int64"), label_tensor])
        inp_pd["input_ids"] = full_ids.unsqueeze(0)
        inp_pd["labels"] = full_labels.unsqueeze(0)
        inp_pd["attention_mask"] = paddle.ones([full_ids.shape[0]], dtype="int64").unsqueeze(0)
        # Remove pixel_values from text call (use image_feats directly)
        inp_pd["pixel_values"] = paddle.to_tensor(img_inputs["pixel_values"])
        inp_pd["image_grid_thw"] = paddle.to_tensor(img_inputs["image_grid_thw"])

        out = model(**inp_pd)
        loss_val = out[0] if isinstance(out, (list, tuple)) else out.loss
        loss_val.backward()
        paddle.nn.utils.clip_grad_norm_(tp, 1.0)
        opt.step(); lrs.step(); opt.clear_grad()
        global_step += 1
        el += loss_val.item()

        del out, inp_pd, full_ids, full_labels, label_tensor
        paddle.device.cuda.empty_cache()

        if global_step % 50 == 0 and global_step > 0:
            avg_loss = el / max(1, (i+1))
            eta = (time.time()-t0)/max(1,global_step) * (total_steps-global_step)/60
            log(f"E{epoch+1}/3 S{global_step}/{total_steps} loss={avg_loss:.4f} ETA={eta:.0f}min")

        if global_step > 0 and global_step % 400 == 0:
            log(f"Checkpoint S{global_step}...")
            model.eval()
            preds, refs = [], []
            eos = proc.tokenizer.eos_token_id or 2
            with paddle.no_grad():
                for s in val[:30]:
                    ip = s["images"][0]
                    if not os.path.exists(ip): ip = ip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
                    img = Image.open(ip).convert("RGB")
                    w, h = img.size; scale = 384 / max(w, h)
                    if scale < 1: img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                    # Get image patch count
                    img_np = np.array(img)
                    img_feats = proc.image_processor(images=[img_np], return_tensors="np")
                    n_patches = int(img_feats["image_grid_thw"][0][1]) * int(img_feats["image_grid_thw"][0][2])
                    n_copies = max(1, n_patches // 4)
                    img_tokens = "<|placeholder|>" * n_copies
                    inp = proc(text=[f"{img_tokens}OCR:"], images=[img_np],
                               return_tensors="np", padding=True, max_length=2048, truncation=True)
                    inp_pd = to_pd(inp)
                    gen = []
                    for _ in range(512):
                        o = model(**inp_pd)
                        logits = o[0] if isinstance(o, (list, tuple)) else o.logits
                        ntl = logits[:, -1, :]
                        for tid in set(gen):
                            s = float(ntl[0, tid])
                            ntl[0, tid] = s * 1.1 if s < 0 else s / 1.1
                        nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
                        if nt == eos: break
                        gen.append(nt)
                        inp_pd["input_ids"] = paddle.concat([inp_pd["input_ids"], paddle.to_tensor([[nt]])], axis=1)
                        inp_pd["attention_mask"] = paddle.concat([inp_pd["attention_mask"], paddle.ones([1, 1])], axis=1)
                    preds.append(proc.tokenizer.decode(gen, skip_special_tokens=True))
                    refs.append(s["messages"][1]["content"])
            model.train()
            m = compute_all(preds, refs, label=f"s{global_step}")
            log(f"  Val: CompF1={m['component_f1']:.4f} jf1={m['joint_f1']:.4f} NED={m['ned']:.4f} RepRate={m['repetition_rate']:.2%}")

            # Save
            ld = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
            paddle.save(ld, f"{CKPT_DIR}/lora_s{global_step}.pdparams")

    log(f"Epoch {epoch+1}: {(time.time()-t0)/60:.1f}min")

log(f"Done! {(time.time()-t0)/60:.1f}min")
ld = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
paddle.save(ld, f"{CKPT_DIR}/lora_final.pdparams")
