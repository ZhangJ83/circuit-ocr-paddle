"""Training script — battle-tested on RTX 4090D 24GB."""
import os, sys, json, time, random, argparse
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Patches: must be exactly this order
sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu, 'get_one_valid_tokenizer_file'):
    mu.get_one_valid_tokenizer_file = lambda d, e: list(mu._filter_valid_tokenizer_files(d, e))

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

# Patch: Paddle 3.2.0 scaled_dot_product_attention doesn't accept enable_gqa (added in 3.3+)
_orig_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _patched_sdpa

import numpy as np
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.generation import GenerationConfig

# Restore torchvision after paddleformers blocked it
sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms, torch
import transformers.utils.import_utils as tiu
tiu.is_torch_available = lambda: (True, '')
tiu.is_torchvision_available = lambda: (True, '')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))
from eval_metrics import compute_all

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def to_pd(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray): out[k] = paddle.to_tensor(v)
        elif isinstance(v, torch.Tensor): out[k] = paddle.to_tensor(v.numpy())
        elif isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], np.ndarray): out[k] = paddle.to_tensor(np.array(v))
            elif isinstance(v[0], torch.Tensor): out[k] = paddle.to_tensor(np.array([x.numpy() for x in v]))
            else: out[k] = v
        else: out[k] = v
    return out


def train(args):
    name = args.name or "unnamed"
    ckpt_dir = os.path.join(args.output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    log(f"=== {name} === epochs={args.epochs} dim={args.max_dim} r={args.rank} freeze_proj={args.freeze_projector} lr={args.lr}")

    train_data = [json.loads(l) for l in open(args.train_data) if l.strip()]
    val_data = [json.loads(l) for l in open(args.val_data) if l.strip()]
    random.shuffle(train_data)
    log(f"Train: {len(train_data)}, Val: {len(val_data)}")

    log("Loading model...")
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForConditionalGeneration.from_pretrained(MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive", low_cpu_mem_usage=True, dtype="bfloat16")
    model.config._attn_implementation = "sdpa"
    model.visual.config._attn_implementation = "sdpa"

    # NOTE: recompute/gradient_checkpointing not available on PaddleOCR-VL model
    # Memory savings come from auto_growth + smaller max_length instead
    log(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Default: freeze projector and mlp_AR (vision→LLM bridge)
    if args.freeze_projector:
        for n, p in model.named_parameters():
            if "mlp_AR" in n or "projector" in n: p.stop_gradient = True

    lc = LoRAConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                    target_modules=["model\\.model\\..*q_proj","model\\.model\\..*k_proj","model\\.model\\..*v_proj","model\\.model\\..*o_proj","model\\.model\\..*linear_1","model\\.model\\..*linear_2"])
    model = LoRAModel(model, lc)
    if not hasattr(model.model, 'full'):
        model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
    if not args.freeze_projector:
        log("UNFREEZING projector + mlp_AR")
        for n, p in model.named_parameters():
            if "mlp_AR" in n or "projector" in n: p.stop_gradient = False

    tp = [p for p in model.parameters() if not p.stop_gradient]
    log(f"Trainable: {sum(p.numel() for p in tp):,} ({sum(p.numel() for p in tp)/sum(p.numel() for p in model.parameters())*100:.2f}%)")

    total_steps = len(train_data) * args.epochs
    cd = paddle.optimizer.lr.CosineAnnealingDecay(args.lr, T_max=max(1, total_steps - args.warmup), eta_min=args.lr/10)
    lrs = paddle.optimizer.lr.LinearWarmup(cd, warmup_steps=args.warmup, start_lr=args.lr/10, end_lr=args.lr)
    opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)
    log(f"Steps: {total_steps}")

    mh, best_loss = [], float('inf')
    gs, t0 = 0, time.time()

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        el = 0.0
        for i, s in enumerate(train_data):
            ip = s["images"][0]
            if not os.path.exists(ip): ip = ip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
            img = Image.open(ip).convert("RGB")
            w, h = img.size
            if args.max_dim > 0 and max(w, h) > args.max_dim:
                scale = args.max_dim / max(w, h)
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            img_np = np.array(img)

            img_inputs = proc.image_processor(images=[img_np], return_tensors="np")
            igt = img_inputs["image_grid_thw"][0]
            n_patches = int(igt[1]) * int(igt[2])
            n_copies = max(1, n_patches // 4)

            label = s["messages"][1]["content"]
            label_ids = proc.tokenizer.encode(label) + [proc.tokenizer.eos_token_id or 2]
            label_tensor = paddle.to_tensor(label_ids, dtype="int64")

            inp = proc(text=[f"{'<|placeholder|>'*n_copies}OCR:"], images=[img_np],
                       return_tensors="np", padding=True, max_length=1024, truncation=True)
            inp_pd = to_pd(inp)

            prompt_len = inp_pd["input_ids"].shape[1]
            inp_pd["input_ids"] = paddle.concat([inp_pd["input_ids"][0], label_tensor]).unsqueeze(0)
            inp_pd["labels"] = paddle.concat([paddle.full([prompt_len], -100, dtype="int64"), label_tensor]).unsqueeze(0)
            inp_pd["attention_mask"] = paddle.ones([1, inp_pd["input_ids"].shape[1]], dtype="int64")
            inp_pd["pixel_values"] = paddle.to_tensor(img_inputs["pixel_values"]) if isinstance(img_inputs["pixel_values"], np.ndarray) else paddle.to_tensor(img_inputs["pixel_values"].numpy())
            inp_pd["image_grid_thw"] = paddle.to_tensor(img_inputs["image_grid_thw"]) if isinstance(img_inputs["image_grid_thw"], np.ndarray) else paddle.to_tensor(img_inputs["image_grid_thw"].numpy())

            out = model(**inp_pd)
            loss_val = out[0] if isinstance(out, (list, tuple)) else out.loss
            loss_val.backward()
            paddle.nn.utils.clip_grad_norm_(tp, 1.0)
            opt.step(); lrs.step(); opt.clear_grad()
            gs += 1; el += loss_val.item()

            del out, inp_pd, label_tensor

            if gs % 50 == 0 and gs > 0:
                eta = (time.time()-t0)/max(1,gs) * (total_steps-gs)/60
                log(f"E{epoch+1}/{args.epochs} S{gs}/{total_steps} loss={el/max(1,i+1):.4f} ETA={eta:.0f}min")

            if gs > 0 and gs % args.checkpoint_steps == 0:
                train_loss = el / max(1, i+1)
                log(f"Checkpoint S{gs} loss={train_loss:.4f}")

                # Save LoRA weights (~5.7MB)
                ld = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                paddle.save(ld, os.path.join(ckpt_dir, f"checkpoint_s{gs}.pdparams"))

                # Keep best by training loss
                if train_loss < best_loss:
                    best_loss = train_loss
                    paddle.save(ld, os.path.join(ckpt_dir, "best.pdparams"))
                    log(f"  *** BEST loss={best_loss:.4f} ***")

        log(f"Epoch {epoch+1}: {(time.time()-t0)/60:.1f}min total")

    tt = (time.time()-t0)/60
    log(f"Done {tt:.1f}min. Best loss={best_loss:.4f}. Checkpoints in {ckpt_dir}")
    return best_loss, ckpt_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--output_dir", default=f"{PROJECT_DIR}/checkpoints")
    ap.add_argument("--train_data", default=f"{PROJECT_DIR}/output/train_clean.jsonl")
    ap.add_argument("--val_data", default=f"{PROJECT_DIR}/output/val_clean.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_dim", type=int, default=384)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--freeze_projector", type=int, default=1)
    ap.add_argument("--checkpoint_steps", type=int, default=400)
    ap.add_argument("--val_samples", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()
    args.freeze_projector = bool(args.freeze_projector)
    train(args)
