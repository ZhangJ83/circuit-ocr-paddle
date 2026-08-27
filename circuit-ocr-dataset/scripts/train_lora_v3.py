#!/usr/bin/env python3
"""PaddleOCR-VL LoRA Training — Manual LoRA implementation.
Applies LoRA to model.layers.*.self_attn.{q,k,v,o}_proj with correct names."""
import os, sys, json, time, math, argparse, random
from pathlib import Path
from datetime import datetime

os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HOME": "/mnt/f/hf_cache/hub",
    "PADDLE_HOME": "/mnt/f/paddle_cache", "HF_HUB_CACHE": "/mnt/f/hf_cache/hub",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
})
DATASET_DIR = "/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset"
LOG_FILE = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr/training.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

# ====== PATCHES ======
import paddle
import paddle.distributed.fleet.meta_parallel as mp
if not hasattr(mp, 'LocalSharedLayerDesc'):
    class _LocalSharedLayerDesc:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    mp.LocalSharedLayerDesc = _LocalSharedLayerDesc

from types import ModuleType
try: import paddle.distributed.flex_checkpoint.dcp.sharded_weight
except:
    dummy = ModuleType('dummy')
    for f in ['build_sharded_state_dict','create_sharded_weight_with_new_local',
              'reshape_sharded_weight','sharded_weight_parallel_cpu',
              'save_state_dict','load_state_dict']:
        setattr(dummy, f, lambda *a, **kw: None)
    for m in ['paddle.distributed.flex_checkpoint','paddle.distributed.flex_checkpoint.dcp',
              'paddle.distributed.flex_checkpoint.dcp.sharded_weight']:
        sys.modules.setdefault(m, dummy)

paddle.float8_e4m3fn = paddle.float32
paddle.float8_e5m2 = paddle.float32
paddle.LongTensor = paddle.Tensor
paddle.linalg.fp8_fp8_half_gemm_fused = None
paddle.Tensor.long = lambda s: s.astype("int64")
paddle.Tensor.float = lambda s: s.astype("float32")
paddle.Tensor.half = lambda s: s.astype("float16")

_old_reshape = paddle.Tensor.reshape
def _patched_reshape(self, *args, **kwargs):
    if args:
        if isinstance(args[0], paddle.dtype): return self.astype(args[0])
        if len(args) > 1: new_shape = list(args)
        elif len(args) == 1 and (isinstance(args[0], int) or hasattr(args[0], '__index__')):
            new_shape = [int(args[0])]
        else: new_shape = args[0]
        return _old_reshape(self, new_shape, **kwargs)
    return _old_reshape(self, **kwargs)
paddle.Tensor.reshape = _patched_reshape
paddle.Tensor.view = _patched_reshape
if not hasattr(paddle.Tensor, "repeat"): paddle.Tensor.repeat = paddle.Tensor.tile

_old_transpose = paddle.Tensor.transpose
def _patched_transpose(self, *args, **kwargs):
    if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
        dim0, dim1 = args[0], args[1]; ndim = self.ndim
        if dim0 < 0: dim0 += ndim
        if dim1 < 0: dim1 += ndim
        perm = list(range(ndim)); perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return _old_transpose(self, perm, **kwargs)
    return _old_transpose(self, *args, **kwargs)
paddle.Tensor.transpose = _patched_transpose

def _patched_masked_scatter(self, mask, source):
    orig = self.shape; mask = mask.astype('bool')
    flat_self, flat_mask, flat_src = self.flatten(), mask.flatten(), source.flatten()
    idx = paddle.nonzero(flat_mask)
    scat = paddle.scatter_nd(idx, flat_src, flat_mask.shape)
    return paddle.where(flat_mask, scat, flat_self).reshape(orig)
paddle.Tensor.masked_scatter = _patched_masked_scatter

_old_gf = paddle.base.framework.get_flags
paddle.base.framework.get_flags = lambda flags: {f: 2 if f == "FLAGS_flash_attn_version" else _old_gf([f]).get(f) for f in flags}
_old_sf = paddle.set_flags
paddle.set_flags = lambda d: _old_sf({k: v for k, v in d.items() if k != "FLAGS_flash_attn_version"}) if {k: v for k, v in d.items() if k != "FLAGS_flash_attn_version"} else None

_old_gelu = paddle.nn.functional.gelu
paddle.nn.functional.gelu = lambda x, approximate=False, name=None: _old_gelu(x, approximate == 'tanh' if isinstance(approximate, str) else approximate, name)

for nm in ['empty','zeros','ones','arange','full','randn','rand']:
    if hasattr(paddle, nm):
        of = getattr(paddle, nm)
        setattr(paddle, nm, lambda *a, _of=of, **kw: _of(*a, **{k: v for k, v in kw.items() if k != 'device'}))

paddle.nn.functional.swiglu = lambda *a, **kw: None

def _frms(x, w, eps=1e-6):
    v = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
    r = paddle.rsqrt(v + eps); return (x * r * w, r)
paddle.incubate.nn.functional.fused_rms_norm_ext = _frms

def _fma(q, k, v, startend_row_indices=None, causal=True):
    qt, kt, vt = q.transpose([0,2,1,3]), k.transpose([0,2,1,3]), v.transpose([0,2,1,3])
    b, hq, lq, d = qt.shape; _, hk, lk, _ = kt.shape
    if hq != hk:
        nr = hq // hk
        kt = paddle.tile(kt.reshape([b,hk,1,lk,d]), [1,1,nr,1,1]).reshape([b,hq,lk,d])
        vt = paddle.tile(vt.reshape([b,hk,1,lk,d]), [1,1,nr,1,1]).reshape([b,hq,lk,d])
    uc = causal and lq == lk; am = None
    if causal and not uc:
        ri = paddle.arange(lq, dtype='int32').reshape([1,1,lq,1])
        ci = paddle.arange(lk, dtype='int32').reshape([1,1,1,lk])
        cb = ci <= (lk - lq + ri)
        am = paddle.where(cb, paddle.zeros([1,1,lq,lk], dtype=q.dtype), paddle.full([1,1,lq,lk], -1e9, dtype=q.dtype))
        if b > 1: am = paddle.tile(am, [b,1,1,1])
    try:
        return paddle.nn.functional.scaled_dot_product_attention(qt,kt,vt,attn_mask=am,is_causal=uc,training=False).transpose([0,2,1,3])
    except:
        scores = paddle.matmul(qt, kt.transpose([0,1,3,2])) / (d ** 0.5)
        if am is not None: scores = scores + am
        if uc:
            gq = paddle.arange(lq, dtype="int32").reshape([lq,1])
            gk = paddle.arange(lk, dtype="int32").reshape([1,lk])
            scores = paddle.where((gk-gq) <= (lk-lq), scores, paddle.to_tensor(-1e9, dtype=scores.dtype))
        return paddle.matmul(paddle.nn.functional.softmax(scores, axis=-1), vt).transpose([0,2,1,3])
paddle.nn.functional.flash_attention.flashmask_attention = _fma
paddle.incubate.tensor.manipulation.create_async_load = lambda *a, **kw: None

log("[Patches] OK")

# ====== MANUAL LORA ======

class LoRALinear(paddle.nn.Layer):
    """LoRA wrapper for a nn.Linear layer."""
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = paddle.nn.Dropout(dropout)
        in_features = base_linear.weight.shape[0]  # Paddle: weight shape is [out, in]? Actually [in, out]
        out_features = base_linear.weight.shape[1]
        # PaddlePaddle Linear weight is [in_features, out_features]
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            default_initializer=paddle.nn.initializer.KaimingUniform(negative_slope=math.sqrt(5))
        )
        self.lora_B = self.create_parameter(
            shape=[r, out_features],
            default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        # Freeze base weight
        self.base.weight.stop_gradient = True
        if self.base.bias is not None:
            self.base.bias.stop_gradient = True

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model, r=8, alpha=16, dropout=0.05, target_suffixes=None):
    """Apply LoRA to attention projections in model.layers.*.self_attn."""
    if target_suffixes is None:
        target_suffixes = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    lora_params = []

    for layer_name, layer in model.named_sublayers():
        if not isinstance(layer, paddle.nn.Linear):
            continue
        # ONLY apply to language model attention (model.layers.X.self_attn.*_proj)
        # NOT vision encoder (visual.vision_model.encoder.layers.X.self_attn.*_proj)
        parts = layer_name.split('.')
        # Pattern: model.layers.{N}.self_attn.{q,k,v,o}_proj
        if not (len(parts) >= 4 and parts[0] == 'model' and parts[1] == 'layers'
                and parts[3] == 'self_attn' and parts[4] in target_suffixes):
            continue

        # Get parent module
        *parent_parts, child_name = parts
        parent = model
        for p in parent_parts:
            parent = getattr(parent, p)

        lora_linear = LoRALinear(layer, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_linear)
        lora_params.append(layer_name)
        log(f"  LoRA applied: {layer_name} [{layer.weight.shape[0]}x{layer.weight.shape[1]}]")

    return lora_params


def compute_loss(model, processor, sample):
    from PIL import Image
    query = sample["messages"][0]["content"]
    label = sample["messages"][1]["content"]
    img_path = sample["images"][0]
    if not img_path.startswith('/'):
        img_path = f"{DATASET_DIR}/{img_path.lstrip('./')}"
    if not Path(img_path).exists(): return None
    try:
        image = Image.open(img_path).convert("RGB")
        # CRITICAL: Resize to 224px max to fit 8GB VRAM
        w, h = image.size
        max_dim = max(w, h)
        if max_dim > 224:
            scale = 224.0 / max_dim
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query.replace("<image>", "")}]}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
        lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=1024)
        loss = model(**inputs, labels=lt["input_ids"][0].unsqueeze(0)).loss
        try: image.close()
        except: pass
        paddle.device.cuda.empty_cache()
        return loss
    except Exception as e:
        try: image.close()
        except: pass
        paddle.device.cuda.empty_cache()
        return None


def count_trainable(model):
    total = sum(p.size for p in model.parameters())
    tr = sum(p.size for p in model.parameters() if not p.stop_gradient)
    return total, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--grad_accum', type=int, default=64)
    ap.add_argument('--output_dir', type=str, default=f'{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr')
    ap.add_argument('--max_eval_samples', type=int, default=50)
    ap.add_argument('--train_data', type=str, default=f'{DATASET_DIR}/ocr_vl_sft-train.jsonl')
    ap.add_argument('--eval_data', type=str, default=f'{DATASET_DIR}/ocr_vl_sft-test.jsonl')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paddle.set_device('gpu')
    log(f"GPU: {paddle.device.cuda.get_device_name(0)}")

    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
    from paddleformers.generation import GenerationConfig

    mpth = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"
    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(mpth)
    log("Loading model (bf16)...")
    model = AutoModelForConditionalGeneration.from_pretrained(mpth, convert_from_hf=True, load_checkpoint_format='naive', low_cpu_mem_usage=True, dtype="bfloat16")
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    log("Model loaded OK")

    # Apply manual LoRA
    log(f"Applying LoRA (r={args.rank}, alpha={args.alpha})...")
    lora_layers = apply_lora_to_model(model, r=args.rank, alpha=args.alpha, target_suffixes=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    log(f"Applied LoRA to {len(lora_layers)} layers")

    # Freeze ALL parameters except LoRA weights
    for p in model.parameters():
        p.stop_gradient = True
    # Unfreeze only lora_A and lora_B
    for name, p in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            p.stop_gradient = False

    total, tr = count_trainable(model)
    log(f"Params: Total={total:,} Trainable={tr:,} ({100*tr/total:.4f}%)")

    if tr == 0:
        log("ERROR: No trainable parameters! Exiting.")
        return

    train = [json.loads(l) for l in open(args.train_data) if l.strip()]
    evals = [json.loads(l) for i, l in enumerate(open(args.eval_data)) if l.strip() and i < args.max_eval_samples]
    log(f"Data: Train={len(train)} Eval={len(evals)}")

    spe = max(1, len(train) // args.grad_accum)
    lr_s = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=args.epochs * spe, eta_min=5e-5)
    opt = paddle.optimizer.AdamW(learning_rate=lr_s, parameters=[p for p in model.parameters() if not p.stop_gradient], weight_decay=0.1, beta1=0.9, beta2=0.95, epsilon=1e-8)

    hist, gs, best = [], 0, float('inf')
    t0 = time.time()
    log(f"{'='*60}")
    log(f"LoRA r={args.rank} epochs={args.epochs} samples={len(train)} grad_accum={args.grad_accum} steps/epoch={spe}")
    log(f"{'='*60}")

    for ep in range(args.epochs):
        e0 = time.time(); sl = 0.0; random.shuffle(train); model.train()
        log(f"[Epoch {ep+1}] Starting, {len(train)} samples...")
        for i, s in enumerate(train):
            loss = compute_loss(model, processor, s)
            if loss is None:
                if i < 5: log(f"  [DBG] Sample {i} returned None loss")
                continue
            if i < 5: log(f"  [DBG] Sample {i} loss={float(loss):.4f}")
            loss = loss / args.grad_accum; loss.backward(); sl += float(loss) * args.grad_accum
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train):
                opt.step(); lr_s.step(); opt.clear_grad(); gs += 1
                al = sl / args.grad_accum; sl = 0.0
                hist.append({'step': gs, 'epoch': ep + 1, 'loss': al, 'lr': opt.get_lr(), 'time': time.time() - t0})
                el = (time.time() - t0) / 60
                eta = (el / gs) * (args.epochs * spe) - el if gs > 0 else 0
                log(f"[E{ep+1} S{gs:3d}/{args.epochs * spe}] loss={al:.4f} lr={opt.get_lr():.2e} {el:.0f}m ETA={eta:.0f}m")

                if gs % 50 == 0 and evals:
                    model.eval()
                    els = []
                    for es in evals[:10]:
                        eloss = compute_loss(model, processor, es)
                        if eloss is not None:
                            els.append(float(eloss))
                    model.train()
                    if els:
                        ae = sum(els) / len(els)
                        log(f"  [Eval S{gs}] loss={ae:.4f}")
                        if ae < best:
                            best = ae
                            paddle.save(model.state_dict(), f"{args.output_dir}/best_model.pdparams")
                            log(f"  [Save] Best @ S{gs}")
                if gs % 50 == 0:
                    paddle.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'step': gs, 'epoch': ep+1, 'loss_history': hist}, f"{args.output_dir}/checkpoint-{gs}.pdparams")
                    json.dump(hist, open(f"{args.output_dir}/loss_history.json", 'w'), indent=2)
                    log(f"  [Ckpt] Saved @ S{gs}")

        log(f"[Epoch {ep+1}] Done in {(time.time()-e0)/60:.1f}m")

    tm = (time.time() - t0) / 60
    log(f"{'='*60}")
    log(f"Done! {tm:.0f}m ({tm/60:.1f}h) Best eval: {best:.4f}")
    log(f"{'='*60}")
    paddle.save(model.state_dict(), f"{args.output_dir}/final_model.pdparams")
    json.dump(hist, open(f"{args.output_dir}/loss_history.json", 'w'), indent=2)
    log("Complete.")

if __name__ == '__main__': main()
