"""Fix: strip 'model.' prefix from saved keys, merge LoRA, save in model-native format."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PATH"] = r"E:\080000software\080900_Miniconda\miniconda3\Library\bin;" + os.environ.get("PATH", "")
import numpy as np
import paddle
from pathlib import Path

# Monkey-patches
import paddle.distributed.fleet.meta_parallel as mp
if not hasattr(mp, 'LocalSharedLayerDesc'):
    class _LocalSharedLayerDesc:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    mp.LocalSharedLayerDesc = _LocalSharedLayerDesc

from types import ModuleType
try:
    import paddle.distributed.flex_checkpoint.dcp.sharded_weight
except Exception:
    dummy = ModuleType('dummy')
    for f in ['build_sharded_state_dict','create_sharded_weight_with_new_local',
              'reshape_sharded_weight','sharded_weight_parallel_cpu',
              'save_state_dict','load_state_dict']:
        setattr(dummy, f, lambda *a, **kw: None)
    for m in ['paddle.distributed.flex_checkpoint','paddle.distributed.flex_checkpoint.dcp',
              'paddle.distributed.flex_checkpoint.dcp.sharded_weight']:
        import sys
        sys.modules.setdefault(m, dummy)

paddle.float8_e4m3fn = paddle.float32; paddle.float8_e5m2 = paddle.float32
paddle.LongTensor = paddle.Tensor
paddle.linalg.fp8_fp8_half_gemm_fused = None
paddle.Tensor.long = lambda s: s.astype("int64")
paddle.Tensor.float = lambda s: s.astype("float32")
paddle.Tensor.half = lambda s: s.astype("float16")

_old_reshape = paddle.Tensor.reshape
def _pr(self, *args, **kwargs):
    if args:
        if isinstance(args[0], paddle.dtype): return self.astype(args[0])
        if len(args) > 1: new_shape = list(args)
        elif len(args) == 1 and (isinstance(args[0], int) or hasattr(args[0], '__index__')):
            new_shape = [int(args[0])]
        else: new_shape = args[0]
        return _old_reshape(self, new_shape, **kwargs)
    return _old_reshape(self, **kwargs)
paddle.Tensor.reshape = _pr; paddle.Tensor.view = _pr
if not hasattr(paddle.Tensor, "repeat"): paddle.Tensor.repeat = paddle.Tensor.tile

_old_transpose = paddle.Tensor.transpose
def _pt(self, *args, **kwargs):
    if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
        dim0, dim1 = args[0], args[1]; ndim = self.ndim
        if dim0 < 0: dim0 += ndim
        if dim1 < 0: dim1 += ndim
        perm = list(range(ndim)); perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return _old_transpose(self, perm, **kwargs)
    return _old_transpose(self, *args, **kwargs)
paddle.Tensor.transpose = _pt

_old_gelu = paddle.nn.functional.gelu
paddle.nn.functional.gelu = lambda x, approximate=False, name=None: _old_gelu(x, approximate == 'tanh' if isinstance(approximate, str) else approximate, name)

for nm in ['empty','zeros','ones','arange','full','randn','rand']:
    if hasattr(paddle, nm):
        of = getattr(paddle, nm)
        setattr(paddle, nm, lambda *a, _of=of, **kw: _of(*a, **{k: v for k, v in kw.items() if k != 'device'}))

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

print("[Patches] OK", flush=True)
paddle.set_device("gpu")

from paddleformers.transformers import AutoModelForConditionalGeneration

LOCAL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
LORA_DIR = "g:/mimo_project/circuit_ocr/circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr"
OUTPUT = f"{LORA_DIR}/base_with_lora_merged.pdparams"

LORA_SCALE = 2.0

# Load base model
print("Loading base model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    LOCAL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="float32"
)
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

# Get base model state dict
base_sd = model.state_dict()
print(f"Base model has {len(base_sd)} parameters")

# Show first few base model keys to understand naming
print("Base model key samples:")
for k in sorted(base_sd.keys())[:8]:
    print(f"  {k}")

# Load LoRA weights
print("\nLoading LoRA weights...")
lora_state = paddle.load(f"{LORA_DIR}/lora_weights_f32.pdparams")

# Strip 'model.' prefix and extract LoRA pairs
lora_pairs = {}
for k, v in lora_state.items():
    if not k.endswith('.lora_A') and not k.endswith('.lora_B'):
        continue
    # Strip 'model.' prefix if present
    clean_k = k[6:] if k.startswith('model.') else k
    if clean_k.endswith('.lora_A'):
        base_name = clean_k[:-len('.lora_A')]
        lora_pairs.setdefault(base_name, {})['A'] = v.numpy()
    elif clean_k.endswith('.lora_B'):
        base_name = clean_k[:-len('.lora_B')]
        lora_pairs.setdefault(base_name, {})['B'] = v.numpy()

print(f"Found {len(lora_pairs)} LoRA adapter pairs (after stripping 'model.' prefix)")

# Merge into base model state dict
merged = 0
skipped = 0
for lora_base, adapters in lora_pairs.items():
    if 'A' not in adapters or 'B' not in adapters:
        continue

    lora_A = adapters['A']
    lora_B = adapters['B']
    delta = (lora_A @ lora_B) * LORA_SCALE

    weight_key = f"{lora_base}.weight"
    if weight_key in base_sd:
        W = base_sd[weight_key].numpy()
        if delta.shape == W.shape:
            base_sd[weight_key].set_value(paddle.to_tensor(W + delta.astype(np.float32)))
            merged += 1
        else:
            if skipped < 3:
                print(f"  Shape mismatch: {weight_key}: delta={delta.shape} vs W={W.shape}")
            skipped += 1
    else:
        if skipped < 3:
            print(f"  Key not found: {weight_key}")
        skipped += 1

print(f"\nMerged: {merged}, Skipped: {skipped}")

if merged > 0:
    print(f"Saving to {OUTPUT}...")
    paddle.save(base_sd, OUTPUT)
    sz_gb = Path(OUTPUT).stat().st_size / 1e9
    print(f"Saved: {sz_gb:.2f} GB")
else:
    print("No merges performed - cannot continue")
print("Done!")
