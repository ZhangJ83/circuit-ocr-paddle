#!/usr/bin/env python3
"""Find correct target_modules pattern for PaddleFormers LoRAConfig."""
import os, sys
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HOME": "/mnt/f/hf_cache/hub",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
})

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

paddle.float8_e4m3fn = paddle.float32; paddle.float8_e5m2 = paddle.float32
paddle.LongTensor = paddle.Tensor; paddle.linalg.fp8_fp8_half_gemm_fused = None
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
paddle.Tensor.reshape = _patched_reshape; paddle.Tensor.view = _patched_reshape
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

paddle.Tensor.masked_scatter = lambda self, mask, source: None
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
    try: return paddle.nn.functional.scaled_dot_product_attention(qt,kt,vt,is_causal=causal,training=False).transpose([0,2,1,3])
    except:
        scores = paddle.matmul(qt, kt.transpose([0,1,3,2])) / (d ** 0.5)
        attn = paddle.nn.functional.softmax(scores, axis=-1)
        return paddle.matmul(attn, vt).transpose([0,2,1,3])
paddle.nn.functional.flash_attention.flashmask_attention = _fma
paddle.incubate.tensor.manipulation.create_async_load = lambda *a, **kw: None

from paddleformers.transformers import AutoModelForConditionalGeneration
from paddleformers.peft import LoRAConfig, LoRAModel

mpth = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"

paddle.set_device('gpu')
print("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(mpth, convert_from_hf=True, load_checkpoint_format='naive', low_cpu_mem_usage=True, dtype="bfloat16")
print("Model loaded")

# Try different target_modules patterns
patterns = [
    ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
    ['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj'],
    ['model.layers.*.self_attn.q_proj', 'model.layers.*.self_attn.k_proj', 'model.layers.*.self_attn.v_proj', 'model.layers.*.self_attn.o_proj'],
]

for pat in patterns:
    try:
        # Fresh model each time (just re-apply LoRA)
        m2 = AutoModelForConditionalGeneration.from_pretrained(mpth, convert_from_hf=True, load_checkpoint_format='naive', low_cpu_mem_usage=True, dtype="bfloat16")
        lc = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=pat)
        m2 = LoRAModel(m2, lc)
        m2.mark_only_lora_as_trainable()
        total = sum(p.size for p in m2.parameters())
        tr = sum(p.size for p in m2.parameters() if not p.stop_gradient)
        print(f"  target_modules={pat}")
        print(f"    Total={total:,} Trainable={tr:,} ({100*tr/total:.4f}%)")
        del m2
        paddle.device.cuda.empty_cache()
    except Exception as e:
        print(f"  target_modules={pat} -> ERROR: {e}")
        paddle.device.cuda.empty_cache()

print("\nDone!")
