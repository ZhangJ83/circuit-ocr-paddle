#!/usr/bin/env python3
"""Quick test: does compute_loss work with our LoRA model?"""
import os, sys, json, math
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

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

DATASET_DIR = "/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset"
mpth = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"

paddle.set_device('gpu')
print("Loading...")
processor = AutoProcessor.from_pretrained(mpth)
model = AutoModelForConditionalGeneration.from_pretrained(mpth, convert_from_hf=True, load_checkpoint_format='naive', low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
print("Model loaded")

# Apply LoRA (same as train_lora_v3)
class LoRALinear(paddle.nn.Layer):
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_linear
        self.scaling = alpha / r
        self.dropout = paddle.nn.Dropout(dropout)
        in_features = base_linear.weight.shape[0]
        out_features = base_linear.weight.shape[1]
        self.lora_A = self.create_parameter(shape=[in_features, r],
            default_initializer=paddle.nn.initializer.KaimingUniform(negative_slope=math.sqrt(5)))
        self.lora_B = self.create_parameter(shape=[r, out_features],
            default_initializer=paddle.nn.initializer.Constant(0.0))
        self.base.weight.stop_gradient = True
        if self.base.bias is not None: self.base.bias.stop_gradient = True
    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

count = 0
for name, layer in model.named_sublayers():
    if isinstance(layer, paddle.nn.Linear):
        parts = name.split('.')
        if parts[0] == 'model' and parts[1] == 'layers' and parts[3] == 'self_attn' and parts[4] in ['q_proj','k_proj','v_proj','o_proj']:
            parent = model
            for p in parts[:-1]: parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(layer))
            count += 1

# Freeze all, unfreeze only lora
for p in model.parameters(): p.stop_gradient = True
for n, p in model.named_parameters():
    if 'lora_A' in n or 'lora_B' in n: p.stop_gradient = False

tr = sum(p.size for p in model.parameters() if not p.stop_gradient)
print(f"LoRA: {count} layers, {tr} trainable params")

# Test compute_loss
from PIL import Image
from pathlib import Path

train = [json.loads(l) for l in open(f"{DATASET_DIR}/ocr_vl_sft-train.jsonl") if l.strip()]
s = train[0]
img_path = s["images"][0]
if not img_path.startswith('/'):
    img_path = f"{DATASET_DIR}/{img_path.lstrip('./')}"
print(f"Sample 0: {img_path} exists={Path(img_path).exists()}")

try:
    image = Image.open(img_path).convert("RGB")
    query = s["messages"][0]["content"]
    label = s["messages"][1]["content"]
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query.replace("<image>", "")}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    print(f"Input keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if hasattr(v, 'shape'): print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=2048)
    print(f"Label ids shape: {lt['input_ids'][0].shape}")

    print("Calling model forward...")
    model.eval()
    out = model(**inputs, labels=lt["input_ids"][0].unsqueeze(0))
    print(f"Output keys: {list(out.keys()) if hasattr(out, 'keys') else type(out)}")
    if hasattr(out, 'loss'):
        print(f"LOSS = {float(out.loss):.6f}")
    else:
        print(f"No loss in output! Output: {out}")
    image.close()
    paddle.device.cuda.empty_cache()
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("Done!")
