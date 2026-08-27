"""Step-by-step LoRA merge + forward diagnostic."""
import os, sys, json, time, traceback
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "F:/hf_cache/hub"
os.environ["PADDLE_HOME"] = "F:/paddle_cache"
os.environ["HF_HUB_CACHE"] = "F:/hf_cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PATH"] = (
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin;"
    r"E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\Lib\site-packages\torch\lib;"
    + os.environ.get("PATH", "")
)

def step(n, msg):
    print(f"[STEP {n}] {msg}", flush=True)

step(1, "Importing paddle...")
import paddle
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

def _pms(self, mask, source):
    orig = self.shape; mask = mask.astype('bool')
    fs, fm, fsrc = self.flatten(), mask.flatten(), source.flatten()
    idx = paddle.nonzero(fm)
    scat = paddle.scatter_nd(idx, fsrc, fm.shape)
    return paddle.where(fm, scat, fs).reshape(orig)
paddle.Tensor.masked_scatter = _pms

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

import numpy as np, tempfile
from safetensors.numpy import save_file, safe_open
tmp_path = tempfile.mktemp(suffix='.safetensors')
save_file({'dummy': np.zeros((1,))}, tmp_path)
with safe_open(tmp_path, framework='np') as f:
    PySafeSlice = type(f.get_slice('dummy'))
    setattr(PySafeSlice, 'shape', property(lambda self: self.get_shape()))
os.remove(tmp_path)

step(2, "Patches done, setting GPU...")
paddle.set_device("gpu")

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig
from PIL import Image
from pathlib import Path

LOCAL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
DATA_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
LORA_DIR = f"{DATA_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
LORA_SCALE = 2.0

step(3, "Loading processor...")
processor = AutoProcessor.from_pretrained(LOCAL_PATH)

step(4, "Loading base model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    LOCAL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="float32"
)
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
step(5, "Base model loaded OK")

# === Test forward pass on base model first ===
step(6, "Testing base model forward pass...")
samples = [json.loads(l) for l in open(f"{DATA_DIR}/ocr_vl_sft-test.jsonl", encoding="utf-8") if l.strip()]
sample = samples[0]
img_path = sample["images"][0]
if not img_path.startswith("/"):
    img_path = f"{DATA_DIR}/{img_path.lstrip('./')}"
image = Image.open(img_path).convert("RGB")
w, h = image.size
max_dim = 768
if max(w, h) > max_dim:
    scale = max_dim / max(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

query = sample["messages"][0]["content"].replace("<image>", "")
msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query}]}]
inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
step(7, f"Inputs prepared: input_ids={list(inputs['input_ids'].shape)}")

with paddle.no_grad():
    out = model(input_ids=inputs["input_ids"], pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"), use_cache=False)
logits = out[0] if isinstance(out, tuple) else out.logits
step(8, f"Base model forward OK, logits shape={list(logits.shape)}")

# === Now merge LoRA ===
step(9, "Loading LoRA weights...")
lora_state = paddle.load(f"{LORA_DIR}/lora_weights_f32.pdparams")
lora_pairs = {}
for k, v in lora_state.items():
    if k.endswith('.lora_A'):
        bn = k[:-len('.lora_A')]; clean = bn[6:] if bn.startswith('model.') else bn
        lora_pairs.setdefault(clean, {})['A'] = v.numpy()
    elif k.endswith('.lora_B'):
        bn = k[:-len('.lora_B')]; clean = bn[6:] if bn.startswith('model.') else bn
        lora_pairs.setdefault(clean, {})['B'] = v.numpy()
step(10, f"Parsed {len(lora_pairs)} LoRA pairs")

base_sd = model.state_dict()
merged = 0
for lora_base, adapters in lora_pairs.items():
    if 'A' not in adapters or 'B' not in adapters: continue
    la, lb = adapters['A'], adapters['B']
    if la.shape[-1] != lb.shape[0]: continue
    delta = la @ lb * LORA_SCALE
    wk = f"{lora_base}.weight"
    if wk in base_sd:
        W = base_sd[wk].numpy()
        if delta.shape == W.shape:
            base_sd[wk].set_value(paddle.to_tensor(W + delta.astype(np.float32), place=base_sd[wk].place))
            merged += 1
model.eval()
step(11, f"Merged {merged}/{len(lora_pairs)} adapters")

# === Test forward pass AFTER merge ===
step(12, "Testing LoRA model forward pass...")
paddle.device.cuda.empty_cache()
with paddle.no_grad():
    out2 = model(input_ids=inputs["input_ids"], pixel_values=inputs.get("pixel_values"),
                 image_grid_thw=inputs.get("image_grid_thw"), use_cache=False)
logits2 = out2[0] if isinstance(out2, tuple) else out2.logits
step(13, f"LoRA model forward OK, logits shape={list(logits2.shape)}")

# === Manual decode test ===
step(14, "Manual decode 10 tokens...")
current_ids = inputs["input_ids"]
gen_ids = []
for i in range(10):
    with paddle.no_grad():
        out = model(input_ids=current_ids, pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"), use_cache=False)
    logits = out[0] if isinstance(out, tuple) else out.logits
    nt = int(paddle.argmax(logits[0, -1, :]).item())
    gen_ids.append(nt)
    current_ids = paddle.concat([current_ids, paddle.to_tensor([[nt]], dtype=current_ids.dtype)], axis=1)
    if nt == 2:  # eos
        step(14, f"EOS at token {i+1}")
        break

result = processor.decode(gen_ids, skip_special_tokens=True)
step(15, f"LoRA decode result: {result[:200]}")
step(16, "ALL DONE - LoRA inference works!")
