#!/usr/bin/env python3
"""Debug PaddleOCR-VL generation in WSL."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ===== Patches =====
from types import ModuleType
import paddle

try:
    import paddle.distributed.flex_checkpoint.dcp.sharded_weight
except:
    dummy = ModuleType("dummy")
    dummy.build_sharded_state_dict = lambda *a,**kw: None
    sys.modules.setdefault("paddle.distributed.flex_checkpoint", dummy)
    sys.modules.setdefault("paddle.distributed.flex_checkpoint.dcp", dummy)
    sys.modules.setdefault("paddle.distributed.flex_checkpoint.dcp.sharded_weight", dummy)

paddle.float8_e4m3fn = paddle.float32
paddle.float8_e5m2 = paddle.float32
paddle.LongTensor = paddle.Tensor
paddle.linalg.fp8_fp8_half_gemm_fused = None
paddle.Tensor.long = lambda self: self.astype("int64")
paddle.Tensor.float = lambda self: self.astype("float32")
paddle.Tensor.half = lambda self: self.astype("float16")

old_reshape = paddle.Tensor.reshape
old_view = paddle.Tensor.view
def patched_view(self, *args, **kwargs):
    if args and isinstance(args[0], paddle.dtype):
        return old_view(self, *args, **kwargs)
    if args:
        if len(args) > 1: new_shape = list(args)
        elif len(args) == 1 and (isinstance(args[0], int) or hasattr(args[0], "__index__")):
            new_shape = [int(args[0])]
        else: new_shape = args[0]
        return old_reshape(self, new_shape, **kwargs)
    return old_reshape(self, **kwargs)
paddle.Tensor.reshape = patched_view
paddle.Tensor.view = patched_view

if not hasattr(paddle.Tensor, "repeat"):
    paddle.Tensor.repeat = paddle.Tensor.tile

old_transpose = paddle.Tensor.transpose
def patched_transpose(self, *args, **kwargs):
    if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
        dim0, dim1 = args[0], args[1]
        ndim = self.ndim
        if dim0 < 0: dim0 += ndim
        if dim1 < 0: dim1 += ndim
        perm = list(range(ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return old_transpose(self, perm, **kwargs)
    return old_transpose(self, *args, **kwargs)
paddle.Tensor.transpose = patched_transpose

def patched_masked_scatter(self, mask, source):
    orig_shape = self.shape
    mask = mask.astype("bool")
    flat_self = self.flatten()
    flat_mask = mask.flatten()
    flat_source = source.flatten()
    indices = paddle.nonzero(flat_mask)
    scattered = paddle.scatter_nd(indices, flat_source, flat_mask.shape)
    out_flat = paddle.where(flat_mask, scattered, flat_self)
    return out_flat.reshape(orig_shape)
paddle.Tensor.masked_scatter = patched_masked_scatter

old_get_flags = paddle.base.framework.get_flags
def patched_get_flags(flags):
    res = {}
    for f in flags:
        if f == "FLAGS_flash_attn_version": res[f] = 2
        else:
            try: res[f] = old_get_flags([f])[f]
            except: res[f] = None
    return res
paddle.base.framework.get_flags = patched_get_flags

old_gelu = paddle.nn.functional.gelu
def patched_gelu(x, approximate=False, name=None):
    if isinstance(approximate, str): approximate = (approximate == "tanh")
    return old_gelu(x, approximate, name)
paddle.nn.functional.gelu = patched_gelu

def patch_creation_func(func_name):
    old_func = getattr(paddle, func_name)
    def patched(*args, **kwargs):
        kwargs.pop("device", None)
        return old_func(*args, **kwargs)
    setattr(paddle, func_name, patched)
for name in ["empty", "zeros", "ones", "arange", "full", "randn", "rand"]:
    if hasattr(paddle, name): patch_creation_func(name)

import paddle.nn.functional as pnf
pnf.swiglu = lambda *args, **kwargs: None

def fallback_fused_rms_norm_ext(x, weight, epsilon=1e-6):
    variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
    rsqrt = paddle.rsqrt(variance + epsilon)
    normalized = x * rsqrt * weight
    return (normalized, rsqrt)
import paddle.incubate.nn.functional as pinf
pinf.fused_rms_norm_ext = fallback_fused_rms_norm_ext

import paddle.nn.functional.flash_attention as fa
def fallback_flashmask_attention(q, k, v, startend_row_indices=None, causal=True):
    q_tr = q.transpose([0, 2, 1, 3]); k_tr = k.transpose([0, 2, 1, 3]); v_tr = v.transpose([0, 2, 1, 3])
    b, h_q, l_q, d = q_tr.shape; _, h_k, l_k, _ = k_tr.shape
    if h_q != h_k:
        n_rep = h_q // h_k
        k_tr = k_tr.reshape([b, h_k, 1, l_k, d]); k_tr = paddle.tile(k_tr, [1, 1, n_rep, 1, 1]); k_tr = k_tr.reshape([b, h_q, l_k, d])
        v_tr = v_tr.reshape([b, h_k, 1, l_k, d]); v_tr = paddle.tile(v_tr, [1, 1, n_rep, 1, 1]); v_tr = v_tr.reshape([b, h_q, l_k, d])
    attn_mask = None; use_causal = False
    if startend_row_indices is not None:
        if startend_row_indices.shape[-1] == 1: startend_row_indices = startend_row_indices.squeeze(-1)
        if startend_row_indices.ndim == 3:
            se = startend_row_indices
            mask = paddle.full([b, 1, l_q, l_k], -1e9, dtype=q.dtype)
            if se.shape[-1] == 2:
                starts, ends = se[..., 0], se[..., 1]
                pos = paddle.arange(l_k, dtype="int32").reshape([1, 1, 1, l_k])
                valid = (pos >= starts.unsqueeze(-1)) & (pos < ends.unsqueeze(-1))
                mask = paddle.where(valid, paddle.zeros_like(mask), mask)
            attn_mask = mask; causal = False
    elif causal and l_q != l_k:
        row_idx = paddle.arange(l_q, dtype="int32").reshape([1, 1, l_q, 1])
        col_idx = paddle.arange(l_k, dtype="int32").reshape([1, 1, 1, l_k])
        causal_bool = col_idx <= (l_k - l_q + row_idx)
        attn_mask = paddle.where(causal_bool, paddle.zeros([1, 1, l_q, l_k], dtype=q.dtype), paddle.full([1, 1, l_q, l_k], -1e9, dtype=q.dtype))
        if b > 1: attn_mask = paddle.tile(attn_mask, [b, 1, 1, 1])
    else: use_causal = causal
    try:
        out_tr = paddle.nn.functional.scaled_dot_product_attention(q_tr, k_tr, v_tr, attn_mask=attn_mask, is_causal=use_causal, training=False)
    except:
        scores = paddle.matmul(q_tr, k_tr.transpose([0, 1, 3, 2])) / (d ** 0.5)
        if attn_mask is not None: scores = scores + attn_mask
        if use_causal:
            grid_q = paddle.arange(l_q, dtype="int32").reshape([l_q, 1])
            grid_k = paddle.arange(l_k, dtype="int32").reshape([1, l_k])
            tril_mask = (grid_k - grid_q) <= (l_k - l_q)
            scores = paddle.where(tril_mask, scores, paddle.to_tensor(-1e9, dtype=scores.dtype))
        attn_weights = paddle.nn.functional.softmax(scores, axis=-1)
        out_tr = paddle.matmul(attn_weights, v_tr)
    return out_tr.transpose([0, 2, 1, 3])
fa.flashmask_attention = fallback_flashmask_attention

import paddle.incubate.tensor.manipulation as m
m.create_async_load = lambda *args, **kwargs: None

import paddle.distributed.fleet.meta_parallel as mp
mp.LocalSharedLayerDesc = mp.SharedLayerDesc
# ===== END PATCHES =====

paddle.set_device("cpu")
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig

local_path = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"

print("Loading model...")
processor = AutoProcessor.from_pretrained(local_path)
model = AutoModelForConditionalGeneration.from_pretrained(
    local_path, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="float32"
)
model.eval()

# Check model structure
print("\n=== Model structure ===")
print("model.visual:", type(model.visual))
print("Model attributes:", [a for a in dir(model) if not a.startswith("_")])
print("Model config:", {k: v for k, v in model.config.to_dict().items() if not k.startswith("_")})

# Check tokenizer
print("\n=== Tokenizer ===")
print("bos_token_id:", processor.tokenizer.bos_token_id)
print("eos_token_id:", processor.tokenizer.eos_token_id)
print("pad_token_id:", processor.tokenizer.pad_token_id)
print("vocab size:", processor.tokenizer.vocab_size)

# Test with text-only input
print("\n=== Text-only test ===")
text_only = processor.tokenizer("Hello world", return_tensors="pd")
print("Text-only IDs:", text_only["input_ids"].tolist())
print("Text-only decoded:", processor.tokenizer.decode(text_only["input_ids"][0].tolist()))

# Test with image
from PIL import Image
img = Image.open("data/test_jpeg/11.jpg").convert("RGB")
print(f"\n=== Image test ({img.size}) ===")

messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "OCR:"}]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd"
)
print("Input keys:", list(inputs.keys()))
for k in inputs:
    v = inputs[k]
    if hasattr(v, "shape"): print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else: print(f"  {k}: {type(v)}")

if "input_ids" in inputs:
    ids = inputs["input_ids"][0].tolist()
    print(f"  input_ids ({len(ids)} tokens): {ids[:30]}...{ids[-10:]}")
    decoded = processor.tokenizer.decode(ids, skip_special_tokens=False)
    print(f"  decoded: [{decoded[:200]}]")

# Generate with debug
gen_config = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=True)
print("\n=== Generation ===")
with paddle.no_grad():
    outputs = model.generate(**inputs, generation_config=gen_config, max_new_tokens=64)
    output_ids = outputs[0].tolist()[0]
    print(f"Output IDs ({len(output_ids)}): {output_ids}")
    print(f"Decoded: [{processor.decode(output_ids, skip_special_tokens=True)}]")
    print(f"Decoded (w/ special): [{processor.decode(output_ids, skip_special_tokens=False)}]")
print("Done")
