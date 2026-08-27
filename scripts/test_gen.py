"""Minimal test: does model.generate() work?"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])
_orig_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _patched_sdpa
import torch
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"
PROJECT_DIR = "/root/circuit_ocr"

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

print("Loading model...")
proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForConditionalGeneration.from_pretrained(MODEL_PATH, load_checkpoint_format="safetensors", dtype="bfloat16")
model.config._attn_implementation = "sdpa"
model.visual.config._attn_implementation = "sdpa"
model.eval()
print("Model loaded!")

# Load 1 val sample
val_data = [json.loads(l) for l in open(os.path.join(PROJECT_DIR, "output/val_clean.jsonl"))]
vs = val_data[0]
print(f"GT: {vs['messages'][1]['content'][:80]}")

# Prepare input
vip = vs["images"][0]
if not os.path.exists(vip): vip = vip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
vimg = Image.open(vip).convert("RGB")
vw, vh = vimg.size
if max(vw, vh) > 384:
    vscale = 384 / max(vw, vh)
    vimg = vimg.resize((int(vw*vscale), int(vh*vscale)), Image.LANCZOS)
vimg_np = np.array(vimg)

# Get patch count
feats = proc.image_processor(images=[vimg_np], return_tensors="np")
g = feats["image_grid_thw"][0]
vn = max(1, int(g[1]) * int(g[2]) // 4)
print(f"Patches: {int(g[1])*int(g[2])}, n_copies={vn}")

# Encode input
inp = proc(text=[f"{'<|placeholder|>'*vn}OCR:"], images=[vimg_np],
            return_tensors="np", padding=False, max_length=1024, truncation=True)
inp_pd = to_pd(inp)
print(f"Input shape: {inp_pd['input_ids'].shape}")

# Generate
gc = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=True)
print("Generating...")
t0 = time.time()
with paddle.no_grad():
    out = model.generate(**inp_pd, generation_config=gc, max_new_tokens=256)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

output_ids = out[0].tolist()[0]
pred = proc.tokenizer.decode(output_ids, skip_special_tokens=True)
print(f"PRED: {pred[:120]}")
print("TEST PASSED!")
