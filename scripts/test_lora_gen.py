"""Test: does LoRA + model.generate() work?"""
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
from paddleformers.peft import LoRAConfig, LoRAModel
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

print("Loading model with LoRA...")
proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForConditionalGeneration.from_pretrained(MODEL_PATH, load_checkpoint_format="safetensors", dtype="bfloat16")
model.config._attn_implementation = "sdpa"
model.visual.config._attn_implementation = "sdpa"

for n, p in model.named_parameters():
    if "mlp_AR" in n or "projector" in n: p.stop_gradient = True

lc = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=[".*q_proj",".*k_proj",".*v_proj",".*o_proj",".*linear_1",".*linear_2"])
model = LoRAModel(model, lc)
model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
model.eval()
print("LoRA model ready!")

# Load 1 val sample
vs = json.loads(open(os.path.join(PROJECT_DIR, "output/val_clean.jsonl")).readline())
vip = vs["images"][0]
if not os.path.exists(vip): vip = vip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
vimg = Image.open(vip).convert("RGB")
vw, vh = vimg.size
if max(vw, vh) > 384:
    vscale = 384 / max(vw, vh)
    vimg = vimg.resize((int(vw*vscale), int(vh*vscale)), Image.LANCZOS)

feats = proc.image_processor(images=[np.array(vimg)], return_tensors="np")
g = feats["image_grid_thw"][0]
vn = max(1, int(g[1]) * int(g[2]) // 4)

inp = proc(text=[f"{'<|placeholder|>'*vn}OCR:"], images=[np.array(vimg)],
            return_tensors="np", padding=False, max_length=1024, truncation=True)
inp_pd = to_pd(inp)
print(f"Input shape: {inp_pd['input_ids'].shape}")

gc = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=True)
print("Generating with LoRA model...")
t0 = time.time()
with paddle.no_grad():
    out = model.generate(**inp_pd, generation_config=gc, max_new_tokens=64)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s!")
print(f"Output tokens: {len(out[0].tolist()[0])}")
print("LORA+GENERATE WORKS!")
