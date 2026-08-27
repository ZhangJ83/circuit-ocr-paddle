"""Quick inference — what does the model actually predict?"""
import os, sys, json
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
for n, p in model.named_parameters():
    if "mlp_AR" in n or "projector" in n: p.stop_gradient = True
lc = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=[".*q_proj",".*k_proj",".*v_proj",".*o_proj",".*linear_1",".*linear_2"])
model = LoRAModel(model, lc)
model.model.full = lambda *a, **kw: iter(model.model.named_parameters())

# Load checkpoint
sd = paddle.load(os.path.join(PROJECT_DIR, "checkpoints/baseline/best.pdparams"))
loaded = 0
for n, p in model.named_parameters():
    if n in sd:
        try:
            p.set_value(paddle.cast(sd[n], p.dtype))
            loaded += 1
        except: pass
print(f"Loaded {loaded} params from checkpoint")
model.eval()

# Test samples
for src_file in ["output/val_clean.jsonl", "output/train_clean.jsonl"]:
    fpath = os.path.join(PROJECT_DIR, src_file)
    for j, line in enumerate(open(fpath)):
        if j >= 2: break
        vs = json.loads(line)
        vip = vs["images"][0]
        if not os.path.exists(vip): vip = vip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
        vimg = Image.open(vip).convert("RGB")
        vw, vh = vimg.size
        if max(vw, vh) > 512:
            vscale = 512 / max(vw, vh)
            vimg = vimg.resize((int(vw*vscale), int(vh*vscale)), Image.LANCZOS)
        vimg_np = np.array(vimg)
        vimg_feats = proc.image_processor(images=[vimg_np], return_tensors="np")
        vig = vimg_feats["image_grid_thw"][0]
        vn = max(1, int(vig[1]) * int(vig[2]) // 4)
        vinp = proc(text=[f"{'<|placeholder|>'*vn}OCR:"], images=[vimg_np],
                    return_tensors="np", padding=False, max_length=2048, truncation=True)
        vinp_pd = to_pd(vinp)
        gen = []; eos = 2
        with paddle.no_grad():
            for _ in range(256):
                vo = model(**vinp_pd)
                vlogits = vo[0] if isinstance(vo, (list, tuple)) else vo.logits
                nt = int(paddle.argmax(vlogits[:, -1, :], axis=-1).numpy()[0])
                if nt == eos: break
                gen.append(nt)
                vinp_pd["input_ids"] = paddle.concat([vinp_pd["input_ids"], paddle.to_tensor([[nt]])], axis=1)
                vinp_pd["attention_mask"] = paddle.concat([vinp_pd["attention_mask"], paddle.ones([1,1], dtype="int64")], axis=1)
        pred = proc.tokenizer.decode(gen, skip_special_tokens=True)
        gt = vs["messages"][1]["content"]
        print(f"[{src_file} #{j}]")
        print(f"  GT: {gt[:120]}")
        print(f"  PRED: {pred[:120]}")
        print()
