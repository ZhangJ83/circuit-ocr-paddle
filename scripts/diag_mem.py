"""GPU memory diagnostic — step by step measurement."""
import os, sys, json
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def gpu_mem():
    """Return GPU memory used in MB."""
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()

print(f"0. After imports: {gpu_mem()}")

# Minimal patches for PaddleOCR-VL
sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu, 'get_one_valid_tokenizer_file'):
    mu.get_one_valid_tokenizer_file = lambda d, e: list(mu._filter_valid_tokenizer_files(d, e))

# ===== Step 1: Paddle init =====
import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])
print(f"1. Paddle init:  {gpu_mem()}")

# ===== Step 2: Model load =====
from paddleformers.transformers import AutoModelForConditionalGeneration
model = AutoModelForConditionalGeneration.from_pretrained(
    "/root/models/official_models/PaddleOCR-VL",
    load_checkpoint_format="safetensors", dtype="bfloat16")
print(f"2. Model loaded: {gpu_mem()}")

# Restore torch/torchvision after paddleformers blocked them
sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms
import transformers.utils.import_utils as tiu
tiu.is_torch_available = lambda: (True, '')
tiu.is_torchvision_available = lambda: (True, '')

# ===== Step 3: Processor =====
from paddleformers.transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("/root/models/official_models/PaddleOCR-VL", trust_remote_code=True)
print(f"3. Processor:    {gpu_mem()}")

# ===== Step 4: Load 1 image =====
import numpy as np
from PIL import Image
with open("/root/circuit_ocr/output/train_clean.jsonl") as f:
    sample = json.loads(f.readline())
ip = sample["images"][0]
if not os.path.exists(ip):
    ip = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(ip)
img = Image.open(ip).convert("RGB")
w, h = img.size; scale = 256 / max(w, h)
if scale < 1: img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
print(f"4. Image loaded: {gpu_mem()}")

# ===== Step 5: Processor encode =====
# First get image patch count to determine placeholder copies
img_inputs = proc.image_processor(images=[np.array(img)], return_tensors="np")
igt = img_inputs["image_grid_thw"][0]
n_patches = int(igt[1]) * int(igt[2])
n_copies = max(1, n_patches // 4)  # <|placeholder|> tokenizes to 4 subword tokens
print(f"   patches={n_patches}, n_copies={n_copies}, igt={igt}")

proc_out = proc(text=[f"{'<|placeholder|>' * n_copies}OCR:"], images=[np.array(img)],
               return_tensors="np", padding=True, max_length=2048, truncation=True)
print(f"5. Proc encode:  {gpu_mem()}")

# Debug: what types are in proc_out?
for k in proc_out:
    v = proc_out[k]
    if hasattr(v, '__class__'): print(f"   {k}: {type(v).__name__}", end='')
    if hasattr(v, 'shape'): print(f" shape={v.shape}", end='')
    if hasattr(v, 'dtype'): print(f" dtype={v.dtype}", end='')
    print()

# ===== Step 6: Build input tensors (handle torch/np/list) =====
import torch as _torch_check
def to_pd(x):
    if isinstance(x, np.ndarray): return paddle.to_tensor(x)
    if isinstance(x, _torch_check.Tensor): return paddle.to_tensor(x.numpy())
    if isinstance(x, (list, tuple)) and len(x) > 0:
        if isinstance(x[0], (np.ndarray, _torch_check.Tensor)):
            return paddle.to_tensor(np.array([v.numpy() if isinstance(v, _torch_check.Tensor) else v for v in x]))
    return x

input_ids = to_pd(proc_out["input_ids"])
pixel_values = to_pd(proc_out.get("pixel_values", np.zeros((0,))))
image_grid_thw = to_pd(proc_out.get("image_grid_thw", np.array([[0,0,0]])))
attention_mask = paddle.ones_like(input_ids)
labels = paddle.full_like(input_ids, -100)
print(f"6. Tensors:      {gpu_mem()}")

# ===== Step 7: Model forward (eval mode) =====
model.eval()
print(f"7a. Before fwd:  {gpu_mem()}")
with paddle.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask,
               pixel_values=pixel_values, image_grid_thw=image_grid_thw,
               labels=labels)
print(f"7b. After fwd:   {gpu_mem()}")
del out
print(f"7c. Del output:  {gpu_mem()}")

# ===== Step 8: Forward + backward (training mode) =====
print(f"\n8a. Before train:{gpu_mem()}")
model.train()
# Use a smaller label to reduce memory
short_labels = paddle.concat([paddle.full([input_ids.shape[1]-5], -100, dtype="int64"),
                              paddle.to_tensor([1,2,3,4,5], dtype="int64")]).unsqueeze(0)
out = model(input_ids=input_ids, attention_mask=attention_mask,
           pixel_values=pixel_values, image_grid_thw=image_grid_thw,
           labels=short_labels)
loss_val = out[0] if isinstance(out, (list, tuple)) else out.loss
print(f"8b. After loss:  {gpu_mem()}")
print(f"    loss={loss_val.item():.4f}")

try:
    loss_val.backward()
    print(f"8c. After bwd:   {gpu_mem()}")
except Exception as e:
    err = str(e)
    if 'OOM' in err.upper() or 'Out of memory' in err:
        print(f"8c. BWD OOM!    {gpu_mem()}")
    else:
        print(f"8c. BWD FAILED: {e}")

print(f"\nFinal: {gpu_mem()}")
