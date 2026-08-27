"""Quick test of processor call pattern."""
import json, numpy as np, sys, os
from PIL import Image

# Apply patches
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import paddle; paddle.set_device("gpu")
from paddleformers.transformers import AutoProcessor

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Load one sample
with open("/root/circuit_ocr/output/train_clean.jsonl") as f:
    sample = json.loads(f.readline())

# Process image
img_path = sample["images"][0]
print(f"Image path from JSONL: {img_path}")
if not os.path.exists(img_path):
    img_path = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(img_path)
print(f"Resolved: {img_path}, exists={os.path.exists(img_path)}")

img = Image.open(img_path).convert("RGB")
w, h = img.size
scale = 384 / max(w, h)
if scale < 1:
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

# V15-Clean approach
messages_batch = [json.dumps(sample["messages"], ensure_ascii=False)]
print("Calling processor...")
try:
    inputs = processor(text=messages_batch, images=[np.array(img)], return_tensors="pd",
                      padding=True, max_length=2048, truncation=True)
    print(f"SUCCESS: input_ids shape={inputs['input_ids'].shape}")
    for k in inputs.keys():
        v = inputs[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}")
        else:
            print(f"  {k}: type={type(v)}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
