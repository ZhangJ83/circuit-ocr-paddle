"""Absolute minimal test: model load + forward."""
import os, json, numpy as np
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"

# Load model (same as test_final.py which WORKS)
print("1. load model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, load_checkpoint_format="safetensors", dtype="float32")
print(f"   OK, params={sum(p.numel() for p in model.parameters()):,}")

# Load processor
print("2. load processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("   OK")

# Load 1 sample
print("3. load sample...")
with open("/root/circuit_ocr/output/train_clean.jsonl") as f:
    sample = json.loads(f.readline())
from PIL import Image
img_path = sample["images"][0]
if not os.path.exists(img_path):
    img_path = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(img_path)
img = Image.open(img_path).convert("RGB")
w, h = img.size
scale = 384 / max(w, h)
if scale < 1:
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
print(f"   OK: {img.size}")

# Process
print("4. processor call...")
msgs = [json.dumps(sample["messages"], ensure_ascii=False)]
inputs = processor(text=msgs, images=[np.array(img)], return_tensors="pd",
                  padding=True, max_length=2048, truncation=True)
print(f"   OK: input_ids={inputs['input_ids'].shape}")

# Forward (no LoRA, just base model)
print("5. forward...")
model.eval()
try:
    with paddle.no_grad():
        outputs = model(**inputs)
    print(f"   OK! loss={outputs.loss.item() if hasattr(outputs,'loss') else 'N/A'}")
    print("MINIMAL TEST PASSED!")
except Exception as e:
    print(f"   FAIL: {e}")
    import traceback; traceback.print_exc()
