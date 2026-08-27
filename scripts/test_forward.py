"""Test model forward pass."""
import json, numpy as np, sys, os
from PIL import Image
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x,2,-1)[0] * F.silu(paddle.chunk(x,2,-1)[1])

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"

# Patch check_imports
import transformers.dynamic_module_utils as _dmu
_orig_check = _dmu.check_imports
def _patched_check(filename, *a, **kw):
    try: return _orig_check(filename, *a, **kw)
    except ModuleNotFoundError as e:
        if 'torchvision' in str(e): return []
        raise
_dmu.check_imports = _patched_check

# Load processor and model
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForConditionalGeneration.from_pretrained(MODEL_PATH, load_checkpoint_format="safetensors", dtype="float32")

# Freeze projector
for n, p in model.named_parameters():
    if "mlp_AR" in n or "projector" in n:
        p.stop_gradient = True

# LoRA
lc = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=[".*q_proj",".*k_proj",".*v_proj",".*o_proj",".*linear_1",".*linear_2"])
model = LoRAModel(model, lc)
print(f"Model ready, trainable: {sum(p.numel() for p in model.parameters() if not p.stop_gradient):,}")

# Load sample
with open("/root/circuit_ocr/output/train_clean.jsonl") as f:
    sample = json.loads(f.readline())

img_path = sample["images"][0]
if not os.path.exists(img_path):
    img_path = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(img_path)

img = Image.open(img_path).convert("RGB")
w, h = img.size
scale = 384 / max(w, h)
if scale < 1:
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

# Test forward pass
messages_batch = [json.dumps(sample["messages"], ensure_ascii=False)]
print(f"Message preview: {messages_batch[0][:100]}...")

try:
    inputs = processor(text=messages_batch, images=[np.array(img)], return_tensors="pd",
                      padding=True, max_length=2048, truncation=True)
    print(f"Processor OK. input_ids shape: {inputs['input_ids'].shape}")

    print("Running forward pass...")
    outputs = model(**inputs)
    print(f"Forward OK. loss={outputs.loss.item():.4f}")
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
