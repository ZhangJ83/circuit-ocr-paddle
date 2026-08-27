"""Minimal test with ALL patches, just model load + forward."""
import os, sys, json, numpy as np
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- COPY ALL PATCHES FROM train.py ---
import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

from importlib.machinery import ModuleSpec
def _make_dummy_pkg(name, is_package=True):
    m = type(sys)(name); m.__version__ = '0.0.0'; m.__path__ = []; m.__spec__ = ModuleSpec(name, None, is_package=is_package)
    return m
_tv = _make_dummy_pkg('torchvision', True)
_tvt = _make_dummy_pkg('torchvision.transforms', True)
_tvtf = _make_dummy_pkg('torchvision.transforms.functional', True)
sys.modules['torchvision'] = _tv
sys.modules['torchvision.transforms'] = _tvt
sys.modules['torchvision.transforms.functional'] = _tvtf

import transformers.dynamic_module_utils as _dmu
_orig = _dmu.check_imports
def _patched(fn, *a, **kw):
    try: return _orig(fn, *a, **kw)
    except ModuleNotFoundError as e:
        if 'torchvision' in str(e): return []
        raise
_dmu.check_imports = _patched

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from PIL import Image

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"

print("1. load processor...")
proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("   OK")

print("2. load model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, load_checkpoint_format="safetensors", dtype="float32")
print(f"   OK, params={sum(p.numel() for p in model.parameters()):,}")

print("3. load sample...")
with open("/root/circuit_ocr/output/train_clean.jsonl") as f:
    sample = json.loads(f.readline())
img_path = sample["images"][0]
if not os.path.exists(img_path):
    img_path = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(img_path)
img = Image.open(img_path).convert("RGB")
w, h = img.size; scale = 384 / max(w, h)
if scale < 1: img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
print(f"   OK")

print("4. processor call...")
msgs = [json.dumps(sample["messages"], ensure_ascii=False)]
inputs = proc(text=msgs, images=[np.array(img)], return_tensors="pd",
             padding=True, max_length=2048, truncation=True)
print(f"   OK, input_ids={inputs['input_ids'].shape}")

print("5. forward (base model, no LoRA)...")
model.eval()
try:
    with paddle.no_grad():
        outputs = model(**inputs)
    print(f"   type={type(outputs).__name__}")
    if hasattr(outputs, 'loss'):
        print(f"   loss={outputs.loss.item():.4f}")
    print("SUCCESS!")
except Exception as e:
    print(f"   FAIL: {e}")
    import traceback; traceback.print_exc()
