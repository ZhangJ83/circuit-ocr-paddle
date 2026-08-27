"""Trace the exact line where PyTorch conversion fails."""
import json, numpy as np, sys, os
from PIL import Image

os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x,2,-1)[0] * F.silu(paddle.chunk(x,2,-1)[1])

# torchvision dummies with __path__
_tv = type(sys)('torchvision'); _tv.__version__ = '0.0.0'; _tv.__path__ = []
_tvt = type(sys)('torchvision.transforms'); _tvt.__path__ = []
_tvtf = type(sys)('torchvision.transforms.functional'); _tvtf.__path__ = []
sys.modules['torchvision'] = _tv
sys.modules['torchvision.transforms'] = _tvt
sys.modules['torchvision.transforms.functional'] = _tvtf

# mistral
try:
    from mistral_common.tokens.tokenizers import utils as _mu
    if not hasattr(_mu, 'get_one_valid_tokenizer_file'):
        _mu.get_one_valid_tokenizer_file = lambda d,e: list(_mu._filter_valid_tokenizer_files(d,e))
except: pass

# check_imports patch
import transformers.dynamic_module_utils as _dmu
_orig = _dmu.check_imports
def _patched(fn, *a, **kw):
    try: return _orig(fn, *a, **kw)
    except ModuleNotFoundError as e:
        if 'torchvision' in str(e): return []
        raise
_dmu.check_imports = _patched

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"

# --- Step 1: Load processor ---
print("1. Loading processor...")
proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("   OK")

# --- Step 2: Load model ---
print("2. Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, load_checkpoint_format="safetensors", dtype="float32")
print(f"   OK. Params: {sum(p.numel() for p in model.parameters()):,}")

# Freeze
for n, p in model.named_parameters():
    if "mlp_AR" in n or "projector" in n:
        p.stop_gradient = True

# LoRA
lc = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=[".*q_proj",".*k_proj",".*v_proj",".*o_proj",".*linear_1",".*linear_2"])
model = LoRAModel(model, lc)
print(f"   LoRA applied, trainable: {sum(p.numel() for p in model.parameters() if not p.stop_gradient):,}")

# --- Step 3: Load sample ---
print("3. Loading sample...")
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
print(f"   OK")

# --- Step 4: Processor call ---
print("4. Calling processor...")
msgs = [json.dumps(sample["messages"], ensure_ascii=False)]
try:
    inputs = proc(text=msgs, images=[np.array(img)], return_tensors="pd",
                 padding=True, max_length=2048, truncation=True)
    print(f"   OK. input_ids shape: {inputs['input_ids'].shape}")
except Exception as e:
    print(f"   FAIL at processor: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# --- Step 5: Model forward ---
print("5. Running model forward...")
model.train()
try:
    outputs = model(**inputs)
    print(f"   OK. type={type(outputs)}")
    if hasattr(outputs, 'loss'):
        print(f"   loss={outputs.loss.item():.4f}")
    else:
        print(f"   No loss attribute, keys: {dir(outputs)[:10]}")
    print("SUCCESS!")
except Exception as e:
    print(f"   FAIL at forward: {e}")
    import traceback; traceback.print_exc()
