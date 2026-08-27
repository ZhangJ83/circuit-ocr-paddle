"""v4: dummy -> paddleformers -> check_imports patch -> processor"""
import sys, os
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
from importlib.machinery import ModuleSpec

def mkpkg(name):
    m = type(sys)(name); m.__version__ = "0.0.0"; m.__path__ = []
    m.__spec__ = ModuleSpec(name, None, is_package=True)
    return m
class IM: pass

for name in ["torchvision", "torchvision.transforms", "torchvision.transforms.functional",
             "torchvision.transforms.v2", "torchvision.transforms.v2.functional"]:
    m = mkpkg(name)
    if name == "torchvision.transforms": m.InterpolationMode = IM
    sys.modules[name] = m

# mistral compat (before paddleformers)
from mistral_common.tokens.tokenizers import utils as _mu
if not hasattr(_mu, 'get_one_valid_tokenizer_file'):
    _mu.get_one_valid_tokenizer_file = lambda d, e: list(_mu._filter_valid_tokenizer_files(d, e))

import paddle; paddle.set_device("gpu")
from paddleformers.transformers import AutoProcessor

# AFTER paddleformers, re-ensure dummies + patch check_imports
for name in ["torchvision", "torchvision.transforms", "torchvision.transforms.functional",
             "torchvision.transforms.v2", "torchvision.transforms.v2.functional"]:
    if sys.modules.get(name) is None:
        m = mkpkg(name)
        if name == "torchvision.transforms": m.InterpolationMode = IM
        sys.modules[name] = m

import transformers.dynamic_module_utils as dmu
_orig = dmu.check_imports
def safe_check(fn, *a, **kw):
    result = _orig(fn, *a, **kw)
    # Restore dummies after each call
    for name in ["torchvision", "torchvision.transforms", "torchvision.transforms.functional",
                 "torchvision.transforms.v2", "torchvision.transforms.v2.functional"]:
        if sys.modules.get(name) is None:
            m = mkpkg(name)
            if name == "torchvision.transforms": m.InterpolationMode = IM
            sys.modules[name] = m
    return result
dmu.check_imports = safe_check

# tensor patches (pt->np)
import transformers.feature_extraction_utils as tfeu
ogf = tfeu.BatchFeature._get_is_as_tensor_fns
tfeu.BatchFeature._get_is_as_tensor_fns = lambda s, t=None: ogf(s, 'np' if t in ('pt','pd') else t)
import transformers.tokenization_utils_base as ttub
octt = ttub.BatchEncoding.convert_to_tensors
ttub.BatchEncoding.convert_to_tensors = lambda s, t=None, pba=False: octt(s, 'np' if t in ('pt','pd') else t, pba)

proc = AutoProcessor.from_pretrained(
    "/root/models/official_models/PaddleOCR-VL", trust_remote_code=True)
print("1. Processor OK")

# Test apply_chat_template
import json, numpy as np
from PIL import Image
with open("/root/circuit_ocr/output/train_clean.jsonl") as f: sample = json.loads(f.readline())
ip = sample["images"][0]
if not os.path.exists(ip): ip = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(ip)
img = Image.open(ip).convert("RGB"); w, h = img.size; s = 384 / max(w, h)
if s < 1: img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)

# Try string content format
msg1 = [{"role": "user", "content": "<image>\nOCR:"}]
try:
    inp = proc.apply_chat_template(msg1, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    print(f"2a. String format OK. input_ids={inp['input_ids'].shape}")
except Exception as e:
    print(f"2a. String format FAIL: {e}")

# Try plain text + images
try:
    inp2 = proc(text=["<image>\nOCR:"], images=[img], return_tensors="pd", padding=True)
    print(f"2b. text+images OK. input_ids={inp2['input_ids'].shape}")
except Exception as e:
    print(f"2b. text+images FAIL: {e}")
