"""Debug PaddleOCR-VL chat template."""
import os, sys, json
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from importlib.machinery import ModuleSpec
def mk(n): m=type(sys)(n); m.__version__="0.0.0"; m.__path__=[]; m.__spec__=ModuleSpec(n,None,is_package=True); return m
class IM: pass
tv=mk("torchvision"); tvt=mk("torchvision.transforms"); tvt.InterpolationMode=IM
tvtf=mk("torchvision.transforms.functional"); tvtv2=mk("torchvision.transforms.v2"); tvtv2f=mk("torchvision.transforms.v2.functional")
for m,d in [("torchvision",tv),("torchvision.transforms",tvt),("torchvision.transforms.functional",tvtf),("torchvision.transforms.v2",tvtv2),("torchvision.transforms.v2.functional",tvtv2f)]:
    sys.modules[m]=d

import paddle; paddle.set_device("gpu")
from paddleformers.transformers import AutoProcessor
from PIL import Image

# Apply runtime patches BEFORE loading processor
import transformers.dynamic_module_utils as dmu
_o = dmu.check_imports
def sc(fn,*a,**kw):
    try:
        if 'torchvision' in open(fn).read(): return []
    except: pass
    return _o(fn,*a,**kw)
dmu.check_imports = sc

import transformers.feature_extraction_utils as tfeu
ogf = tfeu.BatchFeature._get_is_as_tensor_fns
tfeu.BatchFeature._get_is_as_tensor_fns = lambda s,t=None: ogf(s, 'np' if t in ('pt','pd') else t)

import transformers.tokenization_utils_base as ttub
octt = ttub.BatchEncoding.convert_to_tensors
ttub.BatchEncoding.convert_to_tensors = lambda self,t=None,pba=False: octt(self, 'np' if t in ('pt','pd') else t, pba)

MODEL_PATH = "/root/models/official_models/PaddleOCR-VL"

proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Processor loaded")

# Load one image
with open("/root/circuit_ocr/output/train_clean.jsonl") as f:
    sample = json.loads(f.readline())
img_path = sample["images"][0]
if not os.path.exists(img_path):
    img_path = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(img_path)
img = Image.open(img_path).convert("RGB")
w, h = img.size; scale = 384 / max(w, h)
if scale < 1: img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

# Test 1: PIL image in content
msg1 = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "OCR:"}]}]
try:
    inp = proc.apply_chat_template(msg1, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    ids = inp["input_ids"]
    print(f"Test1 OK: shape={ids.shape}, tokens={ids[0][:20].numpy().tolist()}")
except Exception as e:
    print(f"Test1 FAIL: {e}")

# Test 2: V15 JSON + images
try:
    inp2 = proc(text=[json.dumps(sample["messages"], ensure_ascii=False)], images=[img], return_tensors="pd", padding=True, max_length=2048, truncation=True)
    ids2 = inp2["input_ids"]
    print(f"Test2 OK: shape={ids2.shape}")
except Exception as e:
    print(f"Test2 FAIL: {e}")
