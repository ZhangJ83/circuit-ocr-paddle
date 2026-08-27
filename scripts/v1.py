"""Minimal test: does apply_chat_template work with PIL images?"""
import os, sys, json
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
from importlib.machinery import ModuleSpec
def mk(n): m=type(sys)(n); m.__version__="0.0.0"; m.__path__=[]; m.__spec__=ModuleSpec(n,None,is_package=True); return m
class IM: pass
for m,d in [("torchvision",mk("torchvision")),("torchvision.transforms",mk("torchvision.transforms")),
            ("torchvision.transforms.functional",mk("torchvision.transforms.functional")),
            ("torchvision.transforms.v2",mk("torchvision.transforms.v2")),
            ("torchvision.transforms.v2.functional",mk("torchvision.transforms.v2.functional"))]:
    sys.modules[m]=d
sys.modules["torchvision.transforms"].InterpolationMode=IM

import paddle; paddle.set_device("gpu")
from paddleformers.transformers import AutoProcessor
from PIL import Image

# Patches for processor loading
import transformers.dynamic_module_utils as dmu
_o=dmu.check_imports
dmu.check_imports=lambda fn,*a,**kw:(lambda r:([sys.modules.__setitem__(m,d) for m,d in [
    ("torchvision",sys.modules["torchvision"]),("torchvision.transforms",sys.modules["torchvision.transforms"]),
    ("torchvision.transforms.functional",sys.modules["torchvision.transforms.functional"]),
    ("torchvision.transforms.v2",sys.modules["torchvision.transforms.v2"]),
    ("torchvision.transforms.v2.functional",sys.modules["torchvision.transforms.v2.functional"])] if sys.modules.get(m) is None],r)[-1])(_o(fn,*a,**kw))

import transformers.feature_extraction_utils as tfeu
ogf=tfeu.BatchFeature._get_is_as_tensor_fns
tfeu.BatchFeature._get_is_as_tensor_fns=lambda s,t=None:ogf(s,'np' if t in ('pt','pd') else t)
import transformers.tokenization_utils_base as ttub
octt=ttub.BatchEncoding.convert_to_tensors
ttub.BatchEncoding.convert_to_tensors=lambda s,t=None,pba=False:octt(s,'np' if t in ('pt','pd') else t,pba)

MODEL_PATH="/root/models/official_models/PaddleOCR-VL"
proc=AutoProcessor.from_pretrained(MODEL_PATH,trust_remote_code=True)
print("1. Processor OK")

with open("/root/circuit_ocr/output/train_clean.jsonl") as f: sample=json.loads(f.readline())
ip=sample["images"][0]
if not os.path.exists(ip): ip="/root/circuit_ocr/output/review_1000/images/"+os.path.basename(ip)
img=Image.open(ip).convert("RGB"); w,h=img.size; s=384/max(w,h)
if s<1: img=img.resize((int(w*s),int(h*s)),Image.LANCZOS)

# Test: apply_chat_template with PIL image
msg=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"OCR:"}]}]
try:
    inp=proc.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pd")
    print(f"2. Chat template OK. input_ids shape={inp['input_ids'].shape}")
except Exception as e:
    print(f"2. Chat template FAIL: {e}")
