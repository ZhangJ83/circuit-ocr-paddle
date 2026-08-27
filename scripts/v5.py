"""No patches needed after pip install torch torchvision."""
import os, json
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# mistral compat
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu, 'get_one_valid_tokenizer_file'):
    mu.get_one_valid_tokenizer_file = lambda d, e: list(mu._filter_valid_tokenizer_files(d, e))

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
import numpy as np
from PIL import Image

MODEL = "/root/models/official_models/PaddleOCR-VL"

# 1. Unblock torch/torchvision (paddleformers sets them to None + patches is_*_available)
import sys
for m in ['torchvision']:
    if m in sys.modules and sys.modules[m] is None:
        del sys.modules[m]
import torchvision, torchvision.transforms
# torch is installed but not yet in sys.modules; import it now
import torch
# Restore the availability checks that paddleformers disabled
import transformers.utils.import_utils as tiu
tiu.is_torch_available = lambda: (True, '')
tiu.is_torchvision_available = lambda: (True, '')

# 2. Processor
print("2. Processor...")
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
print("   OK")

# 2. Model
print("2. Model...")
model = AutoModelForConditionalGeneration.from_pretrained(MODEL, load_checkpoint_format="safetensors", dtype="float32")
print(f"   OK, params={sum(p.numel() for p in model.parameters()):,}")

# 3. LoRA
print("3. LoRA...")
for n, p in model.named_parameters():
    if "mlp_AR" in n or "projector" in n: p.stop_gradient = True
lc = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=[".*q_proj",".*k_proj",".*v_proj",".*o_proj",".*linear_1",".*linear_2"])
model = LoRAModel(model, lc)
print(f"   OK, trainable={sum(p.numel() for p in model.parameters() if not p.stop_gradient):,}")

# 4. Load 1 sample
print("4. Sample...")
with open("/root/circuit_ocr/output/train_clean.jsonl") as f:
    sample = json.loads(f.readline())
ip = sample["images"][0]
if not os.path.exists(ip): ip = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(ip)
img = Image.open(ip).convert("RGB")
w, h = img.size; scale = 384 / max(w, h)
if scale < 1: img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
print("   OK")

# 5. Process + Forward
print("5. Forward...")
msg = json.dumps(sample["messages"], ensure_ascii=False)
inp = proc(text=[msg], images=[np.array(img)], return_tensors="pd", padding=True, max_length=2048, truncation=True)
outputs = model(**inp)
print(f"   loss={outputs.loss.item():.4f}")
print("ALL OK!")
