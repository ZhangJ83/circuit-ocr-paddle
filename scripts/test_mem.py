"""Minimal memory test: 1 forward + backward on 1 sample."""
import os, sys, json, time
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms, torch
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu, 'get_one_valid_tokenizer_file'):
    mu.get_one_valid_tokenizer_file = lambda d, e: list(mu._filter_valid_tokenizer_files(d, e))

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

import numpy as np
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load processor + model
print("Loading...")
proc = AutoProcessor.from_pretrained("/root/models/official_models/PaddleOCR-VL", trust_remote_code=True)
model = AutoModelForConditionalGeneration.from_pretrained("/root/models/official_models/PaddleOCR-VL", load_checkpoint_format="safetensors", dtype="bfloat16")

for n, p in model.named_parameters():
    if "mlp_AR" in n or "projector" in n: p.stop_gradient = True
lc = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=[".*q_proj",".*k_proj",".*v_proj",".*o_proj",".*linear_1",".*linear_2"])
model = LoRAModel(model, lc)

# Load 1 sample
with open("/root/circuit_ocr/output/train_clean.jsonl") as f:
    sample = json.loads(f.readline())
ip = sample["images"][0]
if not os.path.exists(ip): ip = "/root/circuit_ocr/output/review_1000/images/" + os.path.basename(ip)

img = Image.open(ip).convert("RGB")
w, h = img.size; scale = 256 / max(w, h)
img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

img_np = np.array(img)
img_inputs = proc.image_processor(images=[img_np], return_tensors="np")
igt = img_inputs["image_grid_thw"][0]
n_patches = int(igt[1]) * int(igt[2])
n_copies = max(1, n_patches // 4)

label = sample["messages"][1]["content"]
label_ids = proc.tokenizer.encode(label) + [proc.tokenizer.eos_token_id or 2]
label_tensor = paddle.to_tensor(label_ids, dtype="int64")

inp = proc(text=[f"{'<|placeholder|>'*n_copies}OCR:"], images=[img_np],
           return_tensors="np", padding=True, max_length=2048, truncation=True)

prompt_len = inp["input_ids"].shape[1]
input_ids = paddle.to_tensor(inp["input_ids"])
full_ids = paddle.concat([input_ids[0], label_tensor])
full_labels = paddle.concat([paddle.full([prompt_len], -100, dtype="int64"), label_tensor])

print(f"input_ids: {full_ids.shape}, patches={n_patches}, n_copies={n_copies}")
print(f"prompt_len={prompt_len}, label_len={len(label_ids)}, total={full_ids.shape[0]}")

# Check GPU memory
si,so,se = paddle.device.cuda.memory_stats()
print(f"Before forward: GPU allocated")

# Forward + backward
model.train()
out = model(input_ids=full_ids.unsqueeze(0),
           attention_mask=paddle.ones([1, full_ids.shape[0]], dtype="int64"),
           labels=full_labels.unsqueeze(0),
           pixel_values=paddle.to_tensor(img_inputs["pixel_values"]),
           image_grid_thw=paddle.to_tensor(img_inputs["image_grid_thw"]))
loss_val = out[0] if isinstance(out, (list, tuple)) else out.loss
print(f"Loss: {loss_val.item():.4f}")

loss_val.backward()
print("Backward OK")

# Check memory
print("SUCCESS")
