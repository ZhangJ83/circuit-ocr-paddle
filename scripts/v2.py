"""Step 1: Does test_final.py + processor loading work?"""
import paddle
import paddle.nn.functional as F
if not hasattr(F, "swiglu"):
    F.swiglu = lambda x: paddle.chunk(x, 2, -1)[0] * F.silu(paddle.chunk(x, 2, -1)[1])

paddle.set_device("gpu")
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

# Step 1: Model (proven working)
model = AutoModelForConditionalGeneration.from_pretrained(
    "/root/models/official_models/PaddleOCR-VL",
    load_checkpoint_format="safetensors", dtype="float32")
print("1. Model OK")

# Step 2: Processor (does this work?)
import sys
processor = AutoProcessor.from_pretrained(
    "/root/models/official_models/PaddleOCR-VL", trust_remote_code=True)
print("2. Processor OK")
