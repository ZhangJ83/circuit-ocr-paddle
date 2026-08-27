"""Test eval_benchmark.apply_paddle_patches on cloud."""
import sys, os
sys.path.insert(0, "circuit-ocr-dataset/scripts")
from eval_benchmark import apply_paddle_patches
apply_paddle_patches()

import paddle; paddle.set_device("gpu")
from paddleformers.transformers import AutoProcessor
p = AutoProcessor.from_pretrained("/root/models/official_models/PaddleOCR-VL", trust_remote_code=True)
print("PROCESSOR OK")
