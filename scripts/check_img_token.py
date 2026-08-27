"""Check image token properties."""
import os, sys
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

sys.modules.pop('torchvision', None)
import torchvision, torchvision.transforms, torch
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu, 'get_one_valid_tokenizer_file'):
    mu.get_one_valid_tokenizer_file = lambda d, e: list(mu._filter_valid_tokenizer_files(d, e))

import paddle; paddle.set_device("gpu")
from paddleformers.transformers import AutoProcessor
from PIL import Image

proc = AutoProcessor.from_pretrained("/root/models/official_models/PaddleOCR-VL", trust_remote_code=True)

# Check what image_token is
print(f"image_token attr: {repr(getattr(proc, 'image_token', 'NOT_FOUND'))}")

# Check how the placeholder replacement works
test_text = "<|placeholder|>OCR:"
result = test_text.replace("<|placeholder|>", getattr(proc, 'image_token', '<image>'))
print(f"After replacement: {repr(result)}")

# Tokenize it
toks = proc.tokenizer.encode(result)
print(f"Number of tokens for '<|placeholder|>': {len(proc.tokenizer.encode('<|placeholder|>'))}")
print(f"Total tokens for image portion: {len(toks)}")
print(f"Tokens: {toks}")

# Test with single image
import numpy as np
img = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
img_feats = proc.image_processor(images=[np.array(img)], return_tensors="np")
igt = img_feats["image_grid_thw"][0]
n_patches = int(igt[1]) * int(igt[2])
print(f"\nImage patches for 384x384: {n_patches}")
print(f"image_grid_thw: {igt}")
