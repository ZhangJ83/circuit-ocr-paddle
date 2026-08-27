"""Merge LoRA adapters into base PaddleOCR-VL weights for inference.
Avoids LoRAModel wrapper which crashes during generation on Windows Paddle 2.6.2.
Merges on CPU via numpy to avoid GPU OOM.

lora_scale = alpha / r = 16 / 8 = 2.0
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PATH"] = r"E:\080000software\080900_Miniconda\miniconda3\Library\bin;" + os.environ.get("PATH", "")
import numpy as np
import paddle
from pathlib import Path

LORA_DIR = "g:/mimo_project/circuit_ocr/circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr"
SRC = f"{LORA_DIR}/lora_weights_f32.pdparams"
DST = f"{LORA_DIR}/merged_model_f32.pdparams"

LORA_SCALE = 2.0  # alpha / r = 16 / 8

print(f"Loading merged weights from: {SRC}")
state = paddle.load(SRC)
print(f"Total keys: {len(state)}")

# Convert all to numpy for CPU-side matmul
base_np = {}
lora_np = {}
for k, v in state.items():
    arr = v.numpy()
    if 'lora' in k.lower():
        lora_np[k] = arr
    else:
        base_np[k] = arr
del state  # Free GPU memory
print(f"Base keys: {len(base_np)}, LoRA keys: {len(lora_np)}")

# Group LoRA keys by base parameter name
lora_pairs = {}
for k in lora_np:
    if k.endswith('.lora_A'):
        base_name = k[:-len('.lora_A')]
        lora_pairs.setdefault(base_name, {})['A'] = lora_np[k]
    elif k.endswith('.lora_B'):
        base_name = k[:-len('.lora_B')]
        lora_pairs.setdefault(base_name, {})['B'] = lora_np[k]

print(f"LoRA target modules: {len(lora_pairs)}")

# Merge LoRA into base weights on CPU
merged_count = 0
skipped_count = 0
for lora_base, adapters in lora_pairs.items():
    if 'A' not in adapters or 'B' not in adapters:
        continue

    lora_A = adapters['A']  # numpy [hidden, r]
    lora_B = adapters['B']  # numpy [r, hidden]

    base_weight_key = f"{lora_base}.weight"
    if base_weight_key not in base_np:
        skipped_count += 1
        continue

    W = base_np[base_weight_key]
    delta = lora_A @ lora_B  # [hidden, r] @ [r, hidden] = [hidden, hidden]
    if delta.shape != W.shape:
        skipped_count += 1
        continue

    base_np[base_weight_key] = (W + delta * LORA_SCALE).astype(np.float32)
    merged_count += 1

print(f"Merged: {merged_count}, Skipped: {skipped_count}")

# Convert back to paddle tensors
print("Converting back to paddle tensors...")
merged_state = {}
for k, v in base_np.items():
    merged_state[k] = paddle.to_tensor(v)

# Save
print(f"Saving to {DST}...")
paddle.save(merged_state, DST)
sz_gb = Path(DST).stat().st_size / 1e9
print(f"Saved: {sz_gb:.2f} GB")
print("Done!")
