"""Pure PyTorch: Load safetensors, merge LoRA deltas, save as float32 safetensors.

Safetensors float32 format avoids Paddle 2.6.2 pdparams loading bugs.
"""
import os, sys, shutil
import numpy as np
from pathlib import Path

LOCAL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
DATA_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
LORA_DIR = f"{DATA_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
MERGED_DIR = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL-LoRA-circuit-ocr"
DELTAS_FILE = f"{LORA_DIR}/lora_deltas.npz"

os.makedirs(MERGED_DIR, exist_ok=True)

# === Step 1: Load LoRA deltas ===
print("[1] Loading LoRA deltas from npz...", flush=True)
delta_data = np.load(DELTAS_FILE)
deltas = {k: delta_data[k] for k in delta_data.files}
delta_data.close()
print(f"[1] {len(deltas)} deltas loaded", flush=True)

# === Step 2: Load safetensors via PyTorch ===
print("[2] Loading safetensors...", flush=True)
import torch
from safetensors.torch import load_file
from safetensors.numpy import save_file

st_file = f"{LOCAL_PATH}/model.safetensors"
file_gb = os.path.getsize(st_file) / 1024**3
print(f"[2] Reading {st_file} ({file_gb:.2f} GB)...", flush=True)
tensors = load_file(st_file, device='cpu')
print(f"[2] Loaded {len(tensors)} tensors", flush=True)

# === Step 3: Merge deltas and convert all to float32 ===
print("[3] Merging LoRA deltas + converting to float32...", flush=True)
merged_count = 0
new_tensors = {}
for key, tensor in tensors.items():
    W = tensor.float()  # Convert bf16/fp16 -> float32
    if key in deltas:
        delta = deltas[key]
        if delta.shape == W.shape:
            W = W + torch.from_numpy(delta)
            merged_count += 1
        elif delta.T.shape == W.shape:
            W = W + torch.from_numpy(delta.T)
            merged_count += 1
    new_tensors[key] = W.numpy()  # Store as numpy float32

print(f"[3] Merged {merged_count}/{len(deltas)} weights", flush=True)

# === Step 4: Save as float32 safetensors ===
print("[4] Saving as float32 safetensors...", flush=True)
# Delete old pdparams to save space
old_pdparams = f"{MERGED_DIR}/model_state.pdparams"
if os.path.exists(old_pdparams):
    os.remove(old_pdparams)
    print("[4] Removed old model_state.pdparams", flush=True)

out_path = f"{MERGED_DIR}/model.safetensors"
print(f"[4] Writing {len(new_tensors)} tensors to {out_path}...", flush=True)
save_file(new_tensors, out_path)
out_gb = os.path.getsize(out_path) / 1024**3
print(f"[4] Saved {out_gb:.2f} GB float32 safetensors", flush=True)

# === Step 5: Copy config files ===
print("[5] Copying config files...", flush=True)
for fname in ["config.json", "generation_config.json", "preprocessor_config.json",
              "processor_config.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer.json", "vocab.json", "merges.txt"]:
    src = f"{LOCAL_PATH}/{fname}"
    if os.path.exists(src) and not os.path.exists(f"{MERGED_DIR}/{fname}"):
        shutil.copy2(src, f"{MERGED_DIR}/{fname}")

print(f"[DONE] Merged safetensors model ready at: {MERGED_DIR}", flush=True)
