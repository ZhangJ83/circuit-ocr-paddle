"""Pure PyTorch: Load safetensors, merge LoRA deltas from npz, save as Paddle pdparams.

Runs in a separate process — no Paddle loaded — to avoid cuDNN conflicts.
"""
import os, sys, shutil
import numpy as np
from pathlib import Path

# Suppress torch/PyTorch warnings to avoid confusion
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

st_file = f"{LOCAL_PATH}/model.safetensors"
file_gb = os.path.getsize(st_file) / 1024**3
print(f"[2] Reading {st_file} ({file_gb:.2f} GB)...", flush=True)
tensors = load_file(st_file, device='cpu')
print(f"[2] Loaded {len(tensors)} tensors", flush=True)

# === Step 3: Merge deltas ===
print("[3] Merging LoRA deltas...", flush=True)
merged_count = 0
skipped = 0
for key in list(tensors.keys()):
    tensor = tensors[key]
    if key in deltas:
        # Convert to float32 for merge
        W = tensor.float().numpy()
        delta = deltas[key]
        if delta.shape == W.shape:
            tensors[key] = torch.from_numpy((W + delta).astype(np.float32))
            merged_count += 1
        elif delta.T.shape == W.shape:
            # LoRA delta is transposed relative to weight
            tensors[key] = torch.from_numpy((W + delta.T).astype(np.float32))
            merged_count += 1
        else:
            skipped += 1
            if skipped <= 3:
                print(f"[3] SKIP shape: {key} W={W.shape} delta={delta.shape}", flush=True)
    else:
        # Convert all to float32 for consistency with Paddle model
        if tensor.dtype != torch.float32:
            tensors[key] = tensor.float()

print(f"[3] Merged {merged_count}/{len(deltas)} (skipped {skipped})", flush=True)

# === Step 4: Save as Paddle pdparams ===
print("[4] Saving as Paddle pdparams (converting all to float32)...", flush=True)
import paddle
pd_sd = {}
bf16_count = 0
for key, tensor in tensors.items():
    if tensor.dtype == torch.bfloat16:
        bf16_count += 1
    pd_sd[key] = tensor.float().numpy()

if bf16_count:
    print(f"[4] Converted {bf16_count} bf16 tensors to float32", flush=True)

out_path = f"{MERGED_DIR}/model_state.pdparams"
print(f"[4] Writing {len(pd_sd)} tensors to {out_path}...", flush=True)
paddle.save(pd_sd, out_path)
print(f"[4] Saved ({os.path.getsize(out_path)/1024**3:.2f} GB)", flush=True)

# === Step 5: Copy config files ===
print("[5] Copying config files...", flush=True)
for fname in ["config.json", "generation_config.json", "preprocessor_config.json",
              "processor_config.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer.json", "vocab.json", "merges.txt"]:
    src = f"{LOCAL_PATH}/{fname}"
    if os.path.exists(src):
        shutil.copy2(src, f"{MERGED_DIR}/{fname}")

print(f"[DONE] Merged model ready at: {MERGED_DIR}", flush=True)
