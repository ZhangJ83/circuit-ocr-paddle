"""Use PyTorch to load safetensors, merge LoRA, save as Paddle-compatible pdparams.

Paddle 2.6.2 on Windows crashes when model params are modified after loading.
Workaround: merge LoRA into weights BEFORE loading, using PyTorch for file I/O.
"""
import os, sys, json, shutil
import numpy as np
from pathlib import Path

LOCAL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
DATA_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
LORA_DIR = f"{DATA_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
MERGED_DIR = f"{LORA_DIR}/merged_pt"
LORA_SCALE = 2.0

os.makedirs(MERGED_DIR, exist_ok=True)

# === Step 1: Load LoRA weights (Paddle format, but numpy-compatible) ===
print("[1] Loading LoRA weights...", flush=True)
import paddle
lora_state = paddle.load(f"{LORA_DIR}/lora_weights_f32.pdparams")

lora_pairs = {}
for k, v in lora_state.items():
    if k.endswith('.lora_A'):
        bn = k[:-len('.lora_A')]; clean = bn[6:] if bn.startswith('model.') else bn
        lora_pairs.setdefault(clean, {})['A'] = v.numpy()
    elif k.endswith('.lora_B'):
        bn = k[:-len('.lora_B')]; clean = bn[6:] if bn.startswith('model.') else bn
        lora_pairs.setdefault(clean, {})['B'] = v.numpy()

print(f"[1] {len(lora_pairs)} LoRA pairs", flush=True)

# Build delta map: weight_key -> numpy delta
deltas = {}
for lora_base, adapters in lora_pairs.items():
    if 'A' not in adapters or 'B' not in adapters: continue
    la, lb = adapters['A'], adapters['B']
    if la.shape[-1] != lb.shape[0]: continue
    delta = la @ lb * LORA_SCALE
    wk = f"{lora_base}.weight"
    deltas[wk] = delta.astype(np.float32)
print(f"[1] {len(deltas)} deltas computed", flush=True)
del lora_state, lora_pairs
paddle.device.cuda.empty_cache()

# === Step 2: Use PyTorch to read bf16 safetensors ===
print("[2] Loading safetensors via PyTorch...", flush=True)
import torch
from safetensors.torch import load_file

st_file = f"{LOCAL_PATH}/model.safetensors"
print(f"[2] Loading {st_file} ({os.path.getsize(st_file)/1024**3:.2f} GB)...", flush=True)
tensors = load_file(st_file, device='cpu')
print(f"[2] Loaded {len(tensors)} tensors", flush=True)

# === Step 3: Merge deltas ===
print("[3] Merging LoRA deltas...", flush=True)
merged = 0
for key in list(tensors.keys()):
    if key in deltas:
        W = tensors[key].float().numpy()  # Convert bf16/fp32 to float32 numpy
        delta = deltas[key]
        if delta.shape == W.shape:
            merged_val = W + delta
            tensors[key] = torch.from_numpy(merged_val)
            merged += 1
        else:
            print(f"[3] SKIP shape: {key} W={W.shape} delta={delta.shape}", flush=True)
    else:
        # Convert all tensors to float32 for consistency
        if tensors[key].dtype != torch.float32:
            tensors[key] = tensors[key].float()

print(f"[3] Merged {merged}/{len(deltas)} weights", flush=True)

# === Step 4: Save as Paddle pdparams ===
print("[4] Converting to Paddle pdparams...", flush=True)
pd_sd = {}
for key, tensor in tensors.items():
    pd_sd[key] = tensor.numpy()

out_path = f"{MERGED_DIR}/model_state.pdparams"
print(f"[4] Saving {len(pd_sd)} tensors to {out_path}...", flush=True)
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

print(f"[5] Merged model ready at: {MERGED_DIR}", flush=True)
print("DONE!", flush=True)
