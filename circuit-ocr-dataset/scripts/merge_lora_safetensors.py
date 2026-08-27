"""Merge LoRA into safetensors file, then load as fresh model.

Strategy: Instead of modifying model in-memory (which crashes Paddle 2.6.2),
we modify the safetensors file on disk, then load the model fresh.
This avoids the set_value/set_state_dict segfault.
"""
import os, sys, json, shutil
import numpy as np
from pathlib import Path

# === Config ===
LOCAL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
DATA_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
LORA_DIR = f"{DATA_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
MERGED_DIR = f"{LORA_DIR}/merged_safetensors"
LORA_SCALE = 2.0

os.makedirs(MERGED_DIR, exist_ok=True)

# === Step 1: Load LoRA weights and build delta map ===
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

# Build delta map: safetensors_key -> numpy delta
# The safetensors file uses the same keys as model.state_dict()
deltas = {}
for lora_base, adapters in lora_pairs.items():
    if 'A' not in adapters or 'B' not in adapters: continue
    la, lb = adapters['A'], adapters['B']
    if la.shape[-1] != lb.shape[0]: continue
    delta = la @ lb * LORA_SCALE
    wk = f"{lora_base}.weight"
    deltas[wk] = delta

print(f"[1] {len(deltas)} delta tensors computed", flush=True)

# === Step 2: Load original safetensors, apply deltas, save new ===
print("[2] Processing safetensors...", flush=True)
from safetensors.numpy import save_file

# Use safe_open for lazy loading to handle large files
from safetensors.numpy import safe_open
from safetensors import safe_open as safe_open_any

# We need to handle the .safetensors file(s). PaddleOCR-VL has a single model.safetensors
src_dir = Path(LOCAL_PATH)
st_files = sorted(src_dir.glob("*.safetensors"))
print(f"[2] Found safetensors: {[f.name for f in st_files]}", flush=True)

for st_file in st_files:
    print(f"[2] Processing {st_file.name} ({os.path.getsize(st_file) / 1024**3:.2f} GB)...", flush=True)

    # Read all keys and their tensors
    tensors = {}
    try:
        with safe_open(str(st_file), framework="np") as f:
            keys = list(f.keys())
            print(f"[2]   {len(keys)} keys, loading and merging...", flush=True)
            merged_count = 0
            for key in keys:
                W = f.get_tensor(key)
                if key in deltas:
                    delta = deltas[key]
                    if delta.shape == W.shape:
                        W = (W + delta).astype(np.float32)
                        merged_count += 1
                tensors[key] = W
            print(f"[2]   {merged_count} weights merged with LoRA", flush=True)
    except Exception as e:
        print(f"[2] safe_open failed: {e}, trying load_file...", flush=True)
        from safetensors.numpy import load_file
        tensors = load_file(str(st_file))
        merged_count = 0
        for key in list(tensors.keys()):
            if key in deltas:
                W = tensors[key]
                delta = deltas[key]
                if delta.shape == W.shape:
                    tensors[key] = (W + delta).astype(np.float32)
                    merged_count += 1
        print(f"[2]   {merged_count} weights merged with LoRA", flush=True)

    out_path = f"{MERGED_DIR}/{st_file.name}"
    print(f"[2] Saving to {out_path}...", flush=True)
    save_file(tensors, out_path)
    print(f"[2] Saved OK ({os.path.getsize(out_path) / 1024**3:.2f} GB)", flush=True)

# === Step 3: Copy config files ===
print("[3] Copying config files...", flush=True)
for fname in ["config.json", "generation_config.json", "preprocessor_config.json",
              "processor_config.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer.json", "vocab.json", "merges.txt"]:
    src = f"{LOCAL_PATH}/{fname}"
    if os.path.exists(src):
        shutil.copy2(src, f"{MERGED_DIR}/{fname}")

print(f"[3] Merged model ready at: {MERGED_DIR}", flush=True)
print("DONE!", flush=True)
