"""Convert ALL trained weights (not just LoRA) from bf16/fp8 to float32 safetensors.

Key insight: LoRA adapters are all zero, but non-LoRA weights (mlp_AR, lm_head, etc.)
have meaningful trained values. We need the FULL model state dict.
"""
import os, sys, shutil
import numpy as np
from pathlib import Path

LOCAL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
LORA_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset\PaddleOCR-VL-LoRA-circuit-ocr"
MERGED_DIR = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL-LoRA-circuit-ocr"

os.makedirs(MERGED_DIR, exist_ok=True)

def bf16_to_f32(arr):
    """Convert uint16 bf16 array to float32."""
    return (arr.astype(np.uint32) << 16).view(np.float32)

print("[1] Loading trained checkpoint (bf16/fp8)...", flush=True)
import paddle
state = paddle.load(f"{LORA_DIR}/final_model_light.pdparams", return_numpy=True)
print(f"[1] Loaded {len(state)} keys", flush=True)

# Convert all to float32
print("[2] Converting all weights to float32...", flush=True)
converted = {}
bf16_count = 0
fp8_count = 0
f32_count = 0
for k, v in state.items():
    if str(v.dtype) == 'uint16':
        if v.ndim == 0:
            converted[k] = bf16_to_f32(np.array([v.item()], dtype=np.uint16))[0]
        else:
            converted[k] = bf16_to_f32(v)
        bf16_count += 1
    elif str(v.dtype) == 'float8_e4m3fn':
        converted[k] = v.astype(np.float32)
        fp8_count += 1
    else:
        converted[k] = v.astype(np.float32)
        f32_count += 1

print(f"[2] Converted: {bf16_count} bf16, {fp8_count} fp8, {f32_count} f32", flush=True)

# Count nonzero keys
nonzero = sum(1 for v in converted.values() if np.count_nonzero(v) > 0)
print(f"[2] Keys with nonzero values: {nonzero}/{len(converted)}", flush=True)

# === Now merge with base model weights ===
# The trained checkpoint has keys with "model." prefix
# The base safetensors has the same keys
print("[3] Loading base safetensors via subprocess...", flush=True)
import subprocess

# Save trained weights as npz first
print("[3a] Saving trained weights as npz...", flush=True)
trained_npz_path = f"{MERGED_DIR}/trained_weights.npz"
np.savez_compressed(trained_npz_path, **converted)
print(f"[3a] Saved {len(converted)} keys to npz", flush=True)

# Run merge in subprocess (no Paddle loaded = no DLL conflict)
print("[3b] Running merge in subprocess...", flush=True)
script_path = f"{MERGED_DIR}/_merge_script.py"
result = subprocess.run(
    [r"E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe", script_path],
    capture_output=True, text=True, timeout=600
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])
print(f"[3b] Subprocess exit: {result.returncode}", flush=True)
if result.returncode != 0:
    print("[ERROR] Subprocess failed!", flush=True)
    sys.exit(1)

# === Copy config files ===
print("[4] Copying config files...", flush=True)
for fname in ["config.json", "generation_config.json", "preprocessor_config.json",
              "processor_config.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer.json", "vocab.json", "merges.txt"]:
    src = f"{LOCAL_PATH}/{fname}"
    dst = f"{MERGED_DIR}/{fname}"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

print(f"[DONE] Trained model ready at: {MERGED_DIR}", flush=True)
