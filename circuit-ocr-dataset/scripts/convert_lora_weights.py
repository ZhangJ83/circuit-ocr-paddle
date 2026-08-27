"""Convert LoRA weights from uint16/float8 to float32 that Paddle 2.6.2 can load."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import paddle

SRC = "g:/mimo_project/circuit_ocr/circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/final_model_light.pdparams"
DST = "g:/mimo_project/circuit_ocr/circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/lora_weights_f32.pdparams"

def bf16_to_f32(arr):
    """Convert uint16 bf16 array to float32 via numpy view trick."""
    return (arr.astype(np.uint32) << 16).view(np.float32)

print(f"Loading {SRC}...")
state = paddle.load(SRC, return_numpy=True)
print(f"Loaded {len(state)} keys")

converted = {}
for k, v in state.items():
    if str(v.dtype) == 'uint16':
        # bf16 stored as uint16
        if v.ndim == 0:
            converted[k] = bf16_to_f32(np.array([v.item()], dtype=np.uint16))[0]
        else:
            converted[k] = bf16_to_f32(v)
    elif str(v.dtype) == 'float8_e4m3fn':
        # fp8 → upcast to float32 directly
        converted[k] = v.astype(np.float32)
    else:
        converted[k] = v.astype(np.float32)

print(f"Converted {len(converted)} keys to float32")
print(f"Sample keys:")
for k in list(converted.keys())[:3]:
    print(f"  {k}: dtype={converted[k].dtype} shape={converted[k].shape}")

# Also save light version (only LoRA params)
lora_only = {k: v for k, v in converted.items() if 'lora' in k.lower()}
print(f"LoRA-only params: {len(lora_only)}")
print(f"Saving full to {DST}...")
paddle.save(converted, DST)
print("Done!")
