"""Compare named_parameters, state_dict, and LoRA key names."""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "F:/hf_cache/hub"
os.environ["PADDLE_HOME"] = "F:/paddle_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import paddle
from paddleformers.transformers import AutoModelForConditionalGeneration

LOCAL = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"

paddle.set_device("gpu")
model = AutoModelForConditionalGeneration.from_pretrained(
    LOCAL, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="float32"
)

# Sample some names
np_names = [n for n, _ in model.named_parameters()]
sd_names = list(model.state_dict().keys())

print("=== named_parameters (first 10) ===")
for n in np_names[:10]:
    print(f"  {n}")
print()
print("=== state_dict keys (first 10) ===")
for k in sd_names[:10]:
    print(f"  {k}")
print()

# Load LoRA
lora = paddle.load(r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset\PaddleOCR-VL-LoRA-circuit-ocr\lora_weights_f32.pdparams")
lora_bases = set()
for k in lora:
    if k.endswith('.lora_A'):
        bn = k[:-len('.lora_A')]
        clean = bn[6:] if bn.startswith('model.') else bn
        lora_bases.add(clean)

print(f"=== LoRA base names (first 10 of {len(lora_bases)}) ===")
for n in sorted(lora_bases)[:10]:
    print(f"  {n}")
print()

# Check overlap
np_set = set(np_names)
sd_set = set(sd_names)
print(f"named_parameters count: {len(np_set)}")
print(f"state_dict count: {len(sd_set)}")
print(f"LoRA bases count: {len(lora_bases)}")
print(f"LoRA ∩ named_parameters: {len(lora_bases & np_set)}")
print(f"LoRA ∩ state_dict: {len(lora_bases & sd_set)}")

# Show a few LoRA bases that DON'T match
missing = lora_bases - sd_set
print(f"\nLoRA bases NOT in state_dict (first 10 of {len(missing)}):")
for n in sorted(missing)[:10]:
    print(f"  {n}")

# Check if it's a prefix issue
if missing:
    # Try: do sd_names contain these as substrings?
    sample = sorted(missing)[0]
    matches = [s for s in sd_names if sample in s or s in sample]
    print(f"\n  Sample missing: {sample}")
    print(f"  Partial matches in sd: {matches[:5]}")

# Also check: do state_dict keys have .weight suffix?
print(f"\n=== state_dict keys with .weight (first 10) ===")
w_keys = [k for k in sd_names if k.endswith('.weight')]
for k in w_keys[:10]:
    print(f"  {k}")
