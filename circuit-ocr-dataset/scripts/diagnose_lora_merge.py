"""Diagnose LoRA merge issues in eval_benchmark_v2.py"""
import os, sys, json

# Early patch
from types import ModuleType
_dummy_fc = ModuleType('dummy_flex_checkpoint')
_dummy_fc.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy_fc)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("HF_HOME", "F:/hf_cache/hub")
os.environ.setdefault("HF_HUB_CACHE", "F:/hf_cache/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()

import paddle
paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
from pathlib import Path

MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CKPT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v10_fixed"

print("=" * 60)
print("DIAGNOSIS: LoRA Merge Key Matching")
print("=" * 60)

# Load base model
print("\n[1] Loading base model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
model.eval()

# Collect base parameter names
base_params = {}
for n, p in model.named_parameters():
    base_params[n] = p

print(f"  Base model has {len(base_params)} parameters")
# Print some sample names
sample_names = [n for n in list(base_params.keys())[:10]]
for sn in sample_names:
    print(f"    {sn}  shape={base_params[sn].shape}")

# Find a q_proj weight to check naming
q_proj_names = [n for n in base_params if 'q_proj' in n and 'weight' in n]
if q_proj_names:
    print(f"\n  Sample q_proj weight names:")
    for qn in q_proj_names[:3]:
        print(f"    {qn}  shape={base_params[qn].shape}")

# Load LoRA checkpoint
print(f"\n[2] Loading LoRA checkpoint from: {CKPT_DIR}")
lora_file = f"{CKPT_DIR}/lora_s600.pdparams"
lora_state = paddle.load(lora_file)

# Print sample LoRA keys
lora_keys = list(lora_state.keys())
print(f"  LoRA checkpoint has {len(lora_keys)} keys")
print(f"  Sample LoRA keys:")
for lk in lora_keys[:5]:
    print(f"    {lk}  shape={lora_state[lk].shape}")

# Find a q_proj lora_A key
q_lora_keys = [k for k in lora_keys if 'q_proj' in k]
if q_lora_keys:
    print(f"\n  Sample q_proj LoRA keys:")
    for qk in q_lora_keys[:3]:
        print(f"    {qk}  shape={lora_state[qk].shape}")

# Now do the matching as in eval_benchmark_v2.py
print(f"\n[3] Matching LoRA keys to base model parameters...")
lora_pairs = {}
for k, v in lora_state.items():
    if k.endswith('.lora_A'):
        base_name = k[:-len('.lora_A')]
        clean_base = base_name[6:] if base_name.startswith('model.') else base_name
        lora_pairs.setdefault(clean_base, {})['A'] = v.numpy()
        lora_pairs[clean_base]['_orig_key'] = k
    elif k.endswith('.lora_B'):
        base_name = k[:-len('.lora_B')]
        clean_base = base_name[6:] if base_name.startswith('model.') else base_name
        lora_pairs.setdefault(clean_base, {})['B'] = v.numpy()

print(f"  Found {len(lora_pairs)} LoRA adapter pairs")

# Check first few pairs
for i, (lora_base, adapters) in enumerate(lora_pairs.items()):
    if i >= 3:
        break
    weight_key = f"{lora_base}.weight"
    found = weight_key in base_params
    print(f"  [{i}] lora_base={lora_base[:80]}")
    print(f"      weight_key={weight_key[:80]}")
    print(f"      in base_params: {found}")
    if not found:
        # Try to find similar keys
        similar = [n for n in base_params if lora_base.split('.')[-1] in n]
        if similar:
            print(f"      similar keys in base: {similar[:3]}")

# Do detailed matching stats
print(f"\n[4] Detailed matching analysis...")
matched = 0
skipped_no_match = 0
skipped_shape = 0
match_samples = []
skip_samples = []

for lora_base, adapters in lora_pairs.items():
    if 'A' not in adapters or 'B' not in adapters:
        skipped_no_match += 1
        continue
    weight_key = f"{lora_base}.weight"
    if weight_key not in base_params:
        skipped_no_match += 1
        if len(skip_samples) < 5:
            skip_samples.append((lora_base, weight_key))
        continue

    lora_A = adapters['A']
    lora_B = adapters['B']
    W = base_params[weight_key].numpy()

    if lora_A.shape[-1] != lora_B.shape[0]:
        skipped_shape += 1
        continue

    delta = lora_A @ lora_B * 2.0  # LORA_SCALE
    shape_ok = False
    if delta.shape == W.shape:
        shape_ok = True
    elif delta.shape[0] == W.shape[1] and delta.shape[1] == W.shape[0]:
        shape_ok = True
    elif delta.shape[0] == W.shape[0] and delta.shape[1] > W.shape[1]:
        shape_ok = True
    elif delta.shape[0] < W.shape[0] and W.shape[0] % delta.shape[0] == 0:
        shape_ok = True
    elif delta.shape[0] == W.shape[0] and delta.shape[1] < W.shape[1] and W.shape[1] % delta.shape[1] == 0:
        shape_ok = True

    if shape_ok:
        matched += 1
        if len(match_samples) < 3:
            match_samples.append((lora_base, delta.shape, W.shape))
    else:
        skipped_shape += 1
        if len(skip_samples) < 5:
            skip_samples.append((lora_base, f"delta={delta.shape} vs W={W.shape}"))

print(f"  MATCHED: {matched}")
print(f"  SKIPPED (no_match): {skipped_no_match}")
print(f"  SKIPPED (shape): {skipped_shape}")
print(f"  TOTAL pairs: {len(lora_pairs)}")

if match_samples:
    print(f"\n  Sample matches:")
    for ms in match_samples:
        print(f"    {ms[0][:60]}  delta={ms[1]} W={ms[2]}")

if skip_samples:
    print(f"\n  Sample skips:")
    for ss in skip_samples:
        print(f"    {ss[0][:60]}  {ss[1][:80]}")

# Now test: inference with LoRAModel wrapper (known good)
print(f"\n[5] Testing inference with LoRAModel wrapper...")
from PIL import Image

# Reload model with LoRA wrapper
model2 = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model2.config._attn_implementation = "flashmask"
model2.visual.config._attn_implementation = "flashmask"

TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
model2 = LoRAModel(model2, lc)

# Load LoRA weights
lora_weights = {}
for k, v in lora_state.items():
    lora_weights[k] = v
# Set LoRA weights on the LoRAModel
model2.set_state_dict(lora_weights)
model2.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Find a test image
test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50-pure.jsonl"
with open(test_path, encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()]

sample = test_data[0]
img_path = f"{DATASET_DIR}/{sample['images'][0].lstrip('./')}"
img = Image.open(img_path).convert("RGB")
w, h = img.size
MAX_DIM = 384
if max(w, h) > MAX_DIM:
    scale = MAX_DIM / max(w, h)
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":sample["messages"][0]["content"].replace("<image>","")}]}]
inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")

input_ids = inp["input_ids"]
attention_mask = inp["attention_mask"]
pixel_values = inp.get("pixel_values")
image_grid_thw = inp.get("image_grid_thw")

# Manual greedy decode
MAX_TOKENS = 60
generated = []
with paddle.no_grad():
    for _ in range(MAX_TOKENS):
        outputs = model2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )
        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs.logits
        next_token_logits = logits[:, -1, :]

        # repetition_penalty
        if generated:
            for tid in set(generated):
                score = next_token_logits[0, tid].item()
                if score < 0:
                    next_token_logits[0, tid] = score * 1.1
                else:
                    next_token_logits[0, tid] = score / 1.1

        next_token = int(paddle.argmax(next_token_logits, axis=-1).numpy()[0])
        if next_token == processor.tokenizer.eos_token_id:
            break
        generated.append(next_token)
        next_tensor = paddle.to_tensor([[next_token]], dtype=input_ids.dtype)
        input_ids = paddle.concat([input_ids, next_tensor], axis=1)
        attention_mask = paddle.concat([attention_mask, paddle.ones([1, 1], dtype=attention_mask.dtype)], axis=1)

resp_wrapper = processor.tokenizer.decode(generated, skip_special_tokens=True)
print(f"  LoRA wrapper output: {repr(resp_wrapper[:120])}")
print(f"  GT: {repr(sample['messages'][1]['content'][:120])}")

img.close()
del model2
paddle.device.cuda.empty_cache()

print(f"\n[6] DONE - Diagnosis complete")
