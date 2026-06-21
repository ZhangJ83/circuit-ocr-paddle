#!/usr/bin/env python3
"""End-to-end validation: imports, model loading, single-sample loss."""
import paddle, os, sys, json
from pathlib import Path
from PIL import Image

DD = "/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset"
MP = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"

os.chdir(DD)
sys.path.insert(0, "scripts")
from eval_benchmark import apply_paddle_patches
apply_paddle_patches()

print(f"Paddle: {paddle.__version__}")
print(f"GPU: {paddle.device.cuda.get_device_name(0)}")

# Imports
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
print("Imports OK")

# Processor
proc = AutoProcessor.from_pretrained(MP)
print(f"Processor OK, vocab: {proc.tokenizer.vocab_size}")

# Model
print("Loading model (bf16)...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MP, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

# LoRA
lc = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj'])
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
total = sum(p.size for p in model.parameters())
tr = sum(p.size for p in model.parameters() if not p.stop_gradient)
print(f"Model OK: Total={total:,} Trainable={tr:,} ({100*tr/total:.2f}%)")

# Single sample loss test
print("\n=== Single Sample Loss Test ===")
train_data = [json.loads(l) for l in open(f"{DD}/ocr_vl_sft-train.jsonl") if l.strip()]
s = train_data[0]
query = s["messages"][0]["content"]
label = s["messages"][1]["content"]
img_path = s["images"][0]
if not img_path.startswith("/"):
    img_path = f"{DD}/{img_path.lstrip('./')}"

img = Image.open(img_path).convert("RGB")
w, h = img.size
max_dim = 168
if max(w, h) > max_dim:
    scale = max_dim / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

# Build full input
msgs = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": query.replace("<image>", "")}
]}]
full_msgs = msgs + [{"role": "assistant", "content": [{"type": "text", "text": label}]}]

full_in = proc.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False,
                                   return_dict=True, return_tensors="pd")
prompt_in = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors="pd")
plen = prompt_in["input_ids"].shape[1]
fids = full_in["input_ids"]

# Labels: -100 for prompt, real IDs for response
labels = paddle.full_like(fids, -100)
labels[0, plen:] = fids[0, plen:]

# Forward
model.eval()
with paddle.no_grad():
    out = model(**full_in)
logits = out[0] if isinstance(out, (tuple, list)) else out

# Manual CE loss — paddle.cast preserves grad (unlike .astype)
sl = paddle.cast(logits[:, :-1, :], "float32")  # logits → fp32 for CE kernel
slb = labels[:, 1:]  # keep int64 for CE (labels must be int)
mask = paddle.cast((slb != -100), "float32")
slb_c = paddle.where(slb != -100, slb, paddle.zeros_like(slb))  # int64 zeros
ce = paddle.nn.functional.cross_entropy(
    sl.reshape([-1, sl.shape[-1]]),
    slb_c.reshape([-1]),
    reduction="none"
).reshape(slb.shape)
loss = (ce * mask).sum() / mask.sum().clip(min=1)
loss_val = loss.item()

print(f"  Image: {w}x{h} -> {img.size}")
print(f"  Prompt tokens: {plen}, Full: {fids.shape[1]}, Label: {fids.shape[1]-plen}")
print(f"  LOSS: {loss_val:.6f}")
print(f"  Label: {label[:80]}...")

if loss_val < 0.01:
    print("  FAIL: Loss near zero — model is not learning!")
elif loss_val < 0.5:
    print(f"  WARNING: Loss low ({loss_val:.4f}) — may converge too fast")
else:
    print(f"  PASS: Loss is healthy ({loss_val:.4f})")

# ============================================================
# Backward + optimizer step verification
# ============================================================
print("\n=== Backward + Optimizer Step Test ===")

# Set up optimizer
model.train()
opt = paddle.optimizer.AdamW(
    learning_rate=5e-4,
    parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1, beta1=0.9, beta2=0.95, epsilon=1e-8
)

# Store initial LoRA weights (use numpy after model.train to avoid graph issues)
initial_weights = {}
for name, param in model.named_parameters():
    if not param.stop_gradient:
        initial_weights[name] = param.numpy().copy()

# Recompute loss fresh with model.train()
full_in = proc.apply_chat_template(
    full_msgs, tokenize=True, add_generation_prompt=False,
    return_dict=True, return_tensors="pd"
)
prompt_in = proc.apply_chat_template(
    msgs, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pd"
)
plen = prompt_in["input_ids"].shape[1]
fids = full_in["input_ids"]
labels_t = paddle.full_like(fids, -100)
labels_t[0, plen:] = fids[0, plen:]

out = model(**full_in)
logits = out[0] if isinstance(out, (tuple, list)) else out
sl = paddle.cast(logits[:, :-1, :], "float32")
slb = labels_t[:, 1:]
mask = paddle.cast((slb != -100), "float32")
slb_c = paddle.where(slb != -100, slb, paddle.zeros_like(slb))
ce = paddle.nn.functional.cross_entropy(
    sl.reshape([-1, sl.shape[-1]]),
    slb_c.reshape([-1]),
    reduction="none"
).reshape(slb.shape)
loss = (ce * mask).sum() / mask.sum().clip(min=1)
print(f"  Loss: {loss.item():.4f}")

# Backward
loss.backward()
print("  Backward: OK")

# Check gradients
nonzero_grads = 0
zero_grads = 0
for name, param in model.named_parameters():
    if not param.stop_gradient:
        if param.grad is not None and (param.grad.numpy() != 0).any():
            nonzero_grads += 1
        else:
            zero_grads += 1
print(f"  Gradients: {nonzero_grads} non-zero, {zero_grads} zero")

# Optimizer step
opt.step()
opt.clear_grad()

# Verify weights changed
changed = 0
for name, param in model.named_parameters():
    if not param.stop_gradient:
        if (param.numpy() != initial_weights[name]).any():
            changed += 1
print(f"  Weights changed: {changed}/{nonzero_grads+zero_grads}")

# Final verdict
all_ok = nonzero_grads > 0 and changed > 0
if all_ok:
    print("\n=== FULL CHAIN VERIFIED: forward + backward + update all work ===")
else:
    print(f"\n=== FAIL: nonzero_grads={nonzero_grads} changed={changed} ===")

print("\n=== ALL CHECKS COMPLETE ===")
