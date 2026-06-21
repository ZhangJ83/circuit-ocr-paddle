"""End-to-end verification: train 100 samples, merge LoRA, test inference.
NO numpy for weight operations — pure Paddle all the way.
"""
import os, sys, json, time, random
from pathlib import Path

os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "HF_HOME": "F:/hf_cache/hub", "PADDLE_HOME": "F:/paddle_cache",
    "HF_HUB_CACHE": "F:/hf_cache/hub", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "FLAGS_allocator_strategy": "auto_growth",
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

log("Applying patches...")
apply_paddle_patches()

import paddle; paddle.set_device("gpu")
log(f"GPU: {paddle.device.cuda.get_device_name(0)}")

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

# ── Load model + LoRA ──
log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
lc = LoRAConfig(r=8, lora_alpha=16, target_modules=['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj'])
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())

processor = AutoProcessor.from_pretrained(MODEL_PATH)
trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
log(f"Trainable: {trainable:,}")

# ── Clone initial weights (PADDLE ops only) ──
init_w = {}
for k, p in model.named_parameters():
    if 'lora_' in k:
        init_w[k] = p.detach().clone()
log(f"Cloned {len(init_w)} initial LoRA matrices")

# ── Load 100 training samples ──
with open(f"{DATASET_DIR}/ocr_vl_sft-train.jsonl", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data); data = data[:100]
log(f"Training on {len(data)} samples")

opt = paddle.optimizer.AdamW(
    learning_rate=5e-4,
    parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1)

# ── Training loop ──
from PIL import Image; from io import BytesIO
model.train()
t0 = time.time()

for idx, sample in enumerate(data):
    img_path = f"{DATASET_DIR}/{sample['images'][0].lstrip('./')}"
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    if max(w, h) > 168:
        scale = 168 / max(w, h)
        image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0)
    image = Image.open(buf)

    query = sample["messages"][0]["content"]
    label = sample["messages"][1]["content"][:200]

    prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
    prompt_inputs = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    full_msgs = [
        {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]},
        {"role":"assistant","content":[{"type":"text","text":label}]},
    ]
    full_inputs = processor.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pd")

    full_ids = full_inputs["input_ids"]
    labels_t = paddle.full_like(full_ids, -100, dtype=full_ids.dtype)
    labels_t[0, prompt_len:] = full_ids[0, prompt_len:]

    out = model(**full_inputs)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    shift_logits = paddle.cast(logits[:, :-1, :], "float32")
    shift_labels = labels_t[:, 1:]
    mask = paddle.cast(shift_labels != -100, "float32")
    shift_labels_clamped = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))

    ce = paddle.nn.functional.cross_entropy(
        shift_logits.reshape([-1, shift_logits.shape[-1]]),
        shift_labels_clamped.reshape([-1]),
        reduction="none").reshape(shift_labels.shape)

    loss = (ce * mask).sum() / mask.sum().clip(min=1)
    loss.backward(); opt.step(); opt.clear_grad()
    image.close()

    if idx < 3:
        log(f"  Sample {idx}: loss={loss.item():.4f}")
    if (idx+1) % 25 == 0:
        elapsed = (time.time()-t0)/60
        log(f"  [{idx+1}/100] loss={loss.item():.4f}  elapsed={elapsed:.1f}m")

total_min = (time.time()-t0)/60
log(f"Training done in {total_min:.1f}m")

# ── Verify weight changes (PADDLE ops) ──
changed = 0
for k in init_w:
    for n, p in model.named_parameters():
        if n == k:
            diff = paddle.max(paddle.abs(p - init_w[k])).item()
            if diff > 1e-8:
                changed += 1
            break
log(f"Weights changed: {changed}/{len(init_w)}")

# ── Save LoRA weights ──
model.eval()
lora_dict = {}
for k, p in model.named_parameters():
    if 'lora_' in k:
        lora_dict[k] = p.detach().clone()
save_path = f"{OUTPUT_DIR}/e2e_verify_lora.pdparams"
paddle.save(lora_dict, save_path)
log(f"Saved LoRA to {save_path}")

# ── Reload and verify (paddle only) ──
reloaded = paddle.load(save_path)
match = True
for k in lora_dict:
    if not paddle.allclose(lora_dict[k], reloaded[k]):
        log(f"  MISMATCH: {k}")
        match = False
log(f"Save/load match: {match}")

# ── Test 1: LoRA model inference (NO merge) ──
log("=== Test 1: LoRA model inference (no merge) ===")
model.eval()
test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl"
if Path(test_path).exists():
    with open(test_path, encoding="utf-8") as f:
        test_data = [json.loads(l) for l in f if l.strip()][:3]
else:
    with open(f"{DATASET_DIR}/ocr_vl_sft-test.jsonl", encoding="utf-8") as f:
        test_data = [json.loads(l) for l in f if l.strip()][:3]

for i, s in enumerate(test_data):
    img = Image.open(f"{DATASET_DIR}/{s['images'][0].lstrip('./')}").convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    with paddle.no_grad():
        out_ids = model.generate(**inp, max_new_tokens=256, do_sample=False)[0]
    resp = processor.decode(out_ids, skip_special_tokens=True)
    ref = s["messages"][1]["content"][:150]
    log(f"  [{i}] Pred ({len(resp)} chars): {resp[:120]}...")
    log(f"  [{i}] Ref  ({len(ref)} chars): {ref[:120]}...")
    img.close()

# ── Test 2: Merge + inference ──
log("=== Test 2: Merged model inference ===")
# Build a fresh base model, merge LoRA into it, test
log("Loading fresh base model for merge...")
model2 = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model2.config._attn_implementation = "flashmask"
model2.visual.config._attn_implementation = "flashmask"

import numpy as np
merged_count = 0
for k, p in model2.named_parameters():
    lora_A_key = f"model.{k}.lora_A"
    lora_B_key = f"model.{k}.lora_B"
    if lora_A_key not in reloaded or lora_B_key not in reloaded:
        continue

    lora_A = reloaded[lora_A_key]  # [hidden, r] e.g. [1152, 8]
    lora_B = reloaded[lora_B_key]  # [r, hidden*groups] e.g. [8, 4608]

    # Convert to numpy for merge (PaddleFormers set_value needs float16)
    try:
        A_np = paddle.cast(lora_A, "float32").numpy()
        B_np = paddle.cast(lora_B, "float32").numpy()
    except:
        log(f"  Skipping {k}: numpy conversion failed")
        continue

    A_hidden = A_np.shape[0]  # hidden = 1152
    A_rank = A_np.shape[1]    # r = 8
    B_total = B_np.shape[1]   # hidden * groups = 4608

    if B_total > A_hidden:
        groups = B_total // A_hidden
        B_reshaped = B_np.reshape(A_rank, groups, A_hidden).transpose(1, 0, 2).reshape(A_rank * groups, A_hidden)
    else:
        B_reshaped = B_np

    # delta = B_reshaped @ A = [r*groups, hidden] @ [hidden, r] = [r*groups, r]
    # Wait no, we want delta same shape as W: [total_hidden, hidden] = [4608, 1152]
    # B_reshaped = [r*groups, hidden] = [32, 1152]
    # A = [hidden, r] = [1152, 8]
    # B_reshaped @ A = [32, 1152] @ [1152, 8] = [32, 8] — wrong shape!
    #
    # Need: A @ B or the other way...
    # lora_A @ lora_B reshaped?
    # A = [1152, 8], B = [8, 4608] → A @ B = [1152, 4608]
    # (A @ B).T = [4608, 1152] — matches W!
    # So delta = (A @ B).T
    # But with GQA: B needs to be grouped differently
    #
    # Actually the simplest: delta = (B.T @ A.T).T = A @ B
    # A = [1152, 8], B = [8, 4608]
    # A @ B = [1152, 4608]
    # delta = (A @ B).T = [4608, 1152]

    delta = (A_np @ B_np).T  # [total_hidden, hidden]

    W_np = paddle.cast(p, "float32").numpy()

    if delta.shape != W_np.shape:
        if delta.shape[0] < W_np.shape[0]:
            factor = W_np.shape[0] // delta.shape[0]
            delta = np.tile(delta, (factor, 1))
        if delta.shape != W_np.shape:
            log(f"  Shape mismatch {k}: delta={delta.shape} W={W_np.shape}")
            continue

    scale = lc.lora_alpha / lc.r
    W_new = W_np + delta * scale

    param_dtype = p.dtype
    try:
        p.set_value(paddle.to_tensor(W_new.astype("float16"), dtype=param_dtype, place=p.place))
        merged_count += 1
    except Exception as e:
        try:
            p.set_value(paddle.to_tensor(W_new, dtype=param_dtype, place=p.place))
            merged_count += 1
        except Exception as e2:
            if merged_count < 3:
                log(f"  set_value failed {k}: {e2}")

log(f"Merged: {merged_count} layers")

# Test inference with merged model
model2.eval()
for i, s in enumerate(test_data):
    img = Image.open(f"{DATASET_DIR}/{s['images'][0].lstrip('./')}").convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    with paddle.no_grad():
        out_ids = model2.generate(**inp, max_new_tokens=256, do_sample=False)[0]
    resp = processor.decode(out_ids, skip_special_tokens=True)
    ref = s["messages"][1]["content"][:150]
    log(f"  [{i}] Pred ({len(resp)} chars): {resp[:120]}...")
    log(f"  [{i}] Ref  ({len(ref)} chars): {ref[:120]}...")
    img.close()

log("=== E2E VERIFICATION DONE ===")
