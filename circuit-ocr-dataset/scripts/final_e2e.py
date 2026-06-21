"""FINAL end-to-end verification: train→save→load→merge→inference.
With fixed fp8 patches (no alias), everything should work.
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

# Step 1: Apply FIXED patches
from eval_benchmark import apply_paddle_patches
apply_paddle_patches()

import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
LORA_PATH = f"{OUTPUT_DIR}/lora_final_fp16.pdparams"

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    # Handle Unicode in Windows GBK console
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', errors='replace').decode('ascii'), flush=True)

# ── Load model ──
log("[1/6] Loading model + LoRA...")
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

# Verify numpy works now
for k, p in model.named_parameters():
    if 'lora_A' in k:
        v = p.numpy()
        log(f"  LoRA numpy: dtype={v.dtype} nan={np.isnan(v).any()} min={v.min():.4f} max={v.max():.4f}")
        break

# ── Clone init weights ──
init_w = {}
for k, p in model.named_parameters():
    if 'lora_' in k:
        init_w[k] = p.numpy().copy()  # NOW WORKS!
log(f"  Cloned {len(init_w)} init weights (numpy works!)")

# ── Load data ──
with open(f"{DATASET_DIR}/ocr_vl_sft-train.jsonl", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data); data = data[:100]

opt = paddle.optimizer.AdamW(
    learning_rate=5e-4,
    parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1)

# ── Train ──
log("[2/6] Training 100 samples...")
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

    if (idx+1) % 20 == 0:
        elapsed = (time.time()-t0)/60
        log(f"  [{idx+1}/100] loss={loss.item():.4f}  elapsed={elapsed:.1f}m")

total_min = (time.time()-t0)/60
log(f"  Done in {total_min:.1f}m")

# ── Verify weight changes (numpy now works!) ──
log("[3/6] Verifying weight changes (numpy comparison)...")
changed = 0
unchanged = []
for k in init_w:
    cur = None
    for n, p in model.named_parameters():
        if n == k:
            cur = p.numpy().copy()
            break
    if cur is not None:
        diff = np.abs(cur - init_w[k]).max()
        if diff > 1e-8:
            changed += 1
        else:
            unchanged.append(k[-40:])
log(f"  Changed: {changed}/{len(init_w)}")
if unchanged:
    log(f"  First 3 unchanged: {unchanged[:3]}")

# ── Save LoRA (fp32 — numpy now works!) ──
log("[4/6] Saving LoRA weights...")
model.eval()
lora_dict = {}
for k, p in model.named_parameters():
    if 'lora_' in k:
        lora_dict[k] = p.numpy().copy()  # NOW WORKS!
paddle.save(lora_dict, LORA_PATH)
log(f"  Saved to {LORA_PATH} ({os.path.getsize(LORA_PATH)//1024}KB)")

# ── Reload test ──
reloaded = paddle.load(LORA_PATH)
r0 = reloaded[list(reloaded.keys())[0]]
log(f"  Reload: {len(reloaded)} keys, sample dtype={r0.dtype} shape={r0.shape}")
log(f"  Values: min={r0.min():.4f} max={r0.max():.4f} std={r0.std():.4f}")

# ── Merge LoRA into base model ──
log("[5/6] Merging LoRA into base model...")
model2 = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model2.config._attn_implementation = "flashmask"
model2.visual.config._attn_implementation = "flashmask"

LORA_SCALE = lc.lora_alpha / lc.r
merged = 0
skipped = 0
for k, p in model2.named_parameters():
    lora_A_key = f"model.{k}.lora_A"
    lora_B_key = f"model.{k}.lora_B"

    # Also try without model. prefix (depending on how LoRA was saved)
    if lora_A_key not in reloaded:
        # Try without model. prefix
        for rk in reloaded:
            if rk.endswith(f".{k}.lora_A") or rk == f"model.{k}.lora_A":
                lora_A_key = rk
                break
    if lora_B_key not in reloaded:
        for rk in reloaded:
            if rk.endswith(f".{k}.lora_B") or rk == f"model.{k}.lora_B":
                lora_B_key = rk
                break
    if lora_A_key not in reloaded or lora_B_key not in reloaded:
        continue

    A = reloaded[lora_A_key]  # numpy [hidden, r]
    B = reloaded[lora_B_key]  # numpy [r, hidden*groups]

    # Handle GQA reshape for B
    A_hidden = A.shape[0]
    if B.ndim == 2 and B.shape[1] > A_hidden and B.shape[1] % A_hidden == 0:
        groups = B.shape[1] // A_hidden
        B = B.reshape(B.shape[0], groups, A_hidden).transpose(1, 0, 2).reshape(B.shape[0] * groups, A_hidden)
        # Now B is [r*groups, hidden]

    # delta = (A @ B).T to match W shape [hidden*groups, hidden]
    # Actually: delta = B.T @ A.T but let's think again
    # A = [hidden, r], B = [r, hidden*groups]
    # A @ B = [hidden, hidden*groups]
    # delta = (A @ B).T = [hidden*groups, hidden] — matches W!
    delta = (A @ B).T  # [hidden*groups, hidden]

    W = None
    for n2, p2 in model2.named_parameters():
        if n2 == k:
            W = p2.numpy().copy()  # NOW WORKS!
            break

    if W is None or delta.shape != W.shape:
        if W is not None and delta.shape[0] < W.shape[0] and W.shape[0] % delta.shape[0] == 0:
            factor = W.shape[0] // delta.shape[0]
            delta = np.tile(delta, (factor, 1))
        if W is not None and delta.shape != W.shape:
            if skipped < 5:
                log(f"  Shape mismatch[{skipped}]: {k[-40:]} delta={delta.shape} W={W.shape}")
            skipped += 1
            continue

    W_new = W + delta * LORA_SCALE

    param_dtype = p.dtype
    try:
        p.set_value(paddle.to_tensor(W_new.astype("float16"), dtype=param_dtype, place=p.place))
        merged += 1
    except:
        skipped += 1
        if skipped <= 3:
            log(f"  set_value failed[{skipped}]: {k[-40:]}")

log(f"  Merged: {merged} layers (skipped: {skipped})")

# ── Test inference ──
log("[6/6] Testing inference with merged model...")
model2.eval()

test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl"
with open(test_path, encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()][:3]

for i, s in enumerate(test_data):
    img = Image.open(f"{DATASET_DIR}/{s['images'][0].lstrip('./')}").convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")

    with paddle.no_grad():
        gen_out = model2.generate(**inp, max_new_tokens=256, do_sample=False)

    # Handle generate output format
    if isinstance(gen_out, (list, tuple)):
        tok = gen_out[0]
    else:
        tok = gen_out
    if len(tok.shape) > 1:
        tok = tok[0]

    # Convert to list for decoding
    try:
        ids_list = tok.numpy().tolist()
        ids_list = [int(x) for x in ids_list if int(x) > 0]
        resp = processor.tokenizer.decode(ids_list, skip_special_tokens=True)
    except Exception as e:
        log(f"  Decode error: {e}, tok shape={tok.shape}")
        resp = f"[DECODE ERROR: {e}]"

    ref = s["messages"][1]["content"][:150]
    log(f"  [{i}] Pred ({len(resp.strip())} chars): {resp.strip()[:150]}")
    log(f"  [{i}] Ref  ({len(ref)} chars): {ref[:150]}")
    img.close()

log("=== ALL DONE ===")
