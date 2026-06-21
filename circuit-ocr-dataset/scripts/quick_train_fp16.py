"""Quick train + save (fp16) + inference test.
Minimal, focused — verifies the full pipeline in <8 min.
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

log("Patches...")
apply_paddle_patches()

import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

# ── Load ──
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
log(f"Trainable: {sum(p.size for p in model.parameters() if not p.stop_gradient):,}")

# ── Init clone (paddle ops) ──
init_w = {}
for k, p in model.named_parameters():
    if 'lora_' in k:
        init_w[k] = p.detach().clone()
log(f"Cloned {len(init_w)} init weights")

# ── Data ──
with open(f"{DATASET_DIR}/ocr_vl_sft-train.jsonl", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data); data = data[:80]
log(f"Training on {len(data)} samples")

opt = paddle.optimizer.AdamW(
    learning_rate=5e-4,
    parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1)

# ── Train ──
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
    if (idx+1) % 20 == 0:
        elapsed = (time.time()-t0)/60
        log(f"  [{idx+1}/80] loss={loss.item():.4f}  elapsed={elapsed:.1f}m")

total_min = (time.time()-t0)/60
log(f"Training done in {total_min:.1f}m")

# ── Verify weight changes ──
changed = sum(1 for k in init_w for n, p in model.named_parameters()
              if n == k and abs(paddle.max(paddle.abs(p - init_w[k])).item()) > 1e-8)
log(f"Weights changed: {changed}/{len(init_w)}")

# ── Save as FLOAT16 (can be loaded back!) ──
model.eval()
lora_fp16 = {}
for k, p in model.named_parameters():
    if 'lora_' in k:
        lora_fp16[k] = paddle.cast(p.detach(), "float16")
save_path = f"{OUTPUT_DIR}/lora_trained_fp16.pdparams"
paddle.save(lora_fp16, save_path)
log(f"Saved LoRA (fp16) to {save_path} ({os.path.getsize(save_path)//1024}KB)")

# ── Reload test ──
reloaded = paddle.load(save_path)
log(f"Reload test: {len(reloaded)} keys match={all(k in reloaded for k in lora_fp16)}")
# Check first value
k0 = list(reloaded.keys())[0]
# Use fp64 to bypass float8_e4m3fn alias issue
r0 = paddle.cast(reloaded[k0], "float64").numpy()
r0_nonan = r0[np.isfinite(r0)]
if len(r0_nonan) > 0:
    log(f"  Sample {k0[-40:]}: min={r0_nonan.min():.4f} max={r0_nonan.max():.4f} std={r0_nonan.std():.4f}")
else:
    log(f"  Sample {k0[-40:]}: ALL NaN — need different conversion")

# ── Inference with LoRA model ──
log("Testing LoRA inference...")
model.eval()
test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl"
with open(test_path, encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()][:3]

for i, s in enumerate(test_data):
    img = Image.open(f"{DATASET_DIR}/{s['images'][0].lstrip('./')}").convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    with paddle.no_grad():
        out_ids = model.generate(**inp, max_new_tokens=256, do_sample=False)[0]
    resp = processor.decode(out_ids, skip_special_tokens=True)
    ref = s["messages"][1]["content"][:150]
    # Count non-empty chars
    pred_brief = resp.strip()[:120]
    log(f"  [{i}] Pred ({len(resp.strip())} chars): '{pred_brief}'")
    log(f"  [{i}] Ref  ({len(ref)} chars): '{ref[:100]}'")
    img.close()

log("=== DONE ===")
