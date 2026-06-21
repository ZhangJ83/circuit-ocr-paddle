"""Quick verification: 50 samples, NO AMP, bf16, check weight changes + 2-image inference.
Target: <6 minutes end-to-end.  If this works, the full training can proceed.
"""
import os, sys, json, time, argparse, random
from pathlib import Path

os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "HF_HOME": "F:/hf_cache/hub",
    "PADDLE_HOME": "F:/paddle_cache",
    "HF_HUB_CACHE": "F:/hf_cache/hub",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "FLAGS_allocator_strategy": "auto_growth",
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ── Patch first ──
log("Applying patches...")
apply_paddle_patches()

import paddle
paddle.set_device("gpu")
log(f"GPU: {paddle.device.cuda.get_device_name(0)}")

import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

# ── Load model (bf16) ──
log("Loading model + processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

lora_cfg = LoRAConfig(r=8, lora_alpha=16,
                      target_modules=['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj'])
model = LoRAModel(model, lora_cfg)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'):
    def _full(state, *a, **kw):
        for n, p in model.model.named_parameters():
            yield n, p
    model.model.full = _full

trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
log(f"Trainable params: {trainable:,}")

# ── Save initial weights ──
init_w = {}
for k, v in model.named_parameters():
    if 'lora_' in k:
        init_w[k] = v.numpy().copy()
log(f"Saved {len(init_w)} initial LoRA matrices")

# ── Load 50 training samples ──
with open(f"{DATASET_DIR}/ocr_vl_sft-train.jsonl", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data)
data = data[:50]
log(f"Using {len(data)} training samples")

# ── Optimizer (no AMP!) ──
opt = paddle.optimizer.AdamW(
    learning_rate=5e-4,
    parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1,
)

# ── Quick train ──
from PIL import Image
from io import BytesIO

model.train()
t0 = time.time()
history = []

for idx, sample in enumerate(data):
    # ── Forward pass (same logic as train_lora_final.py) ──
    image = Image.open(f"{DATASET_DIR}/{sample['images'][0].lstrip('./')}").convert("RGB")
    w, h = image.size
    if max(w, h) > 168:
        scale = 168 / max(w, h)
        image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95); buf.seek(0)
    image = Image.open(buf)

    query = sample["messages"][0]["content"]
    label = sample["messages"][1]["content"][:200]  # truncate

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
        reduction="none",
    ).reshape(shift_labels.shape)

    loss_val = ((ce * mask).sum() / mask.sum().clip(min=1)).item()

    loss = (ce * mask).sum() / mask.sum().clip(min=1)
    loss.backward()
    opt.step()
    opt.clear_grad()

    history.append({"idx": idx, "loss": loss_val})
    image.close()

    if idx < 3:
        log(f"  Sample {idx}: loss={loss_val:.4f}")
    if (idx+1) % 10 == 0:
        elapsed = (time.time()-t0)/60
        log(f"  [{idx+1}/50] loss={loss_val:.4f}  elapsed={elapsed:.1f}m")

total_min = (time.time()-t0)/60
log(f"Training done in {total_min:.1f}m")

# ── Check weight changes ──
log("Checking weight changes...")
changed = 0
max_changes = []
for k in init_w:
    cur = model.state_dict()[k].numpy()
    diff = np.abs(cur - init_w[k]).max()
    if diff > 1e-8:
        changed += 1
        max_changes.append((diff, k))
max_changes.sort(reverse=True)
log(f"  Changed: {changed}/{len(init_w)}")
if max_changes:
    log(f"  Top diffs: {[(f'{k[-40:]}', float(d)) for d,k in max_changes[:5]]}")
if changed > 0:
    log("✅ WEIGHTS UPDATED — no AMP works!")
else:
    log("❌ STILL 0 changes — deeper issue than AMP")
    sys.exit(1)

# ── Save weights ──
lora_w = {k: v.numpy() for k, v in model.state_dict().items() if 'lora_' in k.lower()}
save_path = f"{OUTPUT_DIR}/quick_verify_lora.pdparams"
paddle.save(lora_w, save_path)
log(f"Saved LoRA to {save_path}")

# ── Quick inference on 3 test samples ──
log("Running 3-sample inference test...")
model.eval()
with open(f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl", encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()][:3]

for i, s in enumerate(test_data):
    img = Image.open(f"{DATASET_DIR}/{s['images'][0].lstrip('./')}").convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    with paddle.no_grad():
        out_ids = model.generate(**inp, max_new_tokens=256, do_sample=False)[0]
    resp = processor.decode(out_ids, skip_special_tokens=True)
    ref = s["messages"][1]["content"][:150]
    log(f"  Sample {i}:")
    log(f"    Pred: {resp[:120]}")
    log(f"    Ref:  {ref[:120]}")
    img.close()

log("=== VERIFICATION COMPLETE ===")
