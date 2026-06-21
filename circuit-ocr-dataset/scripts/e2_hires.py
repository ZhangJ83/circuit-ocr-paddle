"""E2: High-resolution training test. max_dim=336, 200 samples, 3 epochs.
Hypothesis: 168px too small to read circuit text, 336px should help.
"""
import os, sys, json, time, random
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HOME": "F:/hf_cache/hub",
    "PADDLE_HOME": "F:/paddle_cache", "HF_HUB_CACHE": "F:/hf_cache/hub",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "FLAGS_allocator_strategy": "auto_growth",
})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
MAX_DIM = 336  # 2x the old size
NUM_SAMPLES = 200
EPOCHS = 3

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    try: print(f"[{ts}] {msg}", flush=True)
    except: print(f"[{ts}] {msg.encode('ascii','replace').decode('ascii')}", flush=True)

log(f"E2: max_dim={MAX_DIM}, {NUM_SAMPLES} samples, {EPOCHS} epochs")

# Load model
log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
lc = LoRAConfig(r=8, lora_alpha=16, target_modules=['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj'])
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
log(f"Trainable: {sum(p.size for p in model.parameters() if not p.stop_gradient):,}")

# Data
with open(f"{DATASET_DIR}/ocr_vl_sft-train.jsonl", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data); data = data[:NUM_SAMPLES]
log(f"Training samples: {len(data)}")

from PIL import Image; from io import BytesIO
opt = paddle.optimizer.AdamW(
    learning_rate=5e-4, parameters=[p for p in model.parameters() if not p.stop_gradient],
    weight_decay=0.1)

# Train
model.train()
t0 = time.time()
global_step = 0
total_steps = EPOCHS * len(data)

for epoch in range(EPOCHS):
    random.shuffle(data)
    for idx, sample in enumerate(data):
        img_path = f"{DATASET_DIR}/{sample['images'][0].lstrip('./')}"
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0)
        image = Image.open(buf)

        query = sample["messages"][0]["content"]
        label = sample["messages"][1]["content"][:300]  # longer label for bigger images

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
            shift_labels_clamped.reshape([-1]), reduction="none").reshape(shift_labels.shape)
        loss = (ce * mask).sum() / mask.sum().clip(min=1)
        loss.backward(); opt.step(); opt.clear_grad()
        global_step += 1
        image.close()

        if global_step % 50 == 0:
            elapsed = (time.time()-t0)/60
            log(f"  [E{epoch+1} S{global_step}/{total_steps}] loss={loss.item():.4f} elapsed={elapsed:.0f}m ETA={(elapsed/global_step*total_steps-elapsed):.0f}m")

total_min = (time.time()-t0)/60
log(f"Training done in {total_min:.0f}m")

# Test inference on 5 test samples
log("Testing inference...")
model.eval()
test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl"
with open(test_path, encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()][:5]

for i, s in enumerate(test_data):
    img = Image.open(f"{DATASET_DIR}/{s['images'][0].lstrip('./')}").convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    with paddle.no_grad():
        out = model.generate(**inp, max_new_tokens=50, do_sample=False)
    if isinstance(out, (list,tuple)): tok = out[0]
    else: tok = out
    if len(tok.shape)>1: tok = tok[0]
    ids = [int(x) for x in tok.numpy().tolist() if int(x)>0]
    resp = processor.tokenizer.decode(ids, skip_special_tokens=True)
    ref = s["messages"][1]["content"][:200]
    log(f"  [{i}] Pred({len(resp)}ch): {resp[:150]}")
    log(f"  [{i}] Ref({len(ref)}ch): {ref[:100]}")
    img.close()

log("=== E2 DONE ===")
