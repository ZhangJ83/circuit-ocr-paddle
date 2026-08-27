"""Stage 2: Load Projector weights → freeze → train o_proj ONLY with tiny lr.
Prevents LLM shortcut by starting from strong visual mapping."""
import os, sys, json, time, random
os.environ.update({"KMP_DUPLICATE_LIB_OK":"TRUE","FLAGS_allocator_strategy":"auto_growth"})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
from PIL import Image; from io import BytesIO

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
STAGE1_WEIGHTS = f"{OUTPUT_DIR}/lora_projector_only_336_fp16.pdparams"

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

MAX_DIM, EPOCHS, LR = 336, 1, 1e-5  # 1 epoch, very low lr
TARGETS = ["linear_1", "linear_2", "o_proj"]  # Projector + output only

log("Stage 2: Load Stage1 Projector → freeze → o_proj only, lr=1e-5")
log(f"Stage1 weights: {STAGE1_WEIGHTS}")

# Load base model
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

# Apply LoRA config
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=[".*"+t for t in TARGETS])
model = LoRAModel(model, lc)

# Load Stage 1 Projector weights
stage1 = paddle.load(STAGE1_WEIGHTS)
model.set_state_dict(stage1)
log(f"Loaded {len(stage1)} Stage1 weights")

# Freeze Projector, only train o_proj
model.mark_only_lora_as_trainable()
for name, param in model.named_parameters():
    if 'linear_1' in name or 'linear_2' in name:
        param.stop_gradient = True
model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)

trainable_weights = [n for n, p in model.named_parameters() if not p.stop_gradient]
log(f"Trainable: {len(trainable_weights)} params ({[n[:40] for n in trainable_weights[:5]]}...)")

# Data
with open(f"{DATASET_DIR}/ocr_vl_sft-train.jsonl", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]
random.shuffle(data)
total_steps = EPOCHS * len(data)
log(f"Training: {len(data)} samples x {EPOCHS} epochs = {total_steps} steps")

# Optimizer with tiny lr
lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=LR, T_max=total_steps, eta_min=1e-6)
opt = paddle.optimizer.AdamW(learning_rate=lr_scheduler,
    parameters=[p for p in model.parameters() if not p.stop_gradient], weight_decay=0.1)

model.train()
t0, global_step = time.time(), 0

for epoch in range(EPOCHS):
    random.shuffle(data)
    for idx, sample in enumerate(data):
        img_path = f"{DATASET_DIR}/{sample['images'][0].lstrip('./')}"
        if not os.path.exists(img_path): continue
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0)
        image = Image.open(buf)
        query = sample["messages"][0]["content"]
        label = sample["messages"][1]["content"][:200]

        prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
        prompt_inputs = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        full_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]},
                     {"role":"assistant","content":[{"type":"text","text":label}]}]
        full_inputs = processor.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pd")
        labels_t = paddle.full_like(full_inputs["input_ids"], -100)
        labels_t[0, prompt_len:] = full_inputs["input_ids"][0, prompt_len:]

        out = model(**full_inputs)
        logits = out[0] if isinstance(out,(tuple,list)) else out
        shift_logits = paddle.cast(logits[:,:-1,:],"float32")
        shift_labels = labels_t[:,1:]
        mask = paddle.cast(shift_labels!=-100,"float32")
        shift_labels_clamped = paddle.where(shift_labels!=-100, shift_labels, paddle.zeros_like(shift_labels))
        ce = paddle.nn.functional.cross_entropy(
            shift_logits.reshape([-1,shift_logits.shape[-1]]),
            shift_labels_clamped.reshape([-1]), reduction="none").reshape(shift_labels.shape)
        loss = (ce*mask).sum()/mask.sum().clip(min=1)
        loss.backward(); opt.step(); opt.clear_grad()
        global_step += 1; image.close()

        if global_step%500==0 or global_step==1:
            elapsed=(time.time()-t0)/60
            log(f"  [S{global_step}/{total_steps}] loss={loss.item():.4f} lr={opt.get_lr():.2e} elapsed={elapsed:.0f}m")

total_min=(time.time()-t0)/60
log(f"Stage 2 done in {total_min:.0f}m")

model.eval()
lora_dict = {k: paddle.cast(p.detach(),"float16") for k,p in model.named_parameters() if 'lora_' in k}
save_path = f"{OUTPUT_DIR}/lora_stage2_oproj.pdparams"
paddle.save(lora_dict, save_path)
log(f"Saved {len(lora_dict)} LoRA to {save_path}")
log("=== DONE ===")
