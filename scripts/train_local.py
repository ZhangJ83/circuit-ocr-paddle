"""V10-based local training with clean data."""
import os, sys, json, time, random

# V10 patches
sys.modules.pop('torchvision', None); import torchvision, torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu, 'get_one_valid_tokenizer_file'): mu.get_one_valid_tokenizer_file = lambda d, e: list(mu._filter_valid_tokenizer_files(d, e))

from types import ModuleType
_dummy_fc = ModuleType('dummy_flex_checkpoint')
_dummy_fc.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy_fc)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

import paddle; paddle.set_device("gpu")
import numpy as np; from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
from paddleformers.generation import GenerationConfig

sys.modules.pop('torchvision', None); import torchvision, torchvision.transforms
import transformers.utils.import_utils as tiu
tiu.is_torch_available = lambda: (True, ''); tiu.is_torchvision_available = lambda: (True, '')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import compute_all

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = PROJECT_DIR

# Model path: try local first
LOCAL_MODEL = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
if os.path.exists(LOCAL_MODEL):
    MODEL_PATH = LOCAL_MODEL; USE_HF = True
else:
    MODEL_PATH = "PaddlePaddle/PaddleOCR-VL"; USE_HF = True

OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "local")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# Config
MAX_DIM = 384; EPOCHS = 2; GRAD_ACCUM = 4; GRAD_CLIP = 1.0
CHECKPOINT_STEPS = 400
BASE_LR = 2e-5; WARMUP_STEPS = 100; ETA_MIN = 2e-6
REPETITION_PENALTY = 1.1
TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]

log(f"V10-LOCAL: max_dim={MAX_DIM}, epochs={EPOCHS}, LR={BASE_LR:.0e}, grad_accum={GRAD_ACCUM}")

# Load model
log("Loading model...")
if USE_HF:
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
        low_cpu_mem_usage=True, dtype="bfloat16")
else:
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH, load_checkpoint_format="safetensors", dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS, lora_dropout=0.05)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model, 'full'): model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

tp = [p for p in model.parameters() if not p.stop_gradient]
log(f"Trainable: {sum(p.numel() for p in tp):,}")

# Data
data_path = os.path.join(PROJECT_DIR, "output", "train_clean.jsonl")
with open(data_path, encoding="utf-8") as f:
    all_data = [json.loads(l) for l in f if l.strip()]
random.shuffle(all_data)
split = int(len(all_data) * 0.9)
train_data = all_data[:split]; val_data = all_data[split:]
total_steps = EPOCHS * len(train_data) // GRAD_ACCUM
log(f"Train: {len(train_data)}, Val: {len(val_data)}, Steps: {total_steps}")

# Optimizer
cosine = paddle.optimizer.lr.CosineAnnealingDecay(BASE_LR, T_max=total_steps-WARMUP_STEPS, eta_min=ETA_MIN)
lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=WARMUP_STEPS, start_lr=ETA_MIN, end_lr=BASE_LR)
opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

# Train
model.train()
t0 = time.time(); gs = 0; el = 0.0; opt.clear_grad()
best_loss = float('inf')
val_fixed = val_data[:10]  # Fixed 10 val samples
gc = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False)

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    for idx, s in enumerate(train_data):
        try:
            ip = s['images'][0]
            if not os.path.exists(ip): ip = ip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
            img = Image.open(ip).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buf = BytesIO(); img.save(buf, format="JPEG", quality=95); buf.seek(0); img = Image.open(buf)

            query = s["messages"][0]["content"]
            label = s["messages"][1]["content"]

            # V10 separate tokenization
            msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":query.replace("<image>","")}]}]
            pinp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
            prompt_ids = pinp["input_ids"][0]; prompt_len = prompt_ids.shape[0]

            lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=512)
            label_ids = lt["input_ids"][0]
            eos_t = paddle.to_tensor([processor.tokenizer.eos_token_id], dtype=label_ids.dtype)
            label_ids = paddle.concat([label_ids, eos_t], axis=0); label_len = label_ids.shape[0]

            full_ids = paddle.concat([prompt_ids, label_ids], axis=0).unsqueeze(0)
            full_mask = paddle.concat([pinp["attention_mask"][0], paddle.ones([label_len], dtype="int64")], axis=0).unsqueeze(0)
            labels = paddle.full([1, prompt_len + label_len], -100, dtype="int64")
            labels[0, prompt_len:] = label_ids

            out = model(input_ids=full_ids, attention_mask=full_mask,
                       pixel_values=pinp["pixel_values"], image_grid_thw=pinp.get("image_grid_thw"))
            logits = out[0] if isinstance(out, (tuple, list)) else out.logits

            shift_logits = paddle.cast(logits[:, :-1, :], "float32"); shift_labels = labels[:, 1:]
            mask = paddle.cast(shift_labels != -100, "float32")
            shift_labels_clean = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
            ce = paddle.nn.functional.cross_entropy(
                shift_logits.reshape([-1, shift_logits.shape[-1]]),
                shift_labels_clean.reshape([-1]), reduction="none").reshape(shift_labels.shape)
            loss = (ce * mask).sum() / mask.sum().clip(min=1)

            (loss / GRAD_ACCUM).backward(); el += loss.item(); img.close()

            if (idx + 1) % GRAD_ACCUM == 0 or idx == len(train_data) - 1:
                paddle.nn.utils.clip_grad_norm_(tp, GRAD_CLIP)
                opt.step(); lrs.step(); opt.clear_grad(); gs += 1

                if gs % 20 == 0:
                    eta = (time.time()-t0)/max(1,gs)*(total_steps-gs)/60
                    log(f"E{epoch+1}/{EPOCHS} S{gs}/{total_steps} loss={el/max(1,idx+1):.4f} ETA={eta:.0f}m")

                if gs % CHECKPOINT_STEPS == 0:
                    log(f"Checkpoint S{gs}...")
                    model.eval()
                    lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                    paddle.save(lora_dict, os.path.join(OUTPUT_DIR, f"checkpoint_s{gs}.pdparams"))
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        paddle.save(lora_dict, os.path.join(OUTPUT_DIR, "best.pdparams"))
                        log(f"  BEST loss={best_loss:.4f}")

                    # Validation via model.model.generate (bypass LoRA wrapper)
                    preds = []; refs = []
                    with paddle.no_grad():
                        for vs in val_fixed:
                            try:
                                vip = vs['images'][0]
                                if not os.path.exists(vip): vip = vip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
                                vimg = Image.open(vip).convert("RGB")
                                vw, vh = vimg.size
                                if max(vw, vh) > MAX_DIM:
                                    vs_ = MAX_DIM/max(vw,vh); vimg=vimg.resize((int(vw*vs_),int(vh*vs_)),Image.LANCZOS)
                                vbuf=BytesIO();vimg.save(vbuf,format="JPEG",quality=95);vbuf.seek(0);vimg=Image.open(vbuf)
                                vmsgs=[{"role":"user","content":[{"type":"image","image":vimg},{"type":"text","text":"OCR:"}]}]
                                vinp=processor.apply_chat_template(vmsgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pd")
                                out=model.model.generate(**vinp,generation_config=gc,max_new_tokens=256)
                                preds.append(processor.tokenizer.decode(out[0].tolist()[0],skip_special_tokens=True))
                                refs.append(vs['messages'][1]['content']);vimg.close()
                            except Exception as e:
                                preds.append("[ERR]");refs.append(vs['messages'][1]['content'])
                    model.train()
                    m = compute_all(preds, refs, label=f"s{gs}")
                    log(f"  Val: jf1={m['joint_f1']:.4f} CompF1={m['component_f1']:.4f} RepRate={m['repetition_rate']:.2%}")
                    if preds and preds[0]!="[ERR]": log(f"  Pred[0]: {preds[0][:80]}")
        except Exception as e:
            log(f"  SKIP {idx}: {str(e)[:60]}")
            opt.clear_grad()
            continue

tt = (time.time()-t0)/60
log(f"DONE {tt:.1f}min. Best loss={best_loss:.4f}")
