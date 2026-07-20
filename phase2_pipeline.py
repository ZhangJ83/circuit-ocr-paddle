"""Phase 2: Synthetic text data + top-2 training approaches.
Generates 300 document-style text images, mixes with circuit data,
then trains exp5 (anti-overfit) and exp6 (baseline) sequentially.
"""
import os, sys, json, time, random, re, glob
from types import ModuleType
_dummy = ModuleType('dummy_flex_checkpoint')
_dummy.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuit-ocr-dataset", "scripts"))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()

DATASET_DIR = r"g:/mimo_project/circuit_ocr"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
SYNTH_DIR = os.path.join(DATASET_DIR, "output", "synth_text_images")
SYNTH_JSONL = os.path.join(DATASET_DIR, "output", "train_v10fmt_synth.jsonl")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ================================================================
# PHASE 0: Generate synthetic text images
# ================================================================
def generate_synthetic_data():
    """Create 300 document-style images with circuit text, rendered clearly."""
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(SYNTH_DIR, exist_ok=True)

    # Find a monospace font
    font_path = None
    for fp in ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
               "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
        if os.path.exists(fp):
            font_path = fp; break

    # Extract text content from existing training data
    train_path = os.path.join(DATASET_DIR, "output", "train_v10fmt.jsonl")
    all_labels = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            label = d["messages"][1]["content"]
            all_labels.append(label)

    # Also create some structured text templates
    templates = [
        # Component lists
        lambda: "\n".join([f"R{i}  {random.choice(['10k','2.2k','4.7k','100','1k','47k','330','100k','1M'])}Ω  ±{random.choice(['1','5','10'])}%" for i in range(1, random.randint(8, 20))]),
        lambda: "\n".join([f"C{i}  {random.choice(['100nF','10μF','22μF','0.1μF','1μF','100μF','47μF','4.7μF'])}  {random.choice(['50V','25V','16V','10V','100V'])}  {random.choice(['Ceramic','Electrolytic','MLCC','Tantalum'])}" for i in range(1, random.randint(8, 16))]),
        lambda: "\n".join([f"U{i}  {random.choice(['ESP32','STM32F103','ATmega328P','CH340C','AMS1117','INA219','MPU6050','MAX485','BME280'])}" for i in range(1, random.randint(5, 12))]),
        # Pin tables
        lambda: "Pin Assignments\n" + "\n".join([f"  {i:2d}  {random.choice(['VCC','GND','TX','RX','SCL','SDA','GPIO','ADC','PWM','RESET','EN','INT']):10s}  {random.choice(['Input','Output','Bidirectional','Power','Analog'])}" for i in range(1, random.randint(8, 16))]),
        # Net labels
        lambda: "Net Labels\n" + "\n".join([f"{random.choice(['VCC_5V','VDD_3V3','GND','VBUS','VSYS','VBAT','VIN','VOUT','AGND','PGND','+12V','-12V','+5V','+3.3V'])}" for _ in range(random.randint(8, 20))]),
        # Header pinouts
        lambda: f"J{random.randint(1,4)}: {random.choice(['2.54mm Pin Header','JST-XH','FFC Cable','USB-C','RJ45'])}\n" + "\n".join([f"  {i:2d}  {random.choice(['VCC','GND','SDA','SCL','TX','RX','D+','D-','VBUS','CC1','CC2','SBU1','SBU2','MISO','MOSI','SCK','CS'])}" for i in range(1, random.randint(6, 20))]),
        # BOM-style table
        lambda: "Ref    Value        Package    Qty\n" + "\n".join([f"{random.choice(['R','C','U','J','D','L','Q'])}{i:02d}  {random.choice(['10kΩ','100nF','ESP32','1N4148','SS34','10μH','2N7002']):12s}  {random.choice(['0805','SOT-23','QFN-32','SOD-123','TH','SMD']):10s}  {random.randint(1,10)}" for i in range(1, random.randint(5, 15))]),
    ]

    entries = []
    log(f"Generating 300 synthetic text images...")

    for idx in range(300):
        W, H_img = random.choice([(800, 600), (1024, 768), (1200, 800), (800, 1000), (1000, 700)])
        img = Image.new("RGB", (W, H_img), "white")
        draw = ImageDraw.Draw(img)

        # Choose content: ~60% templates, ~40% real labels
        if random.random() < 0.6:
            text = random.choice(templates)()
        else:
            text = random.choice(all_labels)
            # Truncate if too long
            if len(text) > 500:
                text = text[:500]

        # Render text
        try:
            font_size = random.choice([20, 24, 28, 32])
            fn = ImageFont.truetype(font_path, font_size)
            fn_small = ImageFont.truetype(font_path, font_size - 2)
        except:
            fn = ImageFont.load_default()
            fn_small = fn

        y = random.randint(25, 80)
        for line in text.split("\n"):
            if y > H_img - 50:
                break
            try:
                # Add slight variation in font size
                if random.random() < 0.1:
                    f_use = fn_small
                else:
                    f_use = fn
            except:
                f_use = fn
            x = random.randint(30, 60)
            draw.text((x, y), line, fill="black", font=f_use)
            y += font_size + random.randint(4, 12)

        img_path = os.path.join(SYNTH_DIR, f"s{idx:04d}.png")
        img.save(img_path, "PNG")

        # Create JSONL entry — same format as V10 training data
        entry = {
            "messages": [
                {"role": "user", "content": "<image>OCR:"},
                {"role": "assistant", "content": text}
            ],
            "images": [img_path.replace("\\", "/")]
        }
        entries.append(entry)

        if (idx + 1) % 100 == 0:
            log(f"  Generated {idx + 1}/300 images")

    log(f"  Done: {len(entries)} synthetic entries")
    return entries

# ================================================================
# PHASE 1: Generate & Mix data
# ================================================================
log("="*60)
log("PHASE 0: Synthetic Data Generation")
log("="*60)

synth_entries = generate_synthetic_data()

# Load original training data
train_path = os.path.join(DATASET_DIR, "output", "train_v10fmt.jsonl")
with open(train_path, encoding="utf-8") as f:
    orig_entries = [json.loads(l) for l in f if l.strip()]
log(f"Original training entries: {len(orig_entries)}")

# Mix: use all original + all synthetic
mixed_entries = orig_entries + synth_entries
random.shuffle(mixed_entries)

# Save mixed data
with open(SYNTH_JSONL, "w", encoding="utf-8") as f:
    for e in mixed_entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
log(f"Mixed data saved: {SYNTH_JSONL} ({len(mixed_entries)} entries = {len(orig_entries)} orig + {len(synth_entries)} synth)")
log(f"Synthetic ratio: {len(synth_entries)/len(mixed_entries)*100:.1f}%")

# ================================================================
# PHASE 2: Setup Paddle & Model
# ================================================================
log("="*60)
log("PHASE 2: Loading Model")
log("="*60)

import paddle; paddle.set_device("gpu")
import numpy as np; from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

log("Loading base model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ================================================================
# PHASE 3: Training function
# ================================================================
def run_experiment(name, lr, dropout, epochs, max_dim, data_path, freeze_projector=True):
    """Train one experiment with given config. Returns best checkpoint path."""
    OUTPUT_DIR = os.path.join(DATASET_DIR, "PaddleOCR-VL-LoRA-circuit-ocr", f"checkpoints_{name}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CHECKPOINT_STEPS = 200
    GRAD_ACCUM = 4; GRAD_CLIP = 1.0

    log(f"=== {name} === lr={lr:.0e} dropout={dropout} epochs={epochs} dim={max_dim} freeze_proj={freeze_projector}")

    # Fresh LoRA model for each experiment
    lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS, lora_dropout=dropout)
    lora_model = LoRAModel(model, lc)
    lora_model.mark_only_lora_as_trainable()
    if not hasattr(lora_model.model, 'full'):
        lora_model.model.full = lambda *a, **kw: iter(lora_model.model.named_parameters())

    tp = [p for p in lora_model.parameters() if not p.stop_gradient]
    log(f"  Trainable: {sum(p.numel() for p in tp):,}")

    # Data
    with open(data_path, encoding="utf-8") as f:
        all_data = [json.loads(l) for l in f if l.strip()]
    random.shuffle(all_data)
    split = int(len(all_data) * 0.9)
    train_data = all_data[:split]; val_data = all_data[split:]
    total_steps = epochs * len(train_data) // GRAD_ACCUM
    log(f"  Train: {len(train_data)} Val: {len(val_data)} Steps: {total_steps}")

    cosine = paddle.optimizer.lr.CosineAnnealingDecay(lr, T_max=max(1, total_steps - 100), eta_min=lr/10)
    lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=100, start_lr=lr/10, end_lr=lr)
    opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

    monitor_samples = val_data[:3]
    lora_model.train(); t0 = time.time(); gs = 0; el_acc = 0.0; opt.clear_grad()
    best_loss = float('inf'); best_ckpt = None; skipped = 0

    def quick_inference(samples, max_tokens=60):
        preds = []
        for s in samples:
            try:
                img_path = s['images'][0]
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                content_raw = s["messages"][0]["content"]
                if isinstance(content_raw, list):
                    text_parts = [item["text"] for item in content_raw if item.get("type") == "text"]
                    query = text_parts[0] if text_parts else "<image>OCR:"
                else:
                    query = content_raw
                msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":query.replace("<image>","")}]}]
                inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
                input_ids = inp["input_ids"]; attn = inp["attention_mask"]
                pv = inp.get("pixel_values"); igt = inp.get("image_grid_thw")
                gen = []
                with paddle.no_grad():
                    for _ in range(max_tokens):
                        out = lora_model(input_ids=input_ids, attention_mask=attn, pixel_values=pv, image_grid_thw=igt)
                        logits = out[0] if isinstance(out, (list, tuple)) else out.logits
                        ntl = logits[:, -1, :]
                        for tid in set(gen):
                            sc = float(ntl[0, tid]); ntl[0, tid] = sc * 1.1 if sc < 0 else sc / 1.1
                        nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
                        if nt == processor.tokenizer.eos_token_id: break
                        gen.append(nt)
                        input_ids = paddle.concat([input_ids, paddle.to_tensor([[nt]])], axis=1)
                        attn = paddle.concat([attn, paddle.ones([1,1], dtype=attn.dtype)], axis=1)
                preds.append(processor.tokenizer.decode(gen, skip_special_tokens=True))
                img.close()
            except Exception as e:
                preds.append(f"[ERR:{str(e)[:30]}]")
        return preds

    for epoch in range(epochs):
        random.shuffle(train_data)
        log(f"  --- Epoch {epoch+1}/{epochs} ---")
        for idx, sample in enumerate(train_data):
            try:
                img_path = sample['images'][0]
                if not os.path.exists(img_path):
                    skipped += 1; continue
                image = Image.open(img_path).convert("RGB")
                w, h = image.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h); image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0); image = Image.open(buf)

                content_raw = sample["messages"][0]["content"]
                if isinstance(content_raw, list):
                    text_parts = [item["text"] for item in content_raw if item.get("type") == "text"]
                    query = text_parts[0] if text_parts else "<image>OCR:"
                else:
                    query = content_raw
                label = sample["messages"][1]["content"]

                prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
                prompt_inputs = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
                prompt_ids = prompt_inputs["input_ids"][0]; prompt_len = prompt_ids.shape[0]

                lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=512)
                label_ids = lt["input_ids"][0]
                eos_t = paddle.to_tensor([processor.tokenizer.eos_token_id], dtype=label_ids.dtype)
                label_ids = paddle.concat([label_ids, eos_t], axis=0); label_len = label_ids.shape[0]

                full_ids = paddle.concat([prompt_ids, label_ids], axis=0).unsqueeze(0)
                full_mask = paddle.concat([prompt_inputs["attention_mask"][0], paddle.ones([label_len], dtype="int64")], axis=0).unsqueeze(0)
                labels_t = paddle.full([1, prompt_len + label_len], -100, dtype="int64")
                labels_t[0, prompt_len:] = label_ids

                out = lora_model(input_ids=full_ids, attention_mask=full_mask,
                               pixel_values=prompt_inputs["pixel_values"], image_grid_thw=prompt_inputs.get("image_grid_thw"))
                logits = out[0] if isinstance(out, (tuple, list)) else out.logits
                shift_logits = paddle.cast(logits[:, :-1, :], "float32"); shift_labels = labels_t[:, 1:]
                mask = paddle.cast(shift_labels != -100, "float32")
                shift_labels_clean = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
                ce = paddle.nn.functional.cross_entropy(
                    shift_logits.reshape([-1, shift_logits.shape[-1]]),
                    shift_labels_clean.reshape([-1]), reduction="none").reshape(shift_labels.shape)
                loss = (ce * mask).sum() / mask.sum().clip(min=1)
                (loss / GRAD_ACCUM).backward(); el_acc += loss.item()
                image.close()

                if (idx + 1) % GRAD_ACCUM == 0 or idx == len(train_data) - 1:
                    paddle.nn.utils.clip_grad_norm_(tp, GRAD_CLIP)
                    opt.step(); lrs.step(); opt.clear_grad(); gs += 1

                    if gs % 20 == 0:
                        eta = (time.time()-t0)/max(1,gs)*(total_steps-gs)/60
                        avg_loss = el_acc/max(1, idx+1)
                        log(f"  S{gs}/{total_steps} loss={avg_loss:.4f} lr={opt.get_lr():.2e} ETA={eta:.0f}m")

                    if gs % CHECKPOINT_STEPS == 0:
                        log(f"  --- Checkpoint S{gs} ---")
                        lora_model.eval()
                        lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in lora_model.named_parameters() if 'lora_' in k}
                        ckpt_path = os.path.join(OUTPUT_DIR, f"lora_s{gs}.pdparams")
                        paddle.save(lora_dict, ckpt_path)
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            best_ckpt = ckpt_path
                            paddle.save(lora_dict, os.path.join(OUTPUT_DIR, "best.pdparams"))
                        log(f"    Saved. loss={loss.item():.4f}")
                        preds = quick_inference(monitor_samples)
                        for mi, pred in enumerate(preds[:2]):
                            ref = monitor_samples[mi]["messages"][1]["content"][:60]
                            log(f"    [{mi}] Pred: {pred[:60]}")
                            log(f"    [{mi}] Ref:  {ref}")
                        lora_model.train()
            except Exception as e:
                skipped += 1
                if skipped <= 3: log(f"    SKIP: {str(e)[:60]}")
                try: opt.clear_grad()
                except: pass
                continue

    total_min = (time.time()-t0)/60
    log(f"  DONE {total_min:.0f}m. Best loss={best_loss:.4f} Skipped={skipped}")
    return best_ckpt if best_ckpt else os.path.join(OUTPUT_DIR, "best.pdparams")

# ================================================================
# PHASE 4: Run experiments
# ================================================================
log("="*60)
log("PHASE 3: Training Experiments")
log("="*60)

results = {}

# exp5: Anti-overfit + synthetic (the most promising approach)
log("\n>>> EXP5: Anti-overfit + Synthetic Data <<<")
exp5_ckpt = run_experiment(
    name="exp5_antioverfit_synth",
    lr=1e-5, dropout=0.10, epochs=2, max_dim=384,
    data_path=SYNTH_JSONL, freeze_projector=True)
results['exp5'] = exp5_ckpt
log(f"exp5 best checkpoint: {exp5_ckpt}")

# exp6: 512px high-res + synthetic (different dimension, same data)
log("\n>>> EXP6: High-Res 512px + Synthetic Data <<<")
exp6_ckpt = run_experiment(
    name="exp6_hires_synth",
    lr=2e-5, dropout=0.05, epochs=2, max_dim=512,
    data_path=SYNTH_JSONL, freeze_projector=True)
results['exp6'] = exp6_ckpt
log(f"exp6 best checkpoint: {exp6_ckpt}")

# ================================================================
# PHASE 5: Quick eval
# ================================================================
log("="*60)
log("PHASE 4: Quick Evaluation")
log("="*60)

# Load test data
test_path = os.path.join(DATASET_DIR, "output", "test_clean.jsonl")
with open(test_path, encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()][:30]
refs = [s["messages"][1]["content"] for s in test_data]
log(f"Test samples: {len(test_data)}")

import Levenshtein

def extract_components(text):
    return re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', text)

def comp_f1(preds, refs):
    precs, recs, f1s = [], [], []
    for p, r in zip(preds, refs):
        pc = set(extract_components(p)); rc = set(extract_components(r))
        if not pc and not rc: precs.append(1.0); recs.append(1.0); f1s.append(1.0)
        elif not pc or not rc: precs.append(0.0); recs.append(0.0); f1s.append(0.0)
        else:
            tp = len(pc & rc); prec = tp/len(pc); rec = tp/len(rc)
            f1s.append(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0)
            precs.append(prec); recs.append(rec)
    return sum(f1s)/len(f1s)

def compute_joint_f1(preds, refs):
    def norm_val(v):
        return v.strip().rstrip(',').rstrip(';').replace(' ','').upper()
    def parse_pairs(text):
        pairs = set()
        for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))', text):
            val = norm_val(m.group(2).strip())
            if val and len(val) < 50: pairs.add((m.group(1), val))
        return pairs
    precs, recs, f1s = [], [], []
    for p, r in zip(preds, refs):
        pp = parse_pairs(p); rp = parse_pairs(r)
        if not pp and not rp: precs.append(1.0); recs.append(1.0); f1s.append(1.0)
        elif not pp or not rp: precs.append(0.0); recs.append(0.0); f1s.append(0.0)
        else:
            tp = len(pp & rp); prec = tp/len(pp); rec = tp/len(rp)
            f1s.append(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0)
            precs.append(prec); recs.append(rec)
    return sum(f1s)/len(f1s)

def run_eval(ckpt_path, name, max_dim=384):
    log(f"\n  Evaluating: {name}")
    state = paddle.load(ckpt_path)
    for k, p in model.named_parameters():
        if k in state:
            v = state[k]
            if p.dtype != v.dtype: v = paddle.cast(v, p.dtype)
            if list(p.shape) == list(v.shape):
                p.set_value(v)

    preds = []
    model.eval()
    for i, s in enumerate(test_data):
        try:
            img_path = s['images'][0].replace('/root/circuit_ocr/', 'g:/mimo_project/circuit_ocr/')
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buf = BytesIO(); img.save(buf, format="JPEG", quality=95); buf.seek(0); img = Image.open(buf)
            content_raw = s["messages"][0]["content"]
            if isinstance(content_raw, list):
                text_parts = [item["text"] for item in content_raw if item.get("type") == "text"]
                query = text_parts[0] if text_parts else "<image>OCR:"
            else:
                query = content_raw
            msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":query.replace("<image>","")}]}]
            inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
            input_ids = inp["input_ids"]; attn = inp["attention_mask"]
            pv = inp.get("pixel_values"); igt = inp.get("image_grid_thw")
            gen = []
            with paddle.no_grad():
                for _ in range(80):
                    out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pv, image_grid_thw=igt)
                    logits = out[0] if isinstance(out, (list, tuple)) else out.logits
                    ntl = logits[:, -1, :]
                    for tid in set(gen):
                        sc = float(ntl[0, tid]); ntl[0, tid] = sc * 1.1 if sc < 0 else sc / 1.1
                    nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
                    if nt == processor.tokenizer.eos_token_id: break
                    gen.append(nt)
                    input_ids = paddle.concat([input_ids, paddle.to_tensor([[nt]])], axis=1)
                    attn = paddle.concat([attn, paddle.ones([1,1], dtype=attn.dtype)], axis=1)
            preds.append(processor.tokenizer.decode(gen, skip_special_tokens=True))
            img.close()
        except Exception as e:
            preds.append(f"[ERR:{str(e)[:30]}]")
    model.train()

    cf1 = comp_f1(preds, refs)
    jf1 = compute_joint_f1(preds, refs)
    ned = sum(Levenshtein.distance(p,r)/max(len(p),len(r),1) for p,r in zip(preds,refs))/len(preds)
    div = len(set(p.strip() for p in preds))/len(preds)
    log(f"    CompF1={cf1:.4f}  JointF1={jf1:.4f}  NED={ned:.4f}  Div={div:.2%}")
    log(f"    Sample0: {preds[0][:80]}")
    log(f"    Sample1: {preds[1][:80]}")
    return {"name": name, "comp_f1": cf1, "joint_f1": jf1, "ned": ned, "div": div, "preds": preds}

eval_results = []
for ckpt, name, dim in [
    (results['exp5'], 'exp5_antioverfit_synth', 384),
    (results['exp6'], 'exp6_hires_synth', 512),
]:
    eval_results.append(run_eval(ckpt, name, dim))

# Also eval previous best for comparison
prev_best_path = os.path.join(DATASET_DIR, "PaddleOCR-VL-LoRA-circuit-ocr", "lora_exp4_final.pdparams")
if os.path.exists(prev_best_path):
    eval_results.append(run_eval(prev_best_path, 'prev_best_exp4_final', 384))

# ================================================================
# FINAL SUMMARY
# ================================================================
log("\n" + "="*60)
log("FINAL RESULTS")
log("="*60)
log(f"{'Experiment':<30} {'CompF1':>8} {'JointF1':>8} {'NED':>8} {'Div':>8}")
log("-"*70)
for r in eval_results:
    log(f"{r['name']:<30} {r['comp_f1']:>8.4f} {r['joint_f1']:>8.4f} {r['ned']:>8.4f} {r['div']:>7.2%}")

best = max(eval_results, key=lambda x: x['joint_f1'])
log(f"\nBest JointF1: {best['name']} = {best['joint_f1']:.4f}")
best_cf = max(eval_results, key=lambda x: x['comp_f1'])
log(f"Best CompF1: {best_cf['name']} = {best_cf['comp_f1']:.4f}")

# Save results
res_path = os.path.join(DATASET_DIR, "phase2_results.json")
with open(res_path, "w") as f:
    json.dump([{k: v for k, v in r.items() if k != 'preds'} for r in eval_results], f, indent=2, ensure_ascii=False)
log(f"\nSaved: {res_path}")
log("\nDONE!")
