"""Batch evaluation of all experiment checkpoints."""
import os, sys, json, time, re, glob
from types import ModuleType
_dummy = ModuleType('dummy_flex_checkpoint')
_dummy.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuit-ocr-dataset", "scripts"))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np; from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
import Levenshtein

DATASET_DIR = r"g:/mimo_project/circuit_ocr"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
MAX_DIM = 384
MAX_TOKENS = 80

# ── Metrics ──
def extract_components(text):
    pattern = r'\b((?:LED|[RCDLQUJYF])\d+)\b'
    return re.findall(pattern, text)

def compute_component_f1(predictions, references):
    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        pred_comps = set(extract_components(pred))
        ref_comps = set(extract_components(ref))
        if not pred_comps and not ref_comps:
            precisions.append(1.0); recalls.append(1.0); f1s.append(1.0)
        elif not pred_comps or not ref_comps:
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)
        else:
            tp = len(pred_comps & ref_comps)
            prec = tp / len(pred_comps); rec = tp / len(ref_comps)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
    return {"precision": sum(precisions)/len(precisions), "recall": sum(recalls)/len(recalls), "f1": sum(f1s)/len(f1s)}

def compute_joint_f1(predictions, references):
    """Parse (refdes, value) pairs - both must match."""
    pair_pat = re.compile(r'\b((?:LED|[RCDLQUJYF])\d+)\s+(.+?)(?=\s*(?:\b(?:LED|[RCDLQUJYF])\d+\b|$))', re.DOTALL)
    precs, recs, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        # Normalize values
        def norm_val(v):
            v = v.strip().rstrip(',').rstrip(';')
            v = v.replace(' ', '').replace('Ω', 'Ω').replace('Ohm', 'Ω').replace('ohm', 'Ω')
            return v.upper()
        def parse_pairs(text):
            pairs = set()
            for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))', text):
                refdes = m.group(1)
                val = norm_val(m.group(2).strip())
                if val and len(val) < 50:
                    pairs.add((refdes, val))
            return pairs
        p_pairs = parse_pairs(pred); r_pairs = parse_pairs(ref)
        if not p_pairs and not r_pairs:
            precs.append(1.0); recs.append(1.0); f1s.append(1.0)
        elif not p_pairs or not r_pairs:
            precs.append(0.0); recs.append(0.0); f1s.append(0.0)
        else:
            tp = len(p_pairs & r_pairs)
            prec = tp / len(p_pairs); rec = tp / len(r_pairs)
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
            precs.append(prec); recs.append(rec); f1s.append(f1)
    return {"precision": sum(precs)/len(precs), "recall": sum(recs)/len(recs), "f1": sum(f1s)/len(f1s)}

def compute_ned(predictions, references):
    d = [Levenshtein.distance(p,r)/max(len(p),len(r),1) for p,r in zip(predictions, references)]
    return sum(d)/len(d)

def compute_repetition_rate(predictions, min_repeat=4):
    repeated = 0
    for pred in predictions:
        lines = pred.strip().split('\n')
        if len(lines) < min_repeat: continue
        max_run = 1; cur = 1
        for i in range(1, len(lines)):
            cur = cur+1 if lines[i].strip()==lines[i-1].strip() else 1
            max_run = max(max_run, cur)
        if max_run >= min_repeat: repeated += 1
    return repeated/len(predictions)

def compute_token_recall(predictions, references):
    rec = []
    for p, r in zip(predictions, references):
        pt = set(p.split()); rt = set(r.split())
        rec.append(1.0 if not rt else len(pt&rt)/len(rt) if pt else 0.0)
    return sum(rec)/len(rec)

def compute_diversity(predictions):
    return len(set(p.strip() for p in predictions)) / len(predictions)

def run_inference(model, processor, samples):
    preds = []
    for i, s in enumerate(samples):
        try:
            img_path = s['images'][0].replace('/root/circuit_ocr/', 'g:/mimo_project/circuit_ocr/')
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buf = BytesIO(); img.save(buf, format="JPEG", quality=95); buf.seek(0); img = Image.open(buf)

            # Handle both list and string content formats
            content_raw = s["messages"][0]["content"]
            if isinstance(content_raw, list):
                # Extract text from list format: [{"type":"image",...}, {"type":"text","text":"<image>OCR:"}]
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
                for _ in range(MAX_TOKENS):
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
            preds.append(f"[ERR:{str(e)[:40]}]")
        if (i+1) % 30 == 0:
            print(f"    {i+1}/{len(samples)}", flush=True)
    return preds

print("="*60)
print("BATCH EVALUATION")
print("="*60)

# Load data
test_path = os.path.join(DATASET_DIR, "output", "test_clean.jsonl")
with open(test_path, encoding="utf-8") as f:
    test_data_all = [json.loads(l) for l in f if l.strip()]
# Quick eval with 30 samples first
test_data = test_data_all[:30]
print(f"Test samples: {len(test_data)} (quick eval)")
refs = [s["messages"][1]["content"] for s in test_data]

# Load model once
print("\nLoading base model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS, lora_dropout=0.05)
model = LoRAModel(model, lc)
if not hasattr(model.model, 'full'): model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model.eval()
print("Model ready.\n")

# Find all checkpoints
ckpt_dir = os.path.join(DATASET_DIR, "PaddleOCR-VL-LoRA-circuit-ocr")
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoints_exp*/lora_s*.pdparams"))) + \
        sorted(glob.glob(os.path.join(ckpt_dir, "lora_exp*_best.pdparams"))) + \
        sorted(glob.glob(os.path.join(ckpt_dir, "lora_exp*_final.pdparams")))

# Skip duplicates by basename
seen = set()
unique_ckpts = []
for c in ckpts:
    bn = os.path.basename(c)
    if bn not in seen:
        seen.add(bn)
        unique_ckpts.append(c)

print(f"Checkpoints to evaluate: {len(unique_ckpts)}\n")

results = []
for ci, ckpt_path in enumerate(unique_ckpts):
    name = os.path.relpath(ckpt_path, ckpt_dir).replace("\\", "/")
    print(f"[{ci+1}/{len(unique_ckpts)}] {name}...", end=" ", flush=True)

    try:
        state = paddle.load(ckpt_path)
        # set values
        for k, p in model.named_parameters():
            if k in state:
                v = state[k]
                if p.dtype != v.dtype:
                    v = paddle.cast(v, p.dtype)
                if list(p.shape) != list(v.shape):
                    print(f"SKIP (shape mismatch: {k})")
                    continue
                p.set_value(v)

        preds = run_inference(model, processor, test_data)

        r = {
            "checkpoint": name,
            "comp_f1": compute_component_f1(preds, refs),
            "joint_f1": compute_joint_f1(preds, refs),
            "ned": compute_ned(preds, refs),
            "token_recall": compute_token_recall(preds, refs),
            "rep_rate": compute_repetition_rate(preds),
            "diversity": compute_diversity(preds),
        }
        results.append(r)

        print(f"CompF1={r['comp_f1']['f1']:.4f} JointF1={r['joint_f1']['f1']:.4f} "
              f"NED={r['ned']:.4f} RepRate={r['rep_rate']:.2%} Div={r['diversity']:.2%}")

        # Print 2 sample predictions
        for si in [0, min(1, len(preds)-1)]:
            print(f"    [{si}] {preds[si][:80]}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY (sorted by JointF1)")
print("="*60)
results.sort(key=lambda x: x['joint_f1']['f1'], reverse=True)
print(f"{'Checkpoint':<45} {'CompF1':>8} {'JointF1':>8} {'NED':>8} {'RepRate':>8} {'Div':>8} {'TokRec':>8}")
print("-"*100)
for r in results:
    print(f"{r['checkpoint']:<45} {r['comp_f1']['f1']:>8.4f} {r['joint_f1']['f1']:>8.4f} "
          f"{r['ned']:>8.4f} {r['rep_rate']:>7.2%} {r['diversity']:>7.2%} {r['token_recall']:>8.4f}")

# Save JSON
out_path = os.path.join(ckpt_dir, "batch_eval_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out_path}")

best = results[0]
print(f"\nBEST: {best['checkpoint']}")
print(f"   CompF1={best['comp_f1']['f1']:.4f}  JointF1={best['joint_f1']['f1']:.4f}  NED={best['ned']:.4f}")
