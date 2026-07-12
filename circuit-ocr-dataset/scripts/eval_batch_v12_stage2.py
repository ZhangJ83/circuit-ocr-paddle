#!/usr/bin/env python3
"""
Batch evaluate all V12-Stage2 checkpoints against the test set.
Loads model once, iterates through checkpoints, saves per-checkpoint results.
"""
import os, sys, json, time, re
from pathlib import Path
from collections import Counter

# ── Early patches ──
from types import ModuleType
_dummy_fc = ModuleType('dummy_flex_checkpoint')
_dummy_fc.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy_fc)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
local_hf_cache = "F:/hf_cache/hub"
if os.path.exists(local_hf_cache):
    os.environ["HF_HOME"] = local_hf_cache
    os.environ["HF_HUB_CACHE"] = local_hf_cache

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from eval_benchmark import apply_paddle_patches
apply_paddle_patches()

import paddle
import numpy as np
import Levenshtein
from PIL import Image
from io import BytesIO

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

# ==================== Config ====================
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CKPT_DIR = os.path.join(DATASET_DIR, "PaddleOCR-VL-LoRA-circuit-ocr", "checkpoints_v12_stage2")
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
TEST_DATA = os.path.join(DATASET_DIR, "ocr_vl_sft-test-easy50-pure.jsonl")
OUTPUT_DIR = os.path.join(DATASET_DIR, "PaddleOCR-VL-LoRA-circuit-ocr", "eval_v12_stage2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
LORA_R = 16
LORA_ALPHA = 32
MAX_DIM = 384
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.1

# Checkpoints to evaluate
CHECKPOINTS = ["s200", "s400", "s600", "s800", "s1000", "s1200", "s1400", "s1600"]

# ==================== Metrics ====================
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
            prec = tp / len(pred_comps) if pred_comps else 0.0
            rec = tp / len(ref_comps) if ref_comps else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
    }

def compute_token_recall(predictions, references):
    recalls = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split()); ref_tokens = set(ref.split())
        if not ref_tokens: recalls.append(1.0)
        elif not pred_tokens: recalls.append(0.0)
        else: recalls.append(len(pred_tokens & ref_tokens) / len(ref_tokens))
    return sum(recalls) / len(recalls) if recalls else 0.0

def compute_repetition_rate(predictions, min_repeat=4):
    repeated = 0
    for pred in predictions:
        lines = pred.strip().split('\n')
        if len(lines) < min_repeat: continue
        max_run = 1; current_run = 1
        for i in range(1, len(lines)):
            if lines[i].strip() == lines[i-1].strip():
                current_run += 1; max_run = max(max_run, current_run)
            else: current_run = 1
        if max_run >= min_repeat: repeated += 1
    return repeated / len(predictions) if predictions else 0.0, repeated

def compute_ned(predictions, references):
    distances = []
    for pred, ref in zip(predictions, references):
        d = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref), 1)
        distances.append(d / max_len)
    return sum(distances) / len(distances) if distances else 1.0

# ==================== Main ====================
def main():
    paddle.set_device("gpu")

    # Load test data
    samples = []
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Test samples: {len(samples)}")

    # Load base model + processor
    print(f"Loading model from: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH, convert_from_hf=True, load_checkpoint_format='naive',
        low_cpu_mem_usage=True, dtype="bfloat16"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    # Apply LoRA wrapper (weights will be replaced per checkpoint)
    lc = LoRAConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGETS)
    model = LoRAModel(model, lc)
    model_lora_params = {k: p for k, p in model.named_parameters() if 'lora_' in k}
    print(f"LoRA params in model: {len(model_lora_params)}")

    all_metrics = {}

    for ckpt_name in CHECKPOINTS:
        ckpt_path = os.path.join(CKPT_DIR, f"lora_{ckpt_name}.pdparams")
        if not os.path.exists(ckpt_path):
            print(f"\n{'='*60}")
            print(f"SKIP {ckpt_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}")
        print(f"Checkpoint: {ckpt_path}")

        # Load checkpoint weights
        lora_state = paddle.load(ckpt_path)
        loaded = 0
        for ckpt_key, ckpt_value in lora_state.items():
            if ckpt_key in model_lora_params:
                p = model_lora_params[ckpt_key]
                ckpt_tensor = paddle.cast(ckpt_value, p.dtype)
                p.set_value(ckpt_tensor)
                loaded += 1
        print(f"  Loaded {loaded}/{len(lora_state)} LoRA params")

        model.eval()
        paddle.device.cuda.empty_cache()

        # Evaluate
        results = []
        total_time = 0
        for i, sample in enumerate(samples):
            start = time.time()
            query = sample["messages"][0]["content"]
            image_path = sample["images"][0]

            img_resolved_path = Path(image_path)
            data_dir = Path(TEST_DATA).parent
            if not img_resolved_path.exists():
                alt = data_dir / image_path
                if alt.exists(): img_resolved_path = alt
                else:
                    alt2 = data_dir / img_resolved_path.name
                    if alt2.exists(): img_resolved_path = alt2

            try:
                image = Image.open(img_resolved_path).convert("RGB")
                w, h = image.size
                if max(w, h) > MAX_DIM:
                    scale = MAX_DIM / max(w, h)
                    image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = BytesIO()
                image.save(buf, format='JPEG', quality=95)
                buf.seek(0)
                image = Image.open(buf)

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": query.replace("<image>", "")},
                    ],
                }]
                inp = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pd"
                )

                input_ids = inp["input_ids"]
                attention_mask = inp["attention_mask"]
                pixel_values = inp.get("pixel_values")
                image_grid_thw = inp.get("image_grid_thw")

                generated = []
                with paddle.no_grad():
                    for _ in range(MAX_NEW_TOKENS):
                        outputs = model(
                            input_ids=input_ids, attention_mask=attention_mask,
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw
                        )
                        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs.logits
                        next_token_logits = logits[:, -1, :]

                        if REPETITION_PENALTY != 1.0 and generated:
                            for tid in set(generated):
                                score = float(next_token_logits[0, tid])
                                if score < 0:
                                    next_token_logits[0, tid] = score * REPETITION_PENALTY
                                else:
                                    next_token_logits[0, tid] = score / REPETITION_PENALTY

                        next_token = int(paddle.argmax(next_token_logits, axis=-1).numpy()[0])
                        if next_token == processor.tokenizer.eos_token_id:
                            break
                        generated.append(next_token)
                        next_tensor = paddle.to_tensor([[next_token]], dtype=input_ids.dtype)
                        input_ids = paddle.concat([input_ids, next_tensor], axis=1)
                        attention_mask = paddle.concat(
                            [attention_mask, paddle.ones([1, 1], dtype=attention_mask.dtype)], axis=1
                        )

                prediction = processor.tokenizer.decode(generated, skip_special_tokens=True)
                elapsed = time.time() - start
                total_time += elapsed

                results.append({
                    "id": sample.get("id", i),
                    "image": str(image_path),
                    "reference": sample["messages"][1]["content"],
                    "prediction": prediction,
                    "time_sec": round(elapsed, 2),
                })

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{len(samples)}] {elapsed:.1f}s avg={total_time/(i+1):.1f}s")

                image.close()
                del image, inp, input_ids, attention_mask
                paddle.device.cuda.empty_cache()

            except Exception as e:
                print(f"  [{i+1}/{len(samples)}] ERROR: {e}")
                results.append({
                    "id": sample.get("id", i),
                    "image": str(image_path),
                    "reference": sample["messages"][1]["content"],
                    "prediction": f"[ERROR: {str(e)[:80]}]",
                    "time_sec": 0,
                })

        # Compute metrics
        predictions = [r["prediction"] for r in results]
        references = [r["reference"] for r in results]

        comp_f1 = compute_component_f1(predictions, references)
        token_recall = compute_token_recall(predictions, references)
        rep_rate, rep_count = compute_repetition_rate(predictions)
        ned = compute_ned(predictions, references)
        unique_preds = len(set(predictions))
        diversity = unique_preds / len(predictions) if predictions else 0.0

        metrics = {
            "checkpoint": ckpt_name,
            "n_samples": len(samples),
            "component_precision": round(comp_f1["precision"], 4),
            "component_recall": round(comp_f1["recall"], 4),
            "component_f1": round(comp_f1["f1"], 4),
            "token_recall": round(token_recall, 4),
            "repetition_rate": round(rep_rate, 4),
            "ned": round(ned, 4),
            "diversity": round(diversity, 4),
            "avg_time_sec": round(total_time / len(samples), 1),
            "total_time_min": round(total_time / 60, 1),
        }

        all_metrics[ckpt_name] = metrics

        # Save per-checkpoint results
        out_path = os.path.join(OUTPUT_DIR, f"results_{ckpt_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved: {out_path}")

        # Print metrics
        print(f"  CompF1={metrics['component_f1']:.4f}  "
              f"TokenRec={metrics['token_recall']:.4f}  "
              f"RepRate={metrics['repetition_rate']:.4f}  "
              f"NED={metrics['ned']:.4f}  "
              f"Diversity={metrics['diversity']:.4f}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: V12-Stage2 Evaluation Results")
    print(f"{'='*80}")
    print(f"{'CKPT':<10} {'CompF1':>8} {'Prec':>8} {'Rec':>8} {'TokenRec':>10} {'RepRate':>8} {'NED':>8} {'Div':>8}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    # Include baseline for comparison
    print(f"{'V10-S600':<10} {'0.2060':>8} {'---':>8} {'---':>8} {'0.1540':>10} {'0.1590':>8} {'0.8030':>8} {'0.9090':>8}")

    for ckpt_name in CHECKPOINTS:
        if ckpt_name in all_metrics:
            m = all_metrics[ckpt_name]
            print(f"{ckpt_name:<10} {m['component_f1']:>8.4f} {m['component_precision']:>8.4f} "
                  f"{m['component_recall']:>8.4f} {m['token_recall']:>10.4f} "
                  f"{m['repetition_rate']:>8.4f} {m['ned']:>8.4f} {m['diversity']:>8.4f}")

    print(f"\nSaved summary to: {summary_path}")
    print("Done!")

if __name__ == "__main__":
    main()
