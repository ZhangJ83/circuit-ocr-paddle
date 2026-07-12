#!/usr/bin/env python3
"""
V3 Fixed Evaluation Script (Phase 1)
=====================================
Fixes vs V2:
  1. Uses LoRAModel WRAPPER (same as training) instead of broken numpy merge
  2. Manual param setting via p.set_value() since set_state_dict returns None
  3. Same inference as training-time quick_inference
  4. All metrics: exact_match, component_f1, token_recall, repetition_rate, NED

Usage:
  # Base model:
  python eval_benchmark_v3.py --data_path ../ocr_vl_sft-test-easy50-pure.jsonl

  # Trained model (S600 checkpoint):
  python eval_benchmark_v3.py --data_path ../ocr_vl_sft-test-easy50-pure.jsonl \\
      --lora_checkpoint ../PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v10_fixed/lora_s600.pdparams
"""

import os, sys, json, time, re, argparse
from pathlib import Path
from collections import Counter

# ── Early patch: flex_checkpoint for Paddle 3.1.0 compatibility ──
from types import ModuleType
_dummy_fc = ModuleType('dummy_flex_checkpoint')
_dummy_fc.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy_fc)

# Prepend matching CUDA/cuDNN DLL paths
dll_paths = [
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin",
    r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\lib\site-packages\torch\lib",
    r"E:\080000software\080900_Miniconda\miniconda3\pkgs\cudatoolkit-11.3.1-h59b6b97_2\Library\bin"
]
existing_dll_paths = [p for p in dll_paths if os.path.exists(p)]
if existing_dll_paths:
    os.environ["PATH"] = ";".join(existing_dll_paths) + ";" + os.environ.get("PATH", "")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
local_hf_cache = "F:/hf_cache/hub"
local_paddle_cache = "F:/paddle_cache"
if os.path.exists(local_hf_cache):
    os.environ["HF_HOME"] = local_hf_cache
    os.environ["HF_HUB_CACHE"] = local_hf_cache
if os.path.exists(local_paddle_cache):
    os.environ["PADDLE_HOME"] = local_paddle_cache

# Monkey-patch huggingface_hub
try:
    import huggingface_hub.constants
    if os.path.exists(local_hf_cache):
        huggingface_hub.constants.HF_HOME = "F:/hf_cache"
        huggingface_hub.constants.HF_HUB_CACHE = "F:/hf_cache/hub"
except Exception:
    pass

import Levenshtein
from PIL import Image
from io import BytesIO

# ── Use the proven apply_paddle_patches from eval_benchmark.py ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from eval_benchmark import apply_paddle_patches

# ── Early paddle patches (before any paddleformers import) ──
import paddle
# Fix for Paddle 3.1.0: LongTensor removed
if not hasattr(paddle, 'LongTensor'):
    paddle.LongTensor = paddle.Tensor


# ==================== Metrics ====================
def extract_components(text):
    """Extract component refdes like R1, C2, U3, J4, D5, L6, Q7 from text."""
    pattern = r'\b((?:LED|[RCDLQUJYF])\d+)\b'
    return re.findall(pattern, text)


def compute_exact_match(predictions, references):
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions) if predictions else 0.0


def compute_component_f1(predictions, references):
    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        pred_comps = set(extract_components(pred))
        ref_comps = set(extract_components(ref))
        if not pred_comps and not ref_comps:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
        elif not pred_comps:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
        elif not ref_comps:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
        else:
            tp = len(pred_comps & ref_comps)
            prec = tp / len(pred_comps) if pred_comps else 0.0
            rec = tp / len(ref_comps) if ref_comps else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
    }


def compute_token_recall(predictions, references):
    recalls = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        if not ref_tokens:
            recalls.append(1.0)
        elif not pred_tokens:
            recalls.append(0.0)
        else:
            recalls.append(len(pred_tokens & ref_tokens) / len(ref_tokens))
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_repetition_rate(predictions, min_repeat=4):
    """Fraction of samples with >=min_repeat consecutive identical lines."""
    repeated = 0
    for pred in predictions:
        lines = pred.strip().split('\n')
        if len(lines) < min_repeat:
            continue
        max_run = 1
        current_run = 1
        for i in range(1, len(lines)):
            if lines[i].strip() == lines[i-1].strip():
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        if max_run >= min_repeat:
            repeated += 1
    return repeated / len(predictions) if predictions else 0.0, repeated


def compute_ned(predictions, references):
    distances = []
    for pred, ref in zip(predictions, references):
        d = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref), 1)
        distances.append(d / max_len)
    return sum(distances) / len(distances) if distances else 1.0


# ==================== Main Evaluation ====================
def evaluate(args):
    print("Applying Paddle compatibility patches...")
    apply_paddle_patches()

    print("Loading PaddleOCR-VL libraries...")
    import paddle
    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
    from paddleformers.peft import LoRAConfig, LoRAModel

    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    print(f"Setting Paddle device to: {device}")
    paddle.set_device(device)

    # Resolve paths
    MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = args.model_name_or_path or "PaddlePaddle/PaddleOCR-VL"

    processor_path = MODEL_PATH
    processor = AutoProcessor.from_pretrained(processor_path)

    # ── Load model with LoRA wrapper (if checkpoint provided) ──
    TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
    LORA_SCALE = 2.0  # alpha/r = 32/16
    REPETITION_PENALTY = 1.1

    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        convert_from_hf=True,
        load_checkpoint_format='naive',
        low_cpu_mem_usage=True,
        dtype="bfloat16"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    if args.lora_checkpoint:
        print(f"Loading LoRA checkpoint: {args.lora_checkpoint}")
        lora_file = args.lora_checkpoint
        if not os.path.exists(lora_file):
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_file}")

        # Apply LoRA wrapper
        lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
        model = LoRAModel(model, lc)

        # Load LoRA weights
        lora_state = paddle.load(lora_file)
        print(f"  Checkpoint has {len(lora_state)} keys")

        # Manually set each LoRA parameter (set_state_dict returns None)
        model_lora_params = {k: p for k, p in model.named_parameters() if 'lora_' in k}
        loaded = 0
        skipped = 0
        for ckpt_key, ckpt_value in lora_state.items():
            if ckpt_key in model_lora_params:
                p = model_lora_params[ckpt_key]
                ckpt_tensor = paddle.cast(ckpt_value, p.dtype)
                p.set_value(ckpt_tensor)
                loaded += 1
            else:
                skipped += 1
                if skipped <= 3:
                    print(f"  SKIP (no match): {ckpt_key}")

        print(f"  Loaded {loaded}/{len(lora_state)} LoRA params (skipped={skipped})")

    model.eval()
    paddle.device.cuda.empty_cache()

    # ── Load test data ──
    data_dir = Path(args.data_path).parent
    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]

    # Auto-generate output path
    if args.output_path is None:
        model_tag = "base" if not args.lora_checkpoint else "lora"
        if args.lora_checkpoint:
            ckpt_name = Path(args.lora_checkpoint).stem
            model_tag = f"lora_{ckpt_name}"
        data_name = Path(args.data_path).stem
        args.output_path = f"results_v3_{model_tag}_{data_name}.jsonl"

    print(f"\nEvaluating {len(samples)} test samples")
    print(f"  Model type: {'LoRA' if args.lora_checkpoint else 'Base'}")
    print(f"  repetition_penalty: {REPETITION_PENALTY}")
    print(f"  max_new_tokens: {args.max_length}")
    print(f"  Output: {args.output_path}")

    results = []
    for i, sample in enumerate(samples):
        start = time.time()
        query = sample["messages"][0]["content"]
        image_path = sample["images"][0]

        # Resolve image path
        img_resolved_path = Path(image_path)
        if not img_resolved_path.exists():
            alt_path = data_dir / image_path
            if alt_path.exists():
                img_resolved_path = alt_path
            else:
                alt_path2 = data_dir / image_path.lstrip("./")
                if alt_path2.exists():
                    img_resolved_path = alt_path2
                else:
                    alt_path2 = data_dir / img_resolved_path.name
                    if alt_path2.exists():
                        img_resolved_path = alt_path2

        image = None
        try:
            image = Image.open(img_resolved_path).convert("RGB")
            w, h = image.size
            max_dim = 384
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
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

            # ── Manual greedy decode with repetition_penalty (same as training) ──
            generated = []
            with paddle.no_grad():
                for _ in range(args.max_length):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw
                    )
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs.logits
                    next_token_logits = logits[:, -1, :]

                    # Apply repetition_penalty
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

            result = {
                "id": sample.get("id", i),
                "image": str(image_path),
                "reference": sample["messages"][1]["content"],
                "prediction": prediction,
                "time_sec": round(elapsed, 2),
            }
            results.append(result)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(samples)}] {elapsed:.1f}s "
                      f"pred={repr(prediction[:60])}  ref={repr(sample['messages'][1]['content'][:60])}")

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
            if image:
                try:
                    image.close()
                except Exception:
                    pass

    # ── Compute Metrics ──
    predictions = [r["prediction"] for r in results]
    references = [r["reference"] for r in results]

    exact_match_rate = compute_exact_match(predictions, references)
    comp_f1 = compute_component_f1(predictions, references)
    token_recall = compute_token_recall(predictions, references)
    rep_rate, rep_count = compute_repetition_rate(predictions)
    ned = compute_ned(predictions, references)

    # Diversity: unique predictions
    unique_preds = len(set(predictions))
    diversity = unique_preds / len(predictions) if predictions else 0.0

    metrics = {
        "n_samples": len(samples),
        "exact_match_rate": round(exact_match_rate, 4),
        "component_precision": round(comp_f1["precision"], 4),
        "component_recall": round(comp_f1["recall"], 4),
        "component_f1": round(comp_f1["f1"], 4),
        "token_recall": round(token_recall, 4),
        "repetition_rate": round(rep_rate, 4),
        "repetition_count": rep_count,
        "ned": round(ned, 4),
        "unique_predictions": unique_preds,
        "diversity": round(diversity, 4),
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.output_path}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save results
    output = {
        "config": {
            "model_path": MODEL_PATH,
            "lora_checkpoint": args.lora_checkpoint,
            "data_path": args.data_path,
            "repetition_penalty": REPETITION_PENALTY,
            "max_new_tokens": args.max_length,
        },
        "metrics": metrics,
        "results": results,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {args.output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Fixed Evaluation Script")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Base model path (auto-detected if omitted)")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to LoRA .pdparams checkpoint file")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test data JSONL")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max new tokens to generate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test samples")
    args = parser.parse_args()

    evaluate(args)
