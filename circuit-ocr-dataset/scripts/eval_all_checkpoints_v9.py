"""
Evaluate all v9-pure training checkpoints + final model on easy50.
Finds the best checkpoint by Avg. NED on the 6 pure validation samples.
"""
import os, sys, json, subprocess, shutil
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent
CKPT_DIR = DATASET_DIR / "PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v9_pure"
FINAL_MODEL = DATASET_DIR / "PaddleOCR-VL-LoRA-circuit-ocr/lora_v9_pure_final_fp16.pdparams"
EVAL_DIR = DATASET_DIR / "PaddleOCR-VL-LoRA-circuit-ocr/lora_v9_eval"
MODEL_PATH = "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"
PYTHON = r"E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe"
DATA_PATH = str(DATASET_DIR / "ocr_vl_sft-test-easy50.jsonl")
EVAL_SCRIPT = str(DATASET_DIR / "scripts/eval_benchmark.py")

EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Gather all checkpoints + final model
models_to_eval = {}

# Checkpoints (steps s200 to s1000)
for ckpt in sorted(CKPT_DIR.glob("lora_s*.pdparams")):
    step = int(ckpt.stem.replace("lora_s", ""))
    models_to_eval[f"s{step}"] = str(ckpt)

# Final model
models_to_eval["final"] = str(FINAL_MODEL)

print(f"Evaluating {len(models_to_eval)} models on easy50 (limit 10)...")
print(f"Models: {list(models_to_eval.keys())}")

results = {}

for name, ckpt_path in models_to_eval.items():
    if not Path(ckpt_path).exists():
        print(f"  SKIP {name}: file not found ({ckpt_path})")
        continue

    output_path = str(DATASET_DIR / f"results_v9_{name}_limit10.jsonl")

    # Clean up old output file to avoid reading stale cache
    if Path(output_path).exists():
        os.remove(output_path)

    # Copy to eval dir
    target = EVAL_DIR / "final_model_light.pdparams"
    shutil.copy2(ckpt_path, target)

    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output: {output_path}")

    # Note: we use standard evaluation using model.generate()
    cmd = [
        PYTHON, EVAL_SCRIPT,
        "--model_type", "paddleocr-vl",
        "--model_name_or_path", MODEL_PATH,
        "--paddle_lora_dir", str(EVAL_DIR),
        "--data_path", DATA_PATH,
        "--output_path", output_path,
        "--max_length", "128",
        "--limit", "10",
    ]

    try:
        # Run evaluation
        result = subprocess.run(cmd, cwd=str(DATASET_DIR),
                               capture_output=True, text=True, timeout=900)
        
        # Calculate NED from output file (filtering only pure PNG samples)
        if Path(output_path).exists():
            with open(output_path, encoding='utf-8') as f:
                data = [json.loads(l) for l in f if l.strip()]
            
            # Load ordered image paths of first 10 samples
            with open(str(DATASET_DIR / "ocr_vl_sft-test-easy50.jsonl"), encoding="utf-8") as f_orig:
                easy50_10_orig = [json.loads(l)["images"][0] for l in f_orig if l.strip()][:10]
                
            import Levenshtein
            neds = []
            for idx, d in enumerate(data):
                img = easy50_10_orig[idx]
                if not img.lower().endswith(".png"):
                    continue # Skip textbook JPGs
                pred = d.get("prediction", "")
                label = d.get("label", d.get("messages", [{},{}])[1].get("content", ""))
                dist = Levenshtein.distance(pred, label)
                max_len = max(len(pred), len(label))
                ned = dist / max_len if max_len > 0 else 0.0
                neds.append(ned)
                
            if neds:
                ned = sum(neds) / len(neds)
                print(f"  >> Avg. NED calculated on 6 pure validation samples: {ned:.4f}")
                results[name] = {"ned": ned, "output": output_path}
            else:
                print(f"  >> No predictions found in output file")
                results[name] = {"ned": None, "output": output_path}
        else:
            print(f"  >> Output file {output_path} not found! Output log:")
            print(result.stdout)
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 15min")
        results[name] = {"ned": None, "output": output_path}
    except Exception as e:
        print(f"  ERROR: {e}")
        results[name] = {"ned": None, "output": output_path}

# Summary
print("\n" + "="*60)
print("BEST CHECKPOINT SELECTION (V9)")
print("="*60)
print("Results:")
valid = {k: v for k, v in results.items() if v["ned"] is not None}
if valid:
    best = min(valid.items(), key=lambda x: x[1]["ned"])
    print(f"BEST: {best[0]} with NED={best[1]['ned']:.4f}")

    print("\nAll results (sorted by NED):")
    for name, info in sorted(valid.items(), key=lambda x: x[1]["ned"]):
        print(f"  {name:12s} NED={info['ned']:.4f}")
else:
    print("No valid results! Check eval output.")

# Save results
best_name = best[0] if valid else None
with open(str(DATASET_DIR / "checkpoint_eval_results_v9_limit10.json"), "w") as f:
    json.dump({"results": results, "best": best_name}, f, indent=2)
print(f"\nResults saved to checkpoint_eval_results_v9_limit10.json")
