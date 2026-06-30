"""
Evaluate all training checkpoints + final model on easy50.
Finds the best checkpoint by Avg. NED.
Uses manual decode to avoid Paddle 3.0b2 segfault.
"""
import os, sys, json, subprocess, shutil
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent
CKPT_DIR = DATASET_DIR / "PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v2"
FINAL_MODEL = DATASET_DIR / "PaddleOCR-VL-LoRA-circuit-ocr/lora_projector_v2_final_fp16.pdparams"
EVAL_DIR = DATASET_DIR / "PaddleOCR-VL-LoRA-circuit-ocr/lora_v2_eval"
MODEL_PATH = "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"
PYTHON = r"E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe"
DATA_PATH = str(DATASET_DIR / "ocr_vl_sft-test-easy50.jsonl")
EVAL_SCRIPT = str(DATASET_DIR / "scripts/eval_benchmark.py")

EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Gather all checkpoints + final model
models_to_eval = {}

# Checkpoints
for ckpt in sorted(CKPT_DIR.glob("lora_s*.pdparams")):
    step = int(ckpt.stem.replace("lora_s", ""))
    models_to_eval[f"s{step}"] = str(ckpt)

# Final model
models_to_eval["final"] = str(FINAL_MODEL)

print(f"Evaluating {len(models_to_eval)} models on easy50...")
print(f"Models: {list(models_to_eval.keys())}")

results = {}

for name, ckpt_path in models_to_eval.items():
    if not Path(ckpt_path).exists():
        print(f"  SKIP {name}: file not found ({ckpt_path})")
        continue

    output_path = str(DATASET_DIR / f"results_v2_{name}_easy50.jsonl")

    # Copy to eval dir
    target = EVAL_DIR / "final_model_light.pdparams"
    shutil.copy2(ckpt_path, target)

    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output: {output_path}")

    cmd = [
        PYTHON, EVAL_SCRIPT,
        "--model_type", "paddleocr-vl",
        "--model_name_or_path", MODEL_PATH,
        "--paddle_lora_dir", str(EVAL_DIR),
        "--data_path", DATA_PATH,
        "--output_path", output_path,
        "--max_length", "30",
        "--manual_decode",
        "--resume",
    ]

    try:
        result = subprocess.run(cmd, cwd=str(DATASET_DIR),
                              capture_output=True, text=True, timeout=600)
        # Extract NED from output
        output_lines = result.stdout.split("\n") + result.stderr.split("\n")
        ned = None
        for line in output_lines:
            if "Avg. NED" in line or "Average NED" in line or "avg_ned" in line.lower():
                print(f"  {line.strip()}")
                # Try to parse the NED value
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    ned = float(match.group())

        # Also parse from output file
        if ned is None and Path(output_path).exists():
            with open(output_path) as f:
                data = [json.loads(l) for l in f if l.strip()]
            neds = [d.get("ned", 1.0) for d in data if "ned" in d]
            if neds:
                ned = sum(neds) / len(neds)

        if ned is not None:
            print(f"  >> NED: {ned:.4f}")
            results[name] = {"ned": ned, "output": output_path}
        else:
            print(f"  >> NED not found in output")
            results[name] = {"ned": None, "output": output_path}

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 10min")
        results[name] = {"ned": None, "output": output_path}
    except Exception as e:
        print(f"  ERROR: {e}")
        results[name] = {"ned": None, "output": output_path}

# Summary
print("\n" + "="*60)
print("BEST CHECKPOINT SELECTION")
print("="*60)
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
with open(str(DATASET_DIR / "checkpoint_eval_results.json"), "w") as f:
    json.dump({"results": results, "best": best[0] if valid else None}, f, indent=2)
print(f"\nResults saved to checkpoint_eval_results.json")
