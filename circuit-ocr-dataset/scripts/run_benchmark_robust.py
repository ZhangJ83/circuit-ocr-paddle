"""Robust benchmark runner: handles Paddle crashes by restarting after each sample.

Strategy: Run eval_benchmark.py with --limit 1, then --resume for each sample.
This way each crash only loses the current sample, and we can pick up from where we left off.
"""
import subprocess, sys, os, time, json
from pathlib import Path

PYTHON = r"E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe"
SCRIPT = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset\scripts\eval_benchmark.py"
DATA_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"

# Config: which tiers to run
TIERS = [
    ("ocr_vl_sft-test-easy50.jsonl", "results_paddleocr-vl_easy50.jsonl", 50),
    ("ocr_vl_sft-test-easy100.jsonl", "results_paddleocr-vl_easy100.jsonl", 100),
    ("ocr_vl_sft-test-easy200.jsonl", "results_paddleocr-vl_easy200.jsonl", 200),
    ("ocr_vl_sft-test.jsonl", "results_paddleocr-vl_full523.jsonl", 523),
]

def count_results(output_path):
    """Count completed results in output file."""
    p = Path(output_path)
    if not p.exists():
        return 0
    count = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("prediction") != "":
                        count += 1
                except:
                    pass
    return count

def run_tier(data_path, output_path, total):
    """Run a single tier with crash recovery."""
    data_file = f"{DATA_DIR}/{data_path}"
    out_file = f"{DATA_DIR}/{output_path}"
    print(f"\n{'='*60}")
    print(f"TIER: {data_path} ({total} samples)")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    # First run: try without resume
    cmd = [
        PYTHON, SCRIPT,
        "--model_type", "paddleocr-vl",
        "--model_name_or_path", MODEL_PATH,
        "--data_path", data_file,
        "--output_path", out_file,
        "--max_length", "512",
        "--resume",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=DATA_DIR, capture_output=True, text=True, timeout=3600)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.stderr:
        stderr_short = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        print("STDERR:", stderr_short)

    done = count_results(out_file)
    print(f"Completed: {done}/{total}")

    # If not all done, retry with resume
    retries = 0
    while done < total and retries < 50:
        retries += 1
        print(f"  Retry {retries}: {done}/{total} done, resuming...")
        result = subprocess.run(cmd, cwd=DATA_DIR, capture_output=True, text=True, timeout=3600)
        # Show last OK/FAIL line
        for line in result.stdout.split("\n"):
            if "OK " in line or "FAIL " in line:
                print(f"    {line.strip()}")
        new_done = count_results(out_file)
        if new_done == done:
            print(f"  No progress, sleeping 30s...")
            time.sleep(30)
        else:
            print(f"  Progress: {done} -> {new_done}")
            retries = 0  # Reset retry counter on progress
        done = new_done

    print(f"\nTIER DONE: {done}/{total} completed")
    return done

def main():
    for data_path, output_path, total in TIERS:
        done = run_tier(data_path, output_path, total)
        if done < total:
            print(f"WARNING: Only {done}/{total} completed for {data_path}")

    # Compute overall NED
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for data_path, output_path, total in TIERS:
        out_file = f"{DATA_DIR}/{output_path}"
        done = count_results(out_file)
        print(f"  {output_path}: {done}/{total}")

if __name__ == "__main__":
    main()
