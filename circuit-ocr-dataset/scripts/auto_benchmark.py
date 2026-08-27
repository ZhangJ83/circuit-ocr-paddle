#!/usr/bin/env python3
"""
AUTO BENCHMARK MASTER SCRIPT
============================
Creates tiered subsets, runs all 3 models on all tiers, aggregates results.
Designed to run non-stop until completion or 12:00 deadline.
"""
import json
import os
import sys
import time
import subprocess
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path("g:/mimo_project/circuit_ocr/circuit-ocr-dataset")
TEST_FILE = PROJECT_ROOT / "ocr_vl_sft-test.jsonl"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BENCHMARK_SCRIPT = SCRIPTS_DIR / "eval_benchmark.py"

# Python interpreters
PYTHON_PADDLE = sys.executable  # Current env (has Paddle)
PYTHON_PYTORCH = r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\python.exe"

# Models
PADDLE_MODEL = "PaddlePaddle/PaddleOCR-VL"
QWEN3_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# LoRA adapter - will be searched/downloaded
LORA_PATH = None  # To be determined

# Tier definitions
TIERS = {
    "easy50": 50,
    "easy100": 100,
    "easy200": 200,
    "full523": 523,
}

DEADLINE = "2026-06-19 12:00:00"

# ============================================================
# STEP 1: Create Tiered Subsets
# ============================================================
def create_tiered_subsets():
    """Create easy50, easy100, easy200 JSONL files sorted by complexity."""
    print("=" * 60)
    print("STEP 1: Creating tiered subsets...")
    print("=" * 60)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # Sort by label length (proxy for complexity)
    samples_with_len = []
    for i, s in enumerate(samples):
        lbl = s["messages"][1]["content"]
        samples_with_len.append((len(lbl), i, s))
    samples_with_len.sort(key=lambda x: x[0])

    subsets = {}
    for name, count in TIERS.items():
        subset_samples = [s for _, _, s in samples_with_len[:count]]
        out_path = PROJECT_ROOT / f"ocr_vl_sft-test-{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in subset_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        subsets[name] = str(out_path)

        # Stats
        lens = [len(s["messages"][1]["content"]) for s in subset_samples]
        print(f"  {name}: {len(subset_samples)} samples, label_len range [{min(lens)}, {max(lens)}], saved to {out_path.name}")

    return subsets


# ============================================================
# STEP 2: Run Benchmark
# ============================================================
def run_benchmark(model_type, model_name, data_path, output_path, python_exe,
                  lora_path=None, limit=None, resume=True, timeout=None):
    """Run a single benchmark and return success/failure."""
    cmd = [
        python_exe,
        str(BENCHMARK_SCRIPT),
        "--model_type", model_type,
        "--model_name_or_path", model_name,
        "--data_path", data_path,
        "--output_path", output_path,
        "--max_length", "1024",
        "--resume",
    ]
    if lora_path:
        cmd.extend(["--lora_path", lora_path])
    if limit:
        cmd.extend(["--limit", str(limit)])

    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"\n  CMD: {cmd_str}")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(cmd, env=env, timeout=timeout,
                               cwd=str(PROJECT_ROOT),
                               capture_output=False)  # stream output
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def run_benchmark_streaming(model_type, model_name, data_path, output_path, python_exe,
                            lora_path=None, limit=None, resume=True):
    """Run benchmark with real-time output streaming."""
    cmd = [
        python_exe,
        str(BENCHMARK_SCRIPT),
        "--model_type", model_type,
        "--model_name_or_path", model_name,
        "--data_path", data_path,
        "--output_path", output_path,
        "--max_length", "1024",
        "--resume",
    ]
    if lora_path:
        cmd.extend(["--lora_path", lora_path])
    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"\n  CMD: {' '.join(cmd)}")
    print(f"  Output: {output_path}")
    sys.stdout.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd, env=env, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace"
    )

    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"    {line}")
            sys.stdout.flush()

    process.wait()
    return process.returncode == 0


# ============================================================
# STEP 3: Compute Final Metrics
# ============================================================
def compute_ned_from_file(results_path):
    """Compute Avg NED from a results JSONL file."""
    try:
        import Levenshtein
    except ImportError:
        import sys
        sys.path.insert(0, str(SCRIPTS_DIR))
    import Levenshtein

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        return None, 0

    total_ned = 0
    for r in results:
        pred = r.get("prediction", "")
        ref = r.get("label", "")
        dist = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            total_ned += dist / max_len

    return total_ned / len(results), len(results)


# ============================================================
# STEP 4: Aggregate Report
# ============================================================
def aggregate_report(all_runs):
    """Generate final comparison table."""
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK REPORT")
    print("=" * 70)
    print(f"{'Model':<25} {'Tier':<12} {'Samples':<10} {'Avg NED':<12} {'Status':<10}")
    print("-" * 70)

    summary = {}
    for run in all_runs:
        ned, count = compute_ned_from_file(run["output"])
        ned_str = f"{ned:.4f}" if ned is not None else "FAILED"
        status = "OK" if count >= run["expected"] else f"INCOMPLETE({count}/{run['expected']})"
        print(f"{run['model']:<25} {run['tier']:<12} {count:<10} {ned_str:<12} {status:<10}")

        key = run["model"]
        if key not in summary:
            summary[key] = {}
        summary[key][run["tier"]] = ned

    print("\n--- SUMMARY MATRIX ---")
    tiers_ordered = ["easy50", "easy100", "easy200", "full523"]
    models_ordered = list(summary.keys())
    header = f"{'Model':<25} " + " ".join(f"{t:<12}" for t in tiers_ordered)
    print(header)
    print("-" * (25 + 12 * len(tiers_ordered)))
    for model in models_ordered:
        vals = []
        for t in tiers_ordered:
            v = summary[model].get(t)
            vals.append(f"{v:.4f}" if v is not None else "N/A")
        print(f"{model:<25} " + " ".join(f"{v:<12}" for v in vals))

    # Save report
    report_path = PROJECT_ROOT / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write("FINAL BENCHMARK REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(header + "\n")
        f.write("-" * (25 + 12 * len(tiers_ordered)) + "\n")
        for model in models_ordered:
            vals = []
            for t in tiers_ordered:
                v = summary[model].get(t)
                vals.append(f"{v:.4f}" if v is not None else "N/A")
            f.write(f"{model:<25} " + " ".join(f"{v:<12}" for v in vals) + "\n")
    print(f"\nReport saved to: {report_path}")

    return summary


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
def main():
    print("=" * 60)
    print("AUTO BENCHMARK ORCHESTRATOR")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Deadline: {DEADLINE}")
    print("=" * 60)
    sys.stdout.flush()

    # Create tiered subsets
    subsets = create_tiered_subsets()

    all_runs = []

    # Define the benchmark plan
    benchmarks = [
        # (model_type, model_name, python_exe, lora_path, label)
        ("paddleocr-vl", PADDLE_MODEL, PYTHON_PADDLE, None, "PaddleOCR-VL-0.9B"),
        ("qwen3-vl", QWEN3_MODEL, PYTHON_PYTORCH, None, "Qwen3-VL-8B-Base"),
    ]

    # Add LoRA if available
    if LORA_PATH:
        benchmarks.append(("qwen3-vl-lora", QWEN3_MODEL, PYTHON_PYTORCH, LORA_PATH, "Qwen3-VL-8B-LoRA"))

    tiers_ordered = ["easy50", "easy100", "easy200", "full523"]

    for model_type, model_name, python_exe, lora_path, label in benchmarks:
        print(f"\n{'='*60}")
        print(f"MODEL: {label} ({model_type})")
        print(f"Python: {python_exe}")
        print(f"{'='*60}")
        sys.stdout.flush()

        for tier in tiers_ordered:
            data_path = subsets[tier]
            output_path = str(PROJECT_ROOT / f"results_{model_type}_{tier}.jsonl")
            expected = TIERS[tier]

            # Check deadline
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] Starting {label} / {tier} ({expected} samples)")
            sys.stdout.flush()

            success = run_benchmark_streaming(
                model_type, model_name, data_path, str(output_path), python_exe,
                lora_path=lora_path, resume=True
            )

            all_runs.append({
                "model": label,
                "tier": tier,
                "output": output_path,
                "expected": expected,
                "success": success,
            })

            # Check deadline again
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Completed {label} / {tier}: {'OK' if success else 'FAILED'}")
            sys.stdout.flush()

    # Final aggregation
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Aggregating results...")
    aggregate_report(all_runs)

    print(f"\nFinished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
