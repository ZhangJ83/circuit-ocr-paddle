#!/bin/bash
# AUTO-BENCHMARK NEVER-STOP SCRIPT
# Runs all 3 models × 4 tiers, never stops until 12:00 deadline.
# Launched from: g:/mimo_project/circuit_ocr/circuit-ocr-dataset/

set -e
cd "g:/mimo_project/circuit_ocr/circuit-ocr-dataset"

PY_PADDLE="python"
PY_QWEN="E:/080000software/080900_Miniconda/miniconda3/Library/envs/gpu-pytorch/python.exe"
DEADLINE="2026-06-19 12:00:00"
LOG_DIR="benchmark_logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ===== STEP 1: Create tiered subsets =====
log "Creating tiered subsets..."
$PY_PADDLE -c "
import json
with open('ocr_vl_sft-test.jsonl', 'r', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f if line.strip()]
info = [(len(s['messages'][1]['content']), i, s) for i, s in enumerate(samples)]
info.sort(key=lambda x: x[0])
for name, count in [('easy50',50),('easy100',100),('easy200',200)]:
    subset = [s for _,_,s in info[:count]]
    with open(f'ocr_vl_sft-test-{name}.jsonl', 'w', encoding='utf-8') as out:
        for s in subset:
            out.write(json.dumps(s, ensure_ascii=False) + '\n')
    lens = [len(s['messages'][1]['content']) for s in subset]
    print(f'{name}: {len(subset)} samples, label_len [{min(lens)}, {max(lens)}]')
print('All subsets ready.')
"

# ===== STEP 2: Run PaddleOCR-VL (current env, CPU float32) =====
log "=== PADDLEOCR-VL BENCHMARKS ==="
for TIER in easy50 easy100 easy200; do
    DATA="ocr_vl_sft-test-${TIER}.jsonl"
    OUT="results_paddleocr-vl_${TIER}.jsonl"
    LOGFILE="${LOG_DIR}/paddle_${TIER}.log"
    if [ -f "$OUT" ]; then
        DONE=$(wc -l < "$OUT")
        EXPECTED=$(echo "$TIER" | sed 's/easy//')
        if [ "$DONE" -ge "$EXPECTED" ]; then
            log "Paddle $TIER already complete ($DONE/$EXPECTED), skipping"
            continue
        fi
        log "Paddle $TIER resuming ($DONE/$EXPECTED)"
        RESUME="--resume"
    else
        log "Paddle $TIER starting fresh"
        RESUME=""
    fi
    log "CMD: paddle $TIER -> $OUT"
    $PY_PADDLE scripts/eval_benchmark.py \
        --model_type paddleocr-vl \
        --model_name_or_path PaddlePaddle/PaddleOCR-VL \
        --data_path "$DATA" \
        --output_path "$OUT" \
        --max_length 1024 \
        $RESUME \
        2>&1 | tee "$LOGFILE"
    log "Paddle $TIER done: $(wc -l < "$OUT") samples"
done

# ===== STEP 3: Run Qwen3-VL-8B Base =====
log "=== QWEN3-VL-8B BASE BENCHMARKS ==="
for TIER in easy50 easy100 easy200 full523; do
    DATA="ocr_vl_sft-test-${TIER}.jsonl"
    if [ "$TIER" = "full523" ]; then
        DATA="ocr_vl_sft-test.jsonl"
    fi
    OUT="results_qwen3-vl_${TIER}.jsonl"
    LOGFILE="${LOG_DIR}/qwen3_base_${TIER}.log"
    if [ -f "$OUT" ]; then
        DONE=$(wc -l < "$OUT")
        EXPECTED=$(echo "$TIER" | sed 's/easy//' | sed 's/full//')
        if [ "$TIER" = "full523" ]; then EXPECTED=523; fi
        if [ "$DONE" -ge "$EXPECTED" ]; then
            log "Qwen3-base $TIER already complete ($DONE/$EXPECTED), skipping"
            continue
        fi
        log "Qwen3-base $TIER resuming ($DONE/$EXPECTED)"
        RESUME="--resume"
    else
        log "Qwen3-base $TIER starting fresh"
        RESUME=""
    fi
    log "CMD: qwen3-base $TIER -> $OUT"
    "$PY_QWEN" scripts/eval_benchmark.py \
        --model_type qwen3-vl \
        --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
        --data_path "$DATA" \
        --output_path "$OUT" \
        --max_length 1024 \
        $RESUME \
        2>&1 | tee "$LOGFILE"
    log "Qwen3-base $TIER done: $(wc -l < "$OUT") samples"
done

# ===== STEP 4: Qwen3-VL-LoRA (if available) =====
log "=== QWEN3-VL-LORA BENCHMARKS ==="
LORA_PATH="F:/hf_cache/hub/models--Qwen--Qwen3-VL-8B-circuit-ocr-lora"
if [ -d "$LORA_PATH" ]; then
    log "LoRA adapter found at $LORA_PATH"
    for TIER in easy50 easy100 easy200 full523; do
        DATA="ocr_vl_sft-test-${TIER}.jsonl"
        if [ "$TIER" = "full523" ]; then
            DATA="ocr_vl_sft-test.jsonl"
        fi
        OUT="results_qwen3-vl-lora_${TIER}.jsonl"
        LOGFILE="${LOG_DIR}/qwen3_lora_${TIER}.log"
        if [ -f "$OUT" ]; then
            DONE=$(wc -l < "$OUT")
            EXPECTED=$(echo "$TIER" | sed 's/easy//' | sed 's/full//')
            if [ "$TIER" = "full523" ]; then EXPECTED=523; fi
            if [ "$DONE" -ge "$EXPECTED" ]; then
                log "Qwen3-LoRA $TIER already complete, skipping"
                continue
            fi
            RESUME="--resume"
        else
            RESUME=""
        fi
        log "CMD: qwen3-lora $TIER -> $OUT"
        "$PY_QWEN" scripts/eval_benchmark.py \
            --model_type qwen3-vl-lora \
            --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
            --lora_path "$LORA_PATH" \
            --data_path "$DATA" \
            --output_path "$OUT" \
            --max_length 1024 \
            $RESUME \
            2>&1 | tee "$LOGFILE"
    done
else
    log "No LoRA adapter found, skipping Qwen3-VL-LoRA benchmarks"
fi

# ===== STEP 5: Aggregate Report =====
log "=== AGGREGATING RESULTS ==="
$PY_PADDLE -c "
import json, glob, os
try:
    import Levenshtein
except:
    import sys
    sys.path.insert(0, 'scripts')
    import Levenshtein

files = glob.glob('results_*.jsonl')
print(f'Found {len(files)} result files')

report = {}
for f in sorted(files):
    with open(f, 'r', encoding='utf-8') as fh:
        results = [json.loads(l) for l in fh if l.strip()]
    if not results:
        print(f'{f}: 0 results')
        continue
    total_ned = 0
    for r in results:
        pred = r.get('prediction', '')
        ref = r.get('label', '')
        dist = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref))
        total_ned += dist / max_len if max_len > 0 else 0
    avg_ned = total_ned / len(results)
    report[f] = {'count': len(results), 'avg_ned': avg_ned}
    print(f'{f}: {len(results)} samples, Avg NED = {avg_ned:.4f}')

# Save JSON report
with open('benchmark_report.json', 'w') as fh:
    json.dump(report, fh, indent=2)
print('Report saved to benchmark_report.json')

# Print summary table
print()
print('=' * 70)
print('FINAL BENCHMARK SUMMARY')
print('=' * 70)
for f, r in sorted(report.items()):
    print(f'  {f:<50}  N={r[\"count\"]:>4}  NED={r[\"avg_ned\"]:.4f}')
print('=' * 70)
"

log "=== ALL DONE at $(date) ==="
