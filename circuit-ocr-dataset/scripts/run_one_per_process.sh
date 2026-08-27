#!/bin/bash
# One-process-per-sample benchmark runner for PaddleOCR-VL
# Each sample gets a fresh Python process to avoid Paddle GPU crash (exit 127)
set -e

PYTHON="E:/080000software/080900_Miniconda/miniconda3/Library/envs/gpu-pytorch/python.exe"
SCRIPT="scripts/eval_benchmark.py"
DATA_DIR="g:/mimo_project/circuit_ocr/circuit-ocr-dataset"

TIER="${1:-easy50}"
DATA="ocr_vl_sft-test-${TIER}.jsonl"
OUTPUT="results_paddleocr-vl_${TIER}.jsonl"

cd "$DATA_DIR"

# Count total samples in dataset
TOTAL=$(wc -l < "$DATA" | tr -d ' ')
echo "=== PaddleOCR-VL ${TIER}: ${TOTAL} samples ==="

SUCCESS=0
FAILED=0
START_TIME=$(date +%s)

for ((i=1; i<=TOTAL; i++)); do
    # Check if already processed
    if [ -f "$OUTPUT" ]; then
        DONE_COUNT=$(wc -l < "$OUTPUT" | tr -d ' ')
    else
        DONE_COUNT=0
    fi

    if [ "$DONE_COUNT" -ge "$i" ]; then
        continue
    fi

    echo -n "[${i}/${TOTAL}] "
    START_SAMPLE=$(date +%s)

    # Run single sample with --resume
    if $PYTHON "$SCRIPT" \
        --model_type paddleocr-vl \
        --model_name_or_path PaddlePaddle/PaddleOCR-VL \
        --data_path "$DATA" \
        --output_path "$OUTPUT" \
        --max_length 1024 \
        --resume \
        --limit 1 \
        2>&1 | grep -E "^\[|FAIL|Error" || true
    then
        # Check if a new line was added
        NEW_COUNT=$(wc -l < "$OUTPUT" | tr -d ' ')
        if [ "$NEW_COUNT" -gt "$DONE_COUNT" ]; then
            ELAPSED=$(( $(date +%s) - START_SAMPLE ))
            echo "  -> OK (${ELAPSED}s) [${NEW_COUNT}/${TOTAL}]"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  -> WARN: no output, retrying..."
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  -> FAIL (exit code: $?)"
        FAILED=$((FAILED + 1))
    fi

    # Brief pause to let GPU stabilize
    sleep 1
done

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
echo ""
echo "=== Done: ${SUCCESS} OK, ${FAILED} failed, ${TOTAL_TIME}s total ==="
echo "Results: ${DATA_DIR}/${OUTPUT}"
