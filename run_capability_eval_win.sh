#!/bin/bash
# General Capability Evaluation — catastrophic forgetting check (Windows bash, conda python)
LOGFILE="g:/mimo_project/circuit_ocr/circuit-ocr-dataset/capability_eval.log"
rm -f "$LOGFILE"

echo "=== GPU State ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Capability Eval Start: $(date) ===" | tee -a "$LOGFILE"

cd "g:/mimo_project/circuit_ocr/circuit-ocr-dataset"

python scripts/eval_capability.py --num_images 5 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Capability Eval End: $(date) EXIT_CODE=$? ===" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== GPU after ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"
