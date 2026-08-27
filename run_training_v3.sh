#!/bin/bash
# Kill old training, then run PaddleOCR-VL LoRA training (FIXED: full sequence + traceback)
LOGFILE=/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/training_run_v3.log
rm -f "$LOGFILE"

echo "=== Killing old python processes ===" | tee -a "$LOGFILE"
pkill -f train_lora_v2 2>/dev/null
sleep 2

echo "=== GPU State ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Training Start: $(date) ===" | tee -a "$LOGFILE"

export LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset

python3 scripts/train_lora_v2.py --rank 8 --alpha 16 --epochs 2 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Training End: $(date) EXIT_CODE=$? ===" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Internal training.log ===" | tee -a "$LOGFILE"
cat PaddleOCR-VL-LoRA-circuit-ocr/training.log 2>/dev/null | tail -30 | tee -a "$LOGFILE"
