#!/bin/bash
# V12: AMP mixed precision + small subset test + auto_growth allocator
LOGFILE=/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/training_run_v12.log
rm -f "$LOGFILE"

echo "=== Killing old python processes ===" | tee -a "$LOGFILE"
pkill -f train_lora_v2 2>/dev/null
sleep 2

echo "=== GPU State ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Training Start: $(date) ===" | tee -a "$LOGFILE"

export LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export FLAGS_allocator_strategy=auto_growth
cd /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset

# Quick test: only 100 training samples, 2 epochs, to see if it completes
head -100 ocr_vl_sft-train.jsonl > /tmp/train_tiny.jsonl
python3 scripts/train_lora_v2.py --rank 8 --alpha 16 --epochs 2 --grad_accum 1 --train_data /tmp/train_tiny.jsonl 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Training End: $(date) EXIT_CODE=$? ===" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Loss history ===" | tee -a "$LOGFILE"
python3 -c "import json; h=json.load(open('PaddleOCR-VL-LoRA-circuit-ocr/loss_history.json')); print(f'Steps: {len(h)}'); [print(f'  S{s[\"step\"]:4d} loss={s[\"loss\"]:.4f} lr={s[\"lr\"]:.2e}') for s in h[-5:]]" 2>/dev/null | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "=== GPU after ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"
