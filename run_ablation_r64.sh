#!/bin/bash
# Rank Ablation: r=64, alpha=128, 500-char (1585 samples)
LOGFILE=/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/ablation_r64.log
rm -f "$LOGFILE"

echo "=== GPU State ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Ablation r=64 Start: $(date) ===" | tee -a "$LOGFILE"

export LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export FLAGS_allocator_strategy=auto_growth
cd /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset

python3 scripts/train_lora_light.py --rank 64 --alpha 128 --epochs 2 --max_label_len 500 --grad_accum 4 --output_dir PaddleOCR-VL-LoRA-ablation-r64 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Ablation r=64 End: $(date) EXIT_CODE=$? ===" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "=== GPU after ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"
