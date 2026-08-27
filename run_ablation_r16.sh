#!/bin/bash
# Rank Ablation: r=16, alpha=32, 500-char (1585 samples)
LOGFILE=/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/ablation_r16.log
rm -f "$LOGFILE"

echo "=== GPU State ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Ablation r=16 Start: $(date) ===" | tee -a "$LOGFILE"

export LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export FLAGS_allocator_strategy=auto_growth
cd /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset

python3 scripts/train_lora_light.py --rank 16 --alpha 32 --epochs 2 --max_label_len 500 --grad_accum 4 --output_dir PaddleOCR-VL-LoRA-ablation-r16 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Ablation r=16 End: $(date) EXIT_CODE=$? ===" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "=== GPU after ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"
