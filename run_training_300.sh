#!/bin/bash
# Training: max_label_len=300 (1017 samples, 42% of data)
LOGFILE=/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/training_300.log
rm -f "$LOGFILE"

echo "=== GPU State ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Training Start: $(date) ===" | tee -a "$LOGFILE"

export LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export FLAGS_allocator_strategy=auto_growth
cd /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset

python3 scripts/train_lora_light.py --rank 8 --alpha 16 --epochs 2 --max_label_len 300 --grad_accum 4 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Training End: $(date) EXIT_CODE=$? ===" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Loss history ===" | tee -a "$LOGFILE"
python3 -c "import json; h=json.load(open('PaddleOCR-VL-LoRA-circuit-ocr/loss_history_light.json')); print(f'Steps: {len(h)}'); [print(f'  S{s[\"step\"]:4d} loss={s[\"loss\"]:.4f} lr={s[\"lr\"]:.2e}') for s in h[-5:]]" 2>/dev/null | tee -a "$LOGFILE"
