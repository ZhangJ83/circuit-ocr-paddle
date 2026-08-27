#!/bin/bash
# Light training: filtered short samples + 64px + AMP O2
LOGFILE=/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/training_light.log2
rm -f "$LOGFILE"

echo "=== Killing old python processes ===" | tee -a "$LOGFILE"
pkill -f train_lora 2>/dev/null
sleep 2

echo "=== GPU State ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Training Start: $(date) ===" | tee -a "$LOGFILE"

export LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export FLAGS_allocator_strategy=auto_growth
cd /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset

python3 scripts/train_lora_light.py --rank 8 --alpha 16 --epochs 2 --max_label_len 100 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Training End: $(date) EXIT_CODE=$? ===" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Loss history ===" | tee -a "$LOGFILE"
python3 -c "import json; h=json.load(open('PaddleOCR-VL-LoRA-circuit-ocr/loss_history_light.json')); print(f'Steps: {len(h)}'); [print(f'  S{s[\"step\"]:4d} loss={s[\"loss\"]:.4f} lr={s[\"lr\"]:.2e}') for s in h[-5:]]" 2>/dev/null | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "=== GPU after ===" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE"
