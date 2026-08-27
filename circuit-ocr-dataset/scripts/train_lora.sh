#!/bin/bash
# ============================================================================
# PaddleOCR-VL LoRA Fine-Tuning Launch Script — RTX 4060 8GB
# ============================================================================
# Prerequisites:
#   1. conda environment with CUDA 11.8 + cuDNN 8.9
#   2. PaddlePaddle 2.6.x GPU + PaddleFormers 1.1.x
#   3. LD_LIBRARY_PATH must include CUDA libs for WSL
#
# Usage:
#   bash scripts/train_lora.sh
#
# Before running:
#   1. Review configs/paddleocr-vl_lora_8gb.yaml
#   2. Run prefilter if needed: python scripts/prefilter_training_data.py
#   3. If prefiltered, update train_dataset_path in YAML to the filtered file
# ============================================================================

set -e

# ---- Environment Setup ----
export HF_HOME=/mnt/f/hf_cache/hub
export PADDLE_HOME=/mnt/f/paddle_cache
export HF_HUB_CACHE=/mnt/f/hf_cache/hub
export KMP_DUPLICATE_LIB_OK=TRUE
export LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/configs/paddleocr-vl_lora_8gb.yaml"
OUTPUT_DIR="${PROJECT_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"

cd "$PROJECT_DIR"

echo "============================================"
echo "  PaddleOCR-VL LoRA Fine-Tuning"
echo "  GPU: RTX 4060 8GB"
echo "============================================"
echo "Config:  ${CONFIG_FILE}"
echo "Output:  ${OUTPUT_DIR}"
echo "Project: ${PROJECT_DIR}"
echo ""

# ---- Pre-flight Checks ----
echo "[1/4] Checking GPU..."
python3 -c "
import paddle
paddle.set_device('gpu')
print(f'  GPU available: {paddle.device.get_device()}')
print(f'  Device count:  {paddle.device.cuda.device_count()}')
print(f'  Device name:   {paddle.device.cuda.get_device_name(0)}')
print(f'  Paddle version: {paddle.__version__}')
" || { echo "ERROR: GPU not available!"; exit 1; }

echo ""
echo "[2/4] Checking PaddleFormers..."
python3 -c "
import paddleformers
print(f'  PaddleFormers version: {paddleformers.__version__}')
" || { echo "ERROR: PaddleFormers not installed!"; exit 1; }

echo ""
echo "[3/4] Checking training data..."
python3 -c "
import json, os
data_file = 'ocr_vl_sft-train.jsonl'
count = 0
with open(data_file, 'r') as f:
    for line in f:
        if line.strip(): count += 1
print(f'  Training samples: {count}')
assert count > 0, 'Training data is empty!'
" || { echo "ERROR: Training data check failed!"; exit 1; }

echo ""
echo "[4/4] Checking disk space..."
df -h . | tail -1

echo ""
echo "============================================"
echo "  Starting Training..."
echo "  Estimated time: ~6-12 hours (RTX 4060)"
echo "  Monitor with:   visualdl --logdir ${OUTPUT_DIR}/visualdl_logs/"
echo "============================================"
echo ""

# ---- Launch Training ----
# pre_alloc_memory=6.0 (GB) — reserves 6GB, leaves ~2GB for overhead
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli train "$CONFIG_FILE" \
    model_name_or_path=PaddlePaddle/PaddleOCR-VL \
    train_dataset_path=./ocr_vl_sft-train.jsonl \
    eval_dataset_path=./ocr_vl_sft-test.jsonl \
    pre_alloc_memory=6.0

# ---- Post-Training ----
echo ""
echo "============================================"
echo "  Training Complete!"
echo "  Model saved to: ${OUTPUT_DIR}"
echo ""
echo "  To merge LoRA weights:"
echo "    paddleformers-cli export configs/paddleocr-vl_lora_export.yaml \\"
echo "        model_name_or_path=PaddlePaddle/PaddleOCR-VL \\"
echo "        output_dir=${OUTPUT_DIR}"
echo ""
echo "  Merged model will be at: ${OUTPUT_DIR}/export/"
echo "============================================"
