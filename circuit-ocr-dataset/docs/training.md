# Training Guide

## Prerequisites

1. **GPU**: NVIDIA GPU with ≥ 8GB VRAM (RTX 3070 or better recommended)
2. **CUDA**: 11.7+ with cuDNN 8.0+
3. **Python**: 3.8+
4. **PaddlePaddle**: GPU version
5. **KiCad 8**: For kicad-cli rendering

## Step 1: Data Preparation

```bash
# Option A: Full pipeline (requires GitHub token)
python scripts/collect_data.py --github-token YOUR_TOKEN --max-repos 200
python scripts/build_dataset.py --project-dir .

# Option B: Synthetic only (no token needed)
python scripts/build_dataset.py --skip-scraping --synthetic-count 500
```

## Step 2: Download Pre-trained Models

```bash
# Download PP-OCRv4 pre-trained weights
mkdir -p pretrain_models
# Detection model
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar
tar xf ch_PP-OCRv4_det_train.tar -C pretrain_models/
# Recognition model
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar
tar xf ch_PP-OCRv4_rec_train.tar -C pretrain_models/
```

## Step 3: Training

### Detection Model

```bash
python scripts/train.py --task det --distributed --gpu-ids 0,1
```

Key hyperparameters:
- Learning rate: 0.001 (with warmup)
- Batch size: 16 per GPU
- Epochs: 200
- Backbone: MobileNetV3-large

### Recognition Model

```bash
python scripts/train.py --task rec --distributed --gpu-ids 0,1
```

Key hyperparameters:
- Learning rate: 0.001
- Batch size: 128 per GPU
- Epochs: 200
- Architecture: SVTR_LCNet with CTC + SAR heads

### VLM Fine-tuning (Advanced)

```bash
python scripts/train.py --task vl --epochs 10 --batch-size 4
```

## Step 4: Evaluation

```bash
python scripts/evaluate.py --eval-dir data/eval --output-dir output/eval
```

## Step 5: Export & Inference

```bash
# Export models to inference format
python -m src.model.export_model

# Run inference
python scripts/infer.py --image test.png --verify --save-netlist
```

## Tips for Best Results

1. **More data = better**: Aim for 1000+ training images
2. **Degradation matters**: Always include degraded variants in training
3. **Character dictionary**: Ensure all electronic symbols (Ω, μ, etc.) are in the dictionary
4. **Learning rate**: Start with 0.001, reduce if loss plateaus
5. **Early stopping**: Monitor validation loss, stop if no improvement for 20 epochs
