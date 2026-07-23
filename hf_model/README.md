---
tags:
- paddleocr
- circuit-schematic
- ocr
- lora
- eda
license: mit
---

# CircuitOCR LoRA Weights

LoRA fine-tuning weights for PaddleOCR-VL-0.9B on circuit schematic OCR.

## Available Checkpoints

| Version | File | CompF1 | NED ↓ | Description |
|---------|------|:------:|:-----:|-------------|
| **v2 (Phase 1)** ★ | `lora_phase1_synth5k.pdparams` | **0.304** | **0.942** | **Best model — 5,000 synthetic KiCad pre-training** |
| v1 (exp6) | `lora_exp6_best.pdparams` | 0.119 | 0.946 | 1,500 real + synthetic text (20% mix) |

> Benchmark on test_clean (N=30), greedy decoding, repetition_penalty=1.1, max_tokens=80.

## Architecture

- Base model: PaddleOCR-VL-0.9B
- LoRA: r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj, linear_1, linear_2]
- Trainable params: 5.73M / 908M (0.63%)
- Training: single RTX 4060 8GB

## Quick Start

```python
# Load base model → apply LoRA → load weights via p.set_value()
# See github repo for full training/eval scripts.
```

## Links

- [GitHub Repository](https://github.com/ZhangJ83/circuit-ocr-paddle)
- [Live Demo](https://huggingface.co/spaces/yingchu83/CircuitOCR)
- [Dataset](https://github.com/ZhangJ83/circuit_ocr_dataset_final)
- [Report (CN)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf)
- [Report (EN)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)
