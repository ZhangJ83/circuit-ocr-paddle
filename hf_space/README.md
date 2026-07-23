---
title: CircuitOCR
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: true
license: mit
python_version: "3.10"
---

# CircuitOCR: Schematic Diagram Understanding

PaddleOCR-VL-0.9B + LoRA fine-tuning for circuit schematic OCR.

## Benchmark (test_clean, N=30)

| Model | CompF1 ↑ | LineAcc ↑ | NED ↓ | Description |
|:---|---:|---:|---:|---|
| Base | 0.000 | 0.000 | 0.944 | No fine-tuning |
| **v2 (Phase 1)** ★ | **0.304** | **0.040** | **0.942** | 5,000 synthetic KiCad pre-training |
| v1 (exp6) | 0.119 | 0.033 | 0.946 | 1,500 real + synthetic text |

- CompF1: **2.6×** improvement over v1 | All training on single RTX 4060 8GB

## Key Details

- **Strategy**: Freeze Projector → LoRA (r=16, α=32, 5.7M params) on LLM only
- **v2**: 5,000 synthetic KiCad schematics (kicad-cli), pure synthetic pre-training
- **v1**: 1,500 samples (1,200 real KiCad + 300 synthetic text, 20% mix)
- **Data**: Dataset A — 1,820 samples, 7-round validation, accuracy >99%

[GitHub](https://github.com/ZhangJ83/circuit-ocr-paddle) | [Dataset](https://github.com/ZhangJ83/circuit-ocr-dataset) | [Report (EN)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf) | [Report (CN)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf) | [LoRA Weights](https://huggingface.co/yingchu83/CircuitOCR-lora)
