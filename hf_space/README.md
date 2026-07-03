---
title: CircuitOCR
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.3.0
app_file: app.py
pinned: true
license: mit
python_version: "3.10"
---

# CircuitOCR: Schematic Diagram Understanding

PaddleOCR-VL-0.9B + LoRA fine-tuning for circuit schematic OCR and netlist extraction.

## Phase 1 Benchmark (V10-Fixed S600, easy50-pure, 44 samples)

| Model | CompF1 ↑ | TokenRec ↑ | NED ↓ | RepRate | Diversity |
|:---|---:|---:|---:|---:|---:|
| Base (no fine-tune) | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
| S400 | 0.1820 | 0.1302 | 0.8298 | 20.5% | 95.5% |
| **S600** ★ | **0.2061** | **0.1540** | **0.8031** | 15.9% | 90.9% |
| S800 (overfit) | 0.2080 | 0.1191 | 0.8063 | 40.9% | 93.2% |

- Component F1: **4.5×** improvement | Token Recall: **96×** improvement | NED: **13.6%** relative error reduction
- S600 is the optimal checkpoint; S800 overfits (repetition rate 40.9%)

## Key Technical Details

- **Architecture**: PaddleOCR-VL-0.9B (908M params) + Wide LoRA (r=16, α=32, 5.7M params, 0.63%)
- **Strategy**: LLM-Only LoRA — freeze Projector, fine-tune only LLM layers to avoid modality collapse
- **Dataset**: V5 Golden — 1,857 samples (500 synthetic V3 + 1,357 real KiCad projects), 100% visual-literal alignment
- **Training**: 3 epochs (1,165 steps), single RTX 4060 8GB, ~43 minutes
- **Three bugs fixed**: causal token double-shift, BPE boundary merging, set_state_dict silent failure (Paddle 3.1.0)

[GitHub](https://github.com/ZhangJ83/circuit-ocr-paddle) | [Dataset](https://github.com/ZhangJ83/circuit-ocr-dataset) | [Report (EN)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf) | [Report (CN)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf) | [LoRA Weights](https://huggingface.co/yingchu83/CircuitOCR-lora)
