# CircuitOCR: Built for Schematic Diagram Understanding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PaddleOCR-VL](https://img.shields.io/badge/Base%20Model-PaddleOCR--VL--0.9B-blue)](https://github.com/PaddlePaddle/PaddleOCR)
[![LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA%20(r%3D16)-green)]()

**PaddleOCR-VL-0.9B + LoRA for Circuit Schematic OCR and Netlist Extraction**

The first open-source benchmark and fine-tuning pipeline for circuit schematic OCR. Achieves **+9.6% Avg. NED improvement** over the base model on the easy50 test set.

## Highlights

| Feature | Description |
|---------|-------------|
| **First Circuit OCR Benchmark** | 24,717 training samples + 523 test samples across 4 difficulty tiers |
| **Best Model: r=16 LoRA** | 0.8044 Avg. NED (+9.6% vs. base 0.8895), 53 min training on RTX 4060 |
| **Projector Bottleneck Discovery** | First systematic identification of vision-language projector as key bottleneck in circuit OCR |
| **Rank Sweet Spot** | r=8/16/32 full ablation: r=16 optimal, r=32 diverges (NED 1.0) |
| **Degraded Evaluation Set** | 250 samples with 5 realistic visual transforms (perspective, lighting, blur, noise, JPEG) |
| **Bilingual Report** | 15-page Chinese + 14-page English technical paper with full experimental details |
| **Gradio Demo** | Interactive web interface for circuit OCR inference |
| **Open Source** | Full pipeline: data collection, training, evaluation, inference, visualization |

## Quick Start

### Installation
```bash
conda create -n circuitocr python=3.10
conda activate circuitocr
pip install paddlepaddle-gpu paddleformers gradio pillow opencv-python
```

### One-Click Benchmark
```bash
cd circuit-ocr-dataset
python scripts/eval_benchmark.py \
    --model_type paddleocr-vl \
    --model_name_or_path "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/..." \
    --paddle_lora_dir "PaddleOCR-VL-LoRA-circuit-ocr" \
    --data_path ocr_vl_sft-test-easy50.jsonl \
    --output_path results.jsonl \
    --max_length 30 --resume
```

### Launch Demo
```bash
cd circuit-ocr-dataset
python demo.py  # Opens http://localhost:7860
```

### Train LoRA
```bash
python scripts/train_projector_lora.py        # r=8 (77 min)
python scripts/train_projector_lora_r16.py    # r=16 (53 min) 
python scripts/train_projector_lora_r32_v2.py # r=32 + gradient clip (experimental)
```

## Results Summary

### Model Performance (Avg. NED ↓)

| Configuration | easy50 | easy100 | easy200 | full523 |
|--------------|--------|---------|---------|---------|
| Base (PaddleOCR-VL-0.9B) | 0.8895 | 0.8999 | 0.9139 | 0.9455 |
| LoRA r=8 (q/k/v/o) | 0.8554 | — | — | — |
| LoRA r=8 + Projector 1ep | 0.8271 | — | — | — |
| LoRA r=8 + Projector 3ep | 0.8197 | — | — | — |
| **LoRA r=16 + Projector 3ep** | **0.8044** | **0.8291** | **0.8624** | **0.9164** |
| LoRA r=32 + Projector 3ep | 1.0000 | — | — | — |

### Rank Ablation (easy50)

| Rank | Trainable Params | NED | vs Base |
|------|-----------------|-----|---------|
| r=8 | 2.9M | 0.8197 | +7.8% |
| **r=16** | **5.7M** | **0.8044** | **+9.6%** |
| r=32 | 11.5M | 1.0000 | −12.4% |

## Project Structure

```
circuit-ocr-paddle/
├── arxiv_template/          # Technical report (Chinese + English LaTeX + PDF)
├── circuit-ocr-dataset/
│   ├── scripts/
│   │   ├── eval_benchmark.py          # Main evaluation script
│   │   ├── train_projector_lora.py    # r=8 LoRA training
│   │   ├── train_projector_lora_r16.py # r=16 LoRA training
│   │   ├── train_projector_lora_r32_v2.py # r=32 v2 (gradient clip)
│   │   ├── eval_topology.py           # Topology evaluation
│   │   ├── build_degraded_test.py     # Degraded test set builder
│   │   └── make_figures.py            # Report figure generation
│   ├── PaddleOCR-VL-LoRA-circuit-ocr/ # LoRA weights
│   ├── docs/                          # Documentation
│   ├── data/test_degraded/            # Degraded test images (250 samples)
│   ├── figures/                       # Generated figures
│   └── demo.py                        # Gradio demo
└── README.md
```

## Key Innovations

1. **Projector Bottleneck Discovery**: First to identify mlp_AR as the critical frozen layer in VLM fine-tuning for circuit OCR
2. **Rank Sweet Spot**: Experimental determination that r=16 is optimal for 2,433-sample circuit OCR
3. **OCR + Topology Dual-Task Framework**: Explicit decomposition into text recognition + connection inference sub-tasks

## Documentation

| Document | Content |
|----------|---------|
| [Data Collection](circuit-ocr-dataset/docs/data_collection.md) | Three-source collection pipeline |
| [Annotation Guideline](circuit-ocr-dataset/docs/annotation_guideline.md) | Netlist annotation rules |
| [Quality Control](circuit-ocr-dataset/docs/quality_control.md) | Three-round dedup + LLM scoring |
| [Data Statistics](circuit-ocr-dataset/docs/data_statistics.md) | Distribution analysis + visualizations |
| [Technical Report (CN)](arxiv_template/template.pdf) | Full Chinese paper (15 pages) |
| [Technical Report (EN)](arxiv_template/english.pdf) | Full English paper (14 pages) |

## Citation

```bibtex
@misc{zhang2026circuitocr,
  title={PaddleOCR-VL-Circuit: Built for Schematic Diagram Understanding},
  author={Jianning Zhang and Yifei Chen},
  year={2026},
  url={https://github.com/ZhangJ83/circuit-ocr-paddle},
}
```

## License

MIT License. Open Schematics and Masala-CHAI datasets under CC-BY-4.0.
