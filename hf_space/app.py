import gradio as gr

with gr.Blocks(title="CircuitOCR") as demo:
    gr.Markdown("# CircuitOCR — PaddleOCR-VL-0.9B + LoRA (r=16)")
    gr.Markdown("### v1: exp6 (1500 real samples) | v2: Phase 1 (5000 synthetic KiCad)")

    with gr.Tabs():
        with gr.TabItem("Benchmark"):
            gr.Markdown("""
## Multi-Metric Benchmark (test_clean, N=30)

| Model | CompF1 | JointF1 | NED | Description |
|---|---:|---:|---:|---|
| **v2 (Phase 1)** | **0.304** | 0.005 | 0.942 | 5000 synthetic KiCad pre-training |
| v1 (exp6) | 0.119 | 0.008 | 0.946 | 1500 real circuits + synth text |

**v2 CompF1 is 2.5x v1.** Component refdes recognition greatly improved by synthetic data.
JointF1 remains very low for both models — (refdes, value) joint recognition is the key unsolved challenge.

[Full Report](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf) | [Beamer](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/slides/beamer_slides.pdf)
""")

        with gr.TabItem("Models"):
            gr.Markdown("""
## Available LoRA Weights

| File | Model | CompF1 | JointF1 |
|------|-------|:---:|:---:|
| `lora_phase1_synth5k.pdparams` | v2 — Synthetic pre-training (5000 KiCad) | 0.304 | 0.005 |
| `lora_exp6_best.pdparams` | v1 — Real data baseline (1500 samples) | 0.119 | 0.008 |

**Usage:** Load base model PaddleOCR-VL-0.9B → apply LoRA (r=16, α=32, Q/K/V/O+MLP) → load weights via `p.set_value()`.

[HuggingFace Repo](https://huggingface.co/yingchu83/CircuitOCR-lora)
""")

        with gr.TabItem("About"):
            gr.Markdown("""
## CircuitOCR

PaddleOCR-VL-0.9B + LoRA for circuit schematic OCR.

### Methods
- **v1 (exp6):** 1200 real KiCad schematics + 300 synthetic text (anti-collapse). 2 epochs, lr=2e-5.
- **v2 (Phase 1):** 5000 programmatic KiCad schematics + anti-collapse text. Pure synthetic pre-training. 2 epochs, lr=2e-5.

### Key Findings
- Synthetic text at 20% ratio prevents modality collapse on small datasets
- 5000 synthetic KiCad schematics boost CompF1 2.5x but don't improve joint (refdes, value) pairing
- Phase 2 fine-tuning on real data causes collapse — style gap between synthetic and real KiCad renders

### Links
- [Code](https://github.com/ZhangJ83/circuit-ocr-paddle)
- [Weights](https://huggingface.co/yingchu83/CircuitOCR-lora)
- [Dataset](https://github.com/ZhangJ83/circuit_ocr_dataset_final)
- [Tech Report](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf)

MIT License.
""")

demo.launch()
