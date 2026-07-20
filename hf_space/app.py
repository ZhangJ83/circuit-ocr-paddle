# Monkey patch huggingface_hub for Gradio compatibility
import huggingface_hub
try:
    from huggingface_hub import HfFolder
except ImportError:
    class DummyHfFolder:
        @classmethod def get_token(cls): return __import__('os').environ.get("HF_TOKEN")
        @classmethod def save_token(cls, token): pass
        @classmethod def delete_token(cls): pass
    huggingface_hub.HfFolder = DummyHfFolder

import gradio as gr
import json, os

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(DATASET_DIR, "examples.json")
EXAMPLES = json.load(open(RESULTS_FILE, encoding='utf-8')) if os.path.exists(RESULTS_FILE) else []

# ===== Tab 1: Inference =====
def inference_tab():
    with gr.Column():
        gr.Markdown("""
        ### Research Prototype -- Pre-Computed Results Only

        This Space runs on **CPU-only free tier** and cannot load PaddleOCR-VL-0.9B
        for live inference. All outputs are **pre-computed** on an RTX 4060 8GB GPU.

        **To run inference locally:**
        ```bash
        git clone https://github.com/ZhangJ83/circuit-ocr-paddle
        # Download LoRA weights: https://huggingface.co/yingchu83/CircuitOCR-lora
        python eval_benchmark_v3.py --image your_circuit.png
        ```
        See the **Examples** tab for pre-computed predictions with honest annotations.
        """)
        img = gr.Image(type="filepath", label="Circuit Schematic (display only)")
        btn = gr.Button("Extract Netlist", variant="primary")
        out = gr.Textbox(label="Status", lines=4)
        btn.click(
            lambda x: "Live inference unavailable on CPU-only free tier. See Examples tab.",
            inputs=[img], outputs=[out]
        )

# ===== Tab 2: Examples =====
def examples_tab():
    if not EXAMPLES:
        gr.Markdown("### Examples loading..."); return
    with gr.Column():
        gr.Markdown("""
        ### Phase 2 (exp5 Anti-Overfit + Synthetic Data) -- Annotated Test Samples

        **Each sample annotated with what the model gets right and wrong.**
        exp5 uses lr=1e-5, dropout=0.10, 3 epochs on 1500 mixed samples
        (1200 circuits + 300 synthetic text images at 20%).
        Synthetic data forces visual-text alignment, preventing modality collapse.
        """)
        for i, ex in enumerate(EXAMPLES):
            verdict = ex.get("verdict", "")
            badge = {"partial": "PARTIAL", "failure": "FAILURE"}.get(verdict, "")
            emoji = {"partial": "🟡", "failure": "🔴"}.get(verdict, "⚪")
            with gr.Row():
                with gr.Column(scale=1):
                    img_path = ex.get("image", "")
                    if os.path.exists(img_path):
                        gr.Image(img_path, label=f"Sample {i+1} {emoji} {badge}")
                    else:
                        gr.Markdown(f"*Image {i+1} not found*")
                with gr.Column(scale=2):
                    gt_preview = ex.get('gt', '')[:300]
                    pred_preview = ex.get('pred', '')[:300]
                    gr.Markdown(f"**Ground Truth:**\n```\n{gt_preview}\n```")
                    gr.Markdown(f"**exp5 Prediction:**\n```\n{pred_preview}\n```")
                    if ex.get("note"):
                        gr.Markdown(f"> {ex['note']}")

# ===== Tab 3: Benchmark =====
def benchmark_tab():
    gr.Markdown("""
## Multi-Metric Benchmark

All rows evaluated with `eval_benchmark_v3.py` and `fast_eval.py` -- directly comparable.

### Phase 1: Four Experiments (1,200 samples, no synthetic data)

| Model | CompF1 | JointF1 | NED | RepRate | Notes |
|---|---:|---:|---:|---:|---|
| Base (no fine-tune) | 0.046 | 0.000 | 0.930 | 3% | Generic collapse |
| exp1 (Baseline, 384px) | 0.037 | 0.005 | 0.944 | 53% | Counting collapse |
| exp2 (HiRes, 512px) | 0.058 | 0.000 | 0.941 | 20% | Still collapsed |
| exp3 (Anti-Overfit) | 0.028 | 0.000 | 0.940 | 13% | Gradual recovery |
| exp4 (Unfrozen Proj.) | 0.092 | 0.005 | 0.944 | 47% | Higher CF1, still collapsed |

### Phase 2: Synthetic Data Anti-Collapse (1,500 samples, 20% synth text)

| Model | CompF1 | JointF1 | NED | RepRate | Notes |
|---|---:|---:|---:|---:|---|
| **exp5 (Anti-Overfit + Synth)** | **0.126** | **0.011** | **0.941** | **<15%** | Stable S200-S800 |
| **exp6 (Baseline + Synth)** | **0.156** | **0.011** | **0.941** | **<15%** | Highest CompF1 |

### Key Findings

- **Synthetic data is the single most effective intervention**: CompF1 4.2x (0.037 -> 0.156)
- **JointF1 2.2x**: (refdes, value) pair accuracy doubled (0.005 -> 0.011)
- **Modality collapse solved**: exp5 training validation shows perfect capacitor-BOM recognition at S200
- **Bottleneck remains**: 1,200 training schematics insufficient for JointF1 > 0.05

### Limitations

- ExactMatch = 0%: no model reconstructs a full netlist end-to-end
- Value hallucination: model outputs generic values from training distribution
- NED ~0.94: dominated by repetitive predictions on unseen samples
- 0.9B parameter ceiling: capacity limited by base model architecture

### Version Progression

| Phase | Goal | Status |
|---|---|---|
| Phase 1 | 4 experiments, establish baseline, identify collapse | Complete |
| Phase 2 | Synthetic data anti-collapse, 2 experiments | Complete |
| Phase 3 | Real-camera data, RL+SPICE alignment | Planned |

Full details: [Technical Report](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf)
""")

# ===== Tab 4: About =====
def about_tab():
    gr.Markdown("""
## CircuitOCR -- Research Prototype (Phase 2 Complete)

Open-source LoRA fine-tuned model for circuit schematic OCR, based on
**PaddleOCR-VL-0.9B**. First to demonstrate that synthetic text-image
mixing (20%) can prevent modality collapse in small-dataset VLM-OCR tasks.

### What It Achieves (Phase 2)
- **CompF1**: 0.156 (4.2x over Phase 1 baseline 0.037)
- **JointF1**: 0.011 (2.2x over Phase 1 best 0.005)
- **Modality collapse prevented**: synthetic text data forces visual-text alignment
- **Dataset**: 1,520 samples with 7-round verification, quality report published
- **Dataset evaluation**: 4-dimension framework with multi-dim difficulty labels

### What It Does NOT Achieve (Honest Assessment)
- **ExactMatch = 0%**: cannot perfectly reconstruct any netlist
- **JointF1 = 0.011**: far below usable threshold of 0.05
- **NED ~0.94**: per-token accuracy dominated by collapse on unseen samples
- **Value hallucination**: model outputs generic values from limited training distribution
- **Not production-ready**: research baseline and proof-of-concept

### Links
- [GitHub Repository](https://github.com/ZhangJ83/circuit-ocr-paddle)
- [Dataset Repository](https://github.com/ZhangJ83/circuit_ocr_dataset_final)
- [LoRA Weights](https://huggingface.co/yingchu83/CircuitOCR-lora)
- [Technical Report (Chinese, 24pp)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf)
- [Technical Report (English)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)
- [Beamer Slides (37pp)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/slides/beamer_slides.pdf)

### Citation
```bibtex
@misc{zhang2026circuitocr,
  title={CircuitOCR: LoRA Fine-Tuning PaddleOCR-VL for Circuit Schematic Understanding},
  author={Jianning Zhang},
  year={2026},
  url={https://github.com/ZhangJ83/circuit-ocr-paddle},
}
```
""")

# ===== Build App =====
with gr.Blocks(title="CircuitOCR -- Research Prototype", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CircuitOCR (Phase 2) -- Research Prototype
    ### PaddleOCR-VL-0.9B + LoRA (r=16) | CompF1 0.156 | Phase 2 Complete | Synthetic Data Anti-Collapse
    """)
    with gr.Tabs():
        with gr.TabItem("Inference"): inference_tab()
        with gr.TabItem("Examples"): examples_tab()
        with gr.TabItem("Benchmark"): benchmark_tab()
        with gr.TabItem("About"): about_tab()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
