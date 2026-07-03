# Monkey patch huggingface_hub to include HfFolder to satisfy Gradio's import
import huggingface_hub
try:
    from huggingface_hub import HfFolder
except ImportError:
    class DummyHfFolder:
        @classmethod
        def get_token(cls):
            import os
            return os.environ.get("HF_TOKEN")
        @classmethod
        def save_token(cls, token):
            pass
        @classmethod
        def delete_token(cls):
            pass
    huggingface_hub.HfFolder = DummyHfFolder

import gradio as gr
import json
import os

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(DATASET_DIR, "examples.json")

def load_examples():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return []

EXAMPLES = load_examples()

# ===== Tab 1: Inference =====
def inference_tab():
    with gr.Column():
        gr.Markdown("""
        ### Upload a circuit schematic image

        **Note:** Full model inference requires GPU (RTX 4060 8GB or higher).
        For quick results, see the **Examples** tab for pre-computed
        V10-Fixed S600 predictions on real test samples.
        """)
        img = gr.Image(type="filepath", label="Circuit Schematic")
        btn = gr.Button("Extract Netlist", variant="primary")
        out = gr.Textbox(label="Output", lines=8)
        btn.click(lambda x: "GPU inference available in local version.\nSee Examples tab for pre-computed results.",
                  inputs=[img], outputs=[out])

# ===== Tab 2: Examples =====
def examples_tab():
    if not EXAMPLES:
        with gr.Column():
            gr.Markdown("### Model Comparison Examples")
            gr.Markdown("*Examples loading...*")
        return

    with gr.Column():
        gr.Markdown("""
        ### V10-Fixed S600 vs Base Model — Real Test Samples (easy50-pure)

        **S600 is the optimal checkpoint** (Phase 1 multi-metric evaluation, 44 samples).
        The base model fails systematically: hallucinating generic document text
        or collapsing into single-token repetition. S600 correctly identifies
        component designators (R1, C2), values (10k, 100nF), and net labels (GND, VCC).
        """)
        # Show first 6 examples with base and S600 outputs side by side
        for i, ex in enumerate(EXAMPLES[:6]):
            with gr.Row():
                with gr.Column(scale=1):
                    img_path = ex.get("image", "")
                    if os.path.exists(img_path):
                        gr.Image(img_path, label=f"Sample {i+1}")
                    else:
                        gr.Markdown(f"*Image {i+1} not found*")
                with gr.Column(scale=2):
                    gt_preview = ex.get('gt', '')[:200]
                    base_preview = ex.get('base_pred', '')[:200]
                    s600_preview = ex.get('s600_pred', '')[:200]
                    gr.Markdown(f"**Ground Truth:**\n```\n{gt_preview}\n```")
                    gr.Markdown(f"**Base Model (pre-fine-tune):**\n```\n{base_preview}\n```")
                    gr.Markdown(f"**S600 (fine-tuned):**\n```\n{s600_preview}\n```")

# ===== Tab 3: Benchmark =====
def benchmark_tab():
    gr.Markdown("""
    ## Phase 1 Multi-Metric Benchmark (V10-Fixed, easy50-pure, 44 samples)

    All rows evaluated with the same fixed `eval_benchmark_v3.py` script — directly comparable.

    | Model | CompF1 ↑ | TokenRec ↑ | NED ↓ | RepRate | Diversity |
    |---|---|---:|---:|---:|---:|
    | Base (no fine-tune) | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
    | S400 | 0.1820 | 0.1302 | 0.8298 | 20.5% | 95.5% |
    | **S600** ★ | **0.2061** | **0.1540** | **0.8031** | 15.9% | 90.9% |
    | S800 (overfit) | 0.2080 | 0.1191 | 0.8063 | 40.9% | 93.2% |

    - **CompF1**: 4.5× over baseline | **TokenRec**: 96× over baseline | **NED**: 13.6% relative error reduction
    - S600 is optimal across NED, TokenRec, and CompRec with healthy diversity
    - S800 overfits: repetition rate surges to 40.9%, TokenRec collapses to 0.1191

    ### Version Progression

    Earlier versions used different evaluation protocols and are **not directly comparable**
    to the Phase 1 numbers above. See the [technical report](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)
    for full version history and evaluation methodology.

    ### Key Technical Details

    - **Architecture**: PaddleOCR-VL-0.9B (908M params) + LoRA (r=16, α=32, 5.7M trainable)
    - **Strategy**: LLM-Only LoRA — freeze vision-language Projector to prevent modality collapse
    - **Dataset**: V5 Golden (1,857 samples: 500 synthetic + 1,357 real KiCad projects)
    - **Training**: 3 epochs (1,165 steps), single RTX 4060 8GB, ~43 minutes
    - **Three critical bugs fixed**: causal token double-shift, BPE boundary merging, set_state_dict silent failure (Paddle 3.1.0)
    """)

# ===== Tab 4: About =====
def about_tab():
    gr.Markdown("""
    ## CircuitOCR V10-Fixed

    Open-source LoRA fine-tuned model for circuit schematic OCR and netlist extraction,
    based on PaddleOCR-VL-0.9B. The first model to achieve practical component-level
    recognition after resolving modality collapse.

    ### Results (Phase 1)
    - **CompF1**: 0.2061 (4.5× over baseline 0.0455)
    - **Token Recall**: 0.1540 (96× over baseline 0.0016)
    - **NED**: 0.8031 (13.6% relative error reduction vs baseline 0.9296)
    - **Diversity**: 90.9% (no modality collapse)

    ### Links
    - [GitHub Repository](https://github.com/ZhangJ83/circuit-ocr-paddle)
    - [Dataset Repository](https://github.com/ZhangJ83/circuit-ocr-dataset)
    - [LoRA Weights](https://huggingface.co/yingchu83/CircuitOCR-lora)
    - [Technical Report (English)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)
    - [Technical Report (Chinese)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf)

    ### Citation
    ```bibtex
    @misc{zhang2026circuitocr,
      title={CircuitOCR: LoRA Fine-Tuning PaddleOCR-VL for Circuit Schematic Understanding},
      author={Jianning Zhang and Yifei Chen},
      year={2026},
      url={https://github.com/ZhangJ83/circuit-ocr-paddle},
    }
    ```
    """)

# ===== Build App =====
with gr.Blocks(title="CircuitOCR V10-Fixed", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CircuitOCR V10-Fixed (S600)
    ### PaddleOCR-VL-0.9B + Wide LoRA (r=16) — CompF1 0.2061, NED 0.8031, No Modality Collapse
    """)

    with gr.Tabs():
        with gr.TabItem("Inference"):
            inference_tab()
        with gr.TabItem("Examples"):
            examples_tab()
        with gr.TabItem("Benchmark"):
            benchmark_tab()
        with gr.TabItem("About"):
            about_tab()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
