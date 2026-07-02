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

        **Note:** Full model inference requires GPU. For quick results, see the **Examples** tab
        for pre-computed V8-Fixed predictions on real test samples.
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
        gr.Markdown("### V8-Fixed Predictions (easy100 Pure Test Set)")
        gr.Markdown(f"**Avg. NED: 0.7791** (Base: 0.9390, -17.0% error reduction)")
        for i, ex in enumerate(EXAMPLES[:8]):
            with gr.Row():
                with gr.Column(scale=1):
                    img_path = ex.get("image", "")
                    if os.path.exists(img_path):
                        gr.Image(img_path, label=f"Sample {i+1}")
                    else:
                        gr.Markdown(f"*Image {i+1}*")
                with gr.Column(scale=2):
                    gt_preview = ex.get('gt', '')[:200]
                    v8_preview = ex.get('v8_pred', '')[:200]
                    gr.Markdown(f"**Ground Truth:**\n```\n{gt_preview}\n```")
                    gr.Markdown(f"**V8-Fixed Prediction:**\n```\n{v8_preview}\n```")

# ===== Tab 3: Benchmark =====
def benchmark_tab():
    gr.Markdown("""
    ## Model Evolution

    | Version | Architecture | Train Samples | Params | easy50 NED | easy100 NED |
    |---------|-------------|--------------|--------|------------|-------------|
    | Base | — | 0 | 0 | 0.9424 | 0.9390 |
    | V5 LLM-Only | Frozen Proj, r=8 | 1,357 | 1.25M | 0.9066 | — |
    | **V8-Fixed** | **Wide LoRA, r=16** | **1,554** | **5.7M** | **0.7892** | **0.7791** |

    ### V8-Fixed Key Innovations
    - **Causal 1-Token Shift Fix**: Corrected double-shifting that corrupted gradient signals
    - **BPE Boundary Merging Fix**: Separate prompt/label tokenization eliminates boundary token merging
    - **Wide LoRA Capacity**: r=16, alpha=32 across 310 projection matrices (5.7M params)
    - **V5 Golden Dataset**: 2,555 balanced samples (KiCad 1,857 + Masala 698)

    ### Training
    - 3 epochs (1,800 steps) on RTX 4060 8GB (~2 hours)
    - Best checkpoint: s1600 (Epoch 2.6)
    - Loss: 2.71 → 0.30, no modality collapse
    """)

# ===== Tab 4: About =====
def about_tab():
    gr.Markdown("""
    ## CircuitOCR V8-Fixed: Best Circuit OCR Model

    The first open-source LoRA fine-tuned model for circuit schematic OCR that achieves
    practical performance — Avg. NED 0.7760 on easy100 test set.

    ### Results
    - **easy100 Avg. NED: 0.7760** (Base 0.9372, -17.2% error)
    - **easy50 Avg. NED: 0.8257** (Base 0.9634, -14.3% error)
    - Structured circuit netlist output with correct EOS token generation

    ### Links
    - [GitHub Repository](https://github.com/ZhangJ83/circuit-ocr-paddle)
    - [LoRA Weights](https://huggingface.co/yingchu83/CircuitOCR-lora)
    - [Technical Report (Chinese)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf)
    - [Technical Report (English)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)

    ### Citation
    ```bibtex
    @misc{zhang2026circuitocr,
      title={PaddleOCR-VL-Circuit: Built for Schematic Diagram Understanding},
      author={Jianning Zhang and Yifei Chen},
      year={2026},
      url={https://github.com/ZhangJ83/circuit-ocr-paddle},
    }
    ```
    """)

# ===== Build App =====
with gr.Blocks(title="CircuitOCR V8-Fixed", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CircuitOCR V8-Fixed
    ### PaddleOCR-VL-0.9B + Wide LoRA (r=16) — easy100 NED 0.7760, No Collapse
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
