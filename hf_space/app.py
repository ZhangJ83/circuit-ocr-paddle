"""CircuitOCR HuggingFace Space Demo."""
import gradio as gr
import json
import os

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(DATASET_DIR, "examples.json")

# Load pre-computed examples (lightweight, no GPU needed)
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
        for pre-computed comparisons on real test samples.
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
        gr.Markdown("### V5 LLM-Only LoRA vs Old Model (easy50 Test Set)")
        gr.Markdown("**Old model:** Full LoRA r=16 → collapsed, all samples output `12\\n100\\n100...`")
        gr.Markdown("**V5 model:** LLM-only LoRA r=8, frozen projector → diverse, reads circuit content")
        for i, ex in enumerate(EXAMPLES[:6]):
            with gr.Row():
                with gr.Column(scale=1):
                    img_path = ex.get("image", "")
                    if os.path.exists(img_path):
                        gr.Image(img_path, label=f"Sample {i+1}")
                    else:
                        gr.Markdown(f"*Image {i+1}*")
                with gr.Column(scale=1):
                    gr.Markdown(f"**Ground Truth:**\n```\n{ex.get('gt','')[:150]}\n```")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**OLD (collapsed):**\n```\n{ex.get('old_pred','')[:150]}\n```")
                with gr.Column(scale=1):
                    gr.Markdown(f"**NEW V5 (diverse):**\n```\n{ex.get('v5_pred','')[:150]}\n```")

# ===== Tab 3: Benchmark =====
def benchmark_tab():
    gr.Markdown("""
    ## Model Evolution

    | Version | LoRA Target | Rank | Diversity | Key Issue |
    |---------|------------|------|-----------|-----------|
    | Base | — | — | 100% | Outputs image descriptions, not netlists |
    | V1 (r=16 Full) | q/k/v/o + Projector | r=16 | 4% | Complete collapse: all outputs `12\\n100\\n100...` |
    | V2 (Projector-only) | Projector only | r=16 | 90% | Preserved diversity but destroyed visual features |
    | **V5 (LLM-only)** | **LLM self-attn only** | **r=8** | **90%** | **First diverse model that reads circuits** |

    ### V5 Key Innovations
    - **Freeze Projector**: Preserves pre-trained visual-language alignment
    - **LLM-Only LoRA**: Teaches LLM to format netlists without destroying vision
    - **Resolution 384px**: Higher than V2's 168px, keeps text readable

    """)

# ===== Tab 4: About =====
def about_tab():
    gr.Markdown("""
    ## CircuitOCR V5: First Diverse Circuit OCR Model

    The first open-source LoRA fine-tuned model for circuit schematic OCR that actually works —
    90% output diversity, no modality collapse.

    ### Key Features
    - **Zero Collapse**: 90% output diversity across all test samples
    - **LLM-Only LoRA (r=8)**: Freeze projector to protect vision, fine-tune LLM self-attention
    - **Real Component Recognition**: Identifies ESP32, LM7805, resistors, capacitors
    - **384px Resolution**: 2.3x higher than previous attempts

    ### Links
    - [GitHub Repository](https://github.com/ZhangJ83/circuit-ocr-paddle)
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
with gr.Blocks(title="CircuitOCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CircuitOCR V5: First Diverse Circuit OCR Model
    ### PaddleOCR-VL-0.9B + LLM-Only LoRA (r=8) — 90% Output Diversity, No Collapse
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
