"""CircuitOCR Gradio Demo — Schematic OCR and Netlist Extraction."""
import gradio as gr
import json, os
from pathlib import Path
from PIL import Image

DATASET_DIR = Path(__file__).parent
RESULTS_DIR = DATASET_DIR

# Load pre-computed examples
def load_examples():
    examples = []
    for results_file in ['results_easy50_r16e3.jsonl', 'results_base_easy50.jsonl']:
        path = RESULTS_DIR / results_file
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    img_rel = d['images'][0].lstrip('./')
                    img_path = DATASET_DIR / img_rel
                    if img_path.exists():
                        examples.append({
                            'image': str(img_path),
                            'gt': d.get('label', '')[:200],
                            'pred': d.get('prediction', '')[:200],
                            'source': results_file,
                        })
        break  # Use first available results file
    return examples[:10]


def show_example(idx):
    """Display pre-computed example with GT and prediction."""
    exs = load_examples()
    if not exs or idx >= len(exs):
        return None, "No examples available", "", ""
    ex = exs[idx]
    img = Image.open(ex['image'])
    return img, f"## Ground Truth\n```\n{ex['gt']}\n```", \
           f"## r16 Model Prediction\n```\n{ex['pred']}\n```", \
           f"Source: {ex['source']}"


def create_demo():
    exs = load_examples()
    example_imgs = [e['image'] for e in exs[:6]] if exs else []

    with gr.Blocks(title="CircuitOCR — Circuit Schematic OCR", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # CircuitOCR: Built for Schematic Diagram Understanding
        ### PaddleOCR-VL-0.9B + LoRA (r=16, +9.6% NED improvement)

        Upload a circuit schematic image to extract component references, values, and net labels.
        """)

        with gr.Tabs():
            with gr.TabItem("Inference"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(type="filepath", label="Upload Circuit Schematic")
                        btn = gr.Button("Extract Netlist", variant="primary")
                    with gr.Column(scale=1):
                        output_text = gr.Textbox(label="Extracted Netlist", lines=12, max_lines=20)

                btn.click(
                    fn=lambda x: "Upload an image and click 'Extract Netlist' to run inference. "
                                 "(Full model inference requires GPU with >= 8GB VRAM.)",
                    inputs=[input_img], outputs=[output_text]
                )

                gr.Markdown("""
                **Note**: Full model inference requires GPU with >= 8GB VRAM.
                For quick testing, see the Examples tab for pre-computed results.
                """)

            with gr.TabItem("Examples"):
                gr.Markdown("### Pre-computed Model Comparison (easy50 test set)")
                with gr.Row():
                    with gr.Column(scale=2):
                        ex_img = gr.Image(type="pil", label="Circuit Schematic")
                    with gr.Column(scale=3):
                        ex_gt = gr.Markdown("Ground Truth")
                        ex_pred = gr.Markdown("Model Prediction")
                        ex_info = gr.Markdown("")

                slider = gr.Slider(0, max(0, len(exs)-1), 0, step=1, label="Example Index")
                slider.change(show_example, inputs=[slider],
                            outputs=[ex_img, ex_gt, ex_pred, ex_info])

                gr.Examples(examples=example_imgs, inputs=[input_img], label="Sample Images")

            with gr.TabItem("Benchmark Results"):
                gr.Markdown("""
                ## Model Performance (Avg. NED, lower is better)

                | Tier | Base | r16 LoRA | Improvement |
                |------|------|----------|-------------|
                | easy50 | 0.8895 | **0.8044** | **+9.6%** |
                | easy100 | 0.8999 | **0.8291** | **+7.9%** |
                | easy200 | 0.9139 | **0.8624** | **+5.6%** |
                | full523 | 0.9455 | **0.9164** | **+3.1%** |

                - **Base model**: PaddleOCR-VL-0.9B (908M params)
                - **Best model**: Projector LoRA r=16, 3-epoch training (53 min on RTX 4060 8GB)
                - **Training data**: 24,717 samples (3 sources)
                - **Degraded evaluation set**: 250 samples with 5 realistic visual transforms
                """)

            with gr.TabItem("About"):
                gr.Markdown("""
                ## CircuitOCR

                A circuit schematic OCR and netlist extraction system based on PaddleOCR-VL-0.9B
                with LoRA parameter-efficient fine-tuning.

                ### Key Features
                - **Multi-source dataset**: 24,717 samples from real OSS projects, textbooks, and synthetic generation
                - **LoRA fine-tuning**: r=16 rank, only 0.63% parameters updated
                - **Projector bottleneck discovery**: Vision-language projection layer identified as key bottleneck
                - **4-tier evaluation**: Systematic testing across difficulty levels
                - **Degraded evaluation**: 5 realistic visual transforms for robustness testing

                ### Links
                - [GitHub Repository](https://github.com/ZhangJ83/circuit-ocr-paddle)
                - [Technical Report (Chinese)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf)
                - [Technical Report (English)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)
                - [Dataset](https://github.com/ZhangJ83/circuit_ocr_dataset_final)
                """)

    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
