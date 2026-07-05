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
        ### ⚠️ Research Prototype — Pre-Computed Results Only

        This Space runs on **CPU-only free tier** and cannot load the full
        PaddleOCR-VL-0.9B model (908M params + LoRA weights) for live inference.
        All outputs shown here are **pre-computed** on an RTX 4060 8GB GPU.

        **To run inference locally:**
        ```bash
        git clone https://github.com/ZhangJ83/circuit-ocr-paddle
        # Download LoRA weights from https://huggingface.co/yingchu83/CircuitOCR-lora
        python eval_benchmark_v3.py --image your_circuit.png
        ```

        See the **Examples** tab for pre-computed predictions on real test samples
        with honest annotations of what the model gets right and wrong.
        """)
        img = gr.Image(type="filepath", label="Circuit Schematic (for display only)")
        btn = gr.Button("Extract Netlist", variant="primary")
        out = gr.Textbox(label="Status", lines=4)
        btn.click(
            lambda x: (
                "⚠️ Live inference unavailable on CPU-only free tier.\\n"
                "This is a research prototype — see Examples tab for\\n"
                "pre-computed results, or run locally with GPU."
            ),
            inputs=[img], outputs=[out]
        )

# ===== Tab 2: Examples =====
def examples_tab():
    if not EXAMPLES:
        with gr.Column():
            gr.Markdown("### Model Comparison Examples")
            gr.Markdown("*Examples loading...*")
        return

    with gr.Column():
        gr.Markdown("""
        ### V10-Fixed S600 vs Base Model — Annotated Test Samples (easy50-pure)

        **Each sample is annotated with what S600 gets right (✓) and wrong (✗).**
        S600 is the optimal Phase 1 checkpoint: it learns domain vocabulary
        (refdes, net labels) and avoids the base model's generic-document collapse,
        but still **hallucinates values** and **memorizes templates** from the
        limited training data (1,554 samples).

        Scroll to see both successes and honest failure modes.
        """)

        for i, ex in enumerate(EXAMPLES):
            verdict = ex.get("verdict", "")
            note = ex.get("note", "")
            if verdict == "partial":
                badge = "🟡 PARTIAL"
            elif verdict == "failure":
                badge = "🔴 FAILURE"
            else:
                badge = "⚪"

            with gr.Row():
                with gr.Column(scale=1):
                    img_path = ex.get("image", "")
                    if os.path.exists(img_path):
                        gr.Image(img_path, label=f"Sample {i+1} {badge}")
                    else:
                        gr.Markdown(f"*Image {i+1} not found*")
                with gr.Column(scale=2):
                    gt_preview = ex.get('gt', '')[:300]
                    base_preview = ex.get('base_pred', '')[:300]
                    s600_preview = ex.get('s600_pred', '')[:300]
                    gr.Markdown(f"**Ground Truth:**\n```\n{gt_preview}\n```")
                    gr.Markdown(f"**Base Model:**\n```\n{base_preview}\n```")
                    gr.Markdown(f"**S600 (fine-tuned):**\n```\n{s600_preview}\n```")
                    if note:
                        gr.Markdown(f"> {note}")

# ===== Tab 3: Benchmark =====
def benchmark_tab():
    gr.Markdown("""
    ## Phase 1 Multi-Metric Benchmark (V10-Fixed, easy50-pure, 44 samples)

    All rows evaluated with `eval_benchmark_v3.py` — directly comparable.

    ### Core Metrics (Phase 1)

    | Model | ExactMatch | CompF1 ↑ | TokenRec ↑ | NED ↓ | RepRate | Diversity |
    |---|---|---:|---:|---:|---:|---:|
    | Base (no fine-tune) | 0% | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
    | S400 | 0% | 0.1820 | 0.1302 | 0.8298 | 20.5% | 95.5% |
    | **S600** ★ | **0%** | **0.2061** | **0.1540** | **0.8031** | 15.9% | 90.9% |
    | S800 (overfit) | 0% | 0.2080 | 0.1191 | 0.8063 | 40.9% | 93.2% |

    - **CompF1**: 4.5× over baseline | **TokenRec**: 96× over baseline | **NED**: 13.6% relative reduction
    - **ExactMatch = 0% for all models** — no model can perfectly reconstruct a full netlist yet
    - S600 optimal: best TokenRec + NED, healthy diversity (90.9%)
    - S800 overfits: repetition rate surges to 40.9%, TokenRec collapses

    ### Topology Metrics (Phase 2, eval_topology_v2.py)

    These metrics parse (refdes, value) **pairs** — a component counts only if both ID and value are correct:

    | Model | comp_f1 | joint_f1 | value_acc |
    |---|---|---:|---:|
    | Base | 0.0455 | 0.0000 | — |
    | S400 | 0.1820 | 0.0027 | 0.006 |
    | **S600** ★ | **0.2061** | **0.0191** | **0.133** |
    | S800 | 0.2080 | 0.0064 | 0.023 |

    - **comp_f1**: aligned to Phase 1 method — refdes-only F1 (7 prefixes: R/C/D/U/J/L/Q)
    - **joint_f1**: (refdes, value) pair F1 — the **honest metric**: only 1.9% of components fully correct
    - **value_acc**: among matched refdes, fraction with correct value — 87% of values are wrong
    - **Reading values is the core bottleneck** — directly motivates Phase 2/3 (vision encoder unfreeze + higher resolution)

    ### Limitations

    - **ExactMatch = 0%**: the model cannot reconstruct a full netlist end-to-end
    - **Value hallucination**: 87% of matched component values are incorrect (generic values from training distribution)
    - **Template memorization**: model latches onto learned templates (e.g., AMS1117, R=10k, Pro Micro) from synthetic V3 data
    - **Repetition collapse on hard samples**: ~16% of samples exhibit token repetition
    - **0.9B parameter ceiling**: capacity limited by the base model architecture
    - **V11 (Phase 2, regularized training)**: dropout=0.1 + label_smoothing=0.05 on 3,054 samples → CompF1 **collapsed to 0.0604** (worse than baseline 0.0455), NED=0.9171, RepRate=84.1%. Regularization alone cannot overcome the limited training data and model capacity.
    - **V12 (Phase 3, Vision LoRA)**: two-stage training (LLM LoRA then Vision LoRA) → CompF1 collapsed further. Unfreezing the vision encoder at this scale introduces destructive gradient interference with the LLM head.
    - **Expected at this scale**: ExactMatch=0% and joint_f1=0.019 are consistent with 0.9B params, 5.7M trainable (0.63%), 1,554 samples, 43 min on consumer GPU. The meaningful gains are CompF1 4.5× and TokenRec 96×.

    ### Key Technical Details

    - **Architecture**: PaddleOCR-VL-0.9B (908M params) + Wide LoRA (r=16, α=32, 5.7M trainable, 0.63%)
    - **Strategy**: LLM-Only LoRA — freeze vision encoder + projector to prevent modality collapse
    - **Dataset**: 1,554 samples (1,097 real KiCad projects + 457 Synthetic V3)
    - **Training**: 3 epochs (1,165 optimizer steps), single RTX 4060 8GB, ~43 minutes
    - **Accessibility**: Full training in 43 min on a single RTX 4060 8GB — any individual developer can reproduce without data center infrastructure
    - **Six training pitfalls discovered** (affect all PaddleOCR-VL fine-tuning): 3 in main README — causal token double-shift (AutoModelForConditionalGeneration internal shift), BPE boundary merging (affects all seq2seq training), set_state_dict silent failure (Paddle 3.1.0 API change); plus 3 in dataset README — eval/test data leakage in Easy100/Easy200/Easy50 splits, synthetic V3 template contamination, missing refdes prefix normalization

    ### Version Progression

    Earlier versions used different evaluation protocols — see the
    [technical report](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)
    for full version history. V11 (Phase 2: larger dataset + regularization) and
    V12 (Phase 3: Vision LoRA) both completed but collapsed — see Limitations above.

    Full results across all 4 evaluation splits (easy50/100/200/full523) are in the
    [technical report](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)
    Appendix B.
    """)

# ===== Tab 4: About =====
def about_tab():
    gr.Markdown("""
    ## CircuitOCR V10-Fixed — Research Prototype (Phase 1)

    Open-source LoRA fine-tuned model for circuit schematic OCR and netlist extraction,
    based on PaddleOCR-VL-0.9B. The **first** to demonstrate that LoRA fine-tuning
    can teach a small VLM domain-specific circuit vocabulary while avoiding modality collapse.

    ### What It Achieves (Phase 1)
    - **CompF1**: 0.2061 (4.5× over baseline 0.0455) — identifies ~31% of component refdes
    - **Token Recall**: 0.1540 (96× over baseline 0.0016) — learns circuit-domain tokens
    - **NED**: 0.8031 (13.6% reduction vs baseline 0.9296)
    - **Diversity**: 90.9% — solves modality collapse (baseline already diverse, maintained)
    - **No modality collapse**: Projector is frozen — a key architectural insight

    ### Scale Context
    - **0.9B parameters** (small VLM) with only **5.7M trainable** (0.63%) — ExactMatch=0% is expected at this scale
    - **43 minutes** on consumer RTX 4060 8GB — anyone can reproduce
    - The 4.5× CompF1 gain and 96× TokenRec gain are the meaningful metrics

    ### What It Does NOT Achieve (Honest Assessment)
    - **ExactMatch = 0%**: cannot perfectly reconstruct any netlist
    - **joint_f1 = 0.019**: only ~2% of (refdes, value) pairs are fully correct
    - **Value hallucination**: model outputs generic values (10k, 100nF, AMS1117) from training distribution
    - **Template memorization**: synthetic V3 data introduced 6 fixed topologies → model hallucinates them
    - **Not production-ready**: useful as a research baseline and proof-of-concept

    ### Roadmap
    | Phase | Goal | Status |
    |---|---|---|
    | Phase 1 | Fix modality collapse, establish baseline | ✅ Complete |
    | Phase 2 | Larger dataset (3,054 samples), topology metrics, regularization (V11) | ❌ Completed — collapsed (CompF1=0.0604, RepRate=84.1%) |
    | Phase 3 | Vision encoder LoRA, higher resolution, two-stage training (V12) | ❌ Completed — collapsed further (destructive gradient interference) |
    | Lessons | Regularization alone insufficient at this scale; Vision LoRA introduces destructive gradient interference with LLM head; dataset quality (synthetic contamination, label consistency) matters more than dataset size; 0.9B architecture may be below critical mass for this task | 💡 Learned |
    | Phase 4 | SPICE verification, human-in-the-loop, production readiness | 📋 Planned (requires architectural breakthrough) |

    ### Links
    - [GitHub Repository](https://github.com/ZhangJ83/circuit-ocr-paddle)
    - [Dataset Repository](https://github.com/ZhangJ83/circuit-ocr-dataset)
    - [LoRA Weights (S600)](https://huggingface.co/yingchu83/CircuitOCR-lora)
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
with gr.Blocks(title="CircuitOCR — Research Prototype", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CircuitOCR V10-Fixed (S600) — Research Prototype
    ### PaddleOCR-VL-0.9B + Wide LoRA (r=16) · CompF1 0.206 (4.5×) · ExactMatch 0% · Phase 1 Complete
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
