import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return __import__("os").environ.get("HF_TOKEN")
        @staticmethod
        def save_token(t): pass
        @staticmethod
        def delete_token(): pass
    huggingface_hub.HfFolder = _HfFolder

import gradio as gr

with gr.Blocks(title="CircuitOCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CircuitOCR (Phase 2)")
    with gr.Tabs():
        with gr.TabItem("Inference"): gr.Markdown("Test.")
        with gr.TabItem("Examples"): gr.Markdown("Test.")
        with gr.TabItem("Benchmark"): gr.Markdown("Test.")
        with gr.TabItem("About"): gr.Markdown("Test.")

demo.launch(server_name="0.0.0.0", server_port=7860)
