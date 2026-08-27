import json
import urllib.request
import os

DATASET_DIR = r'G:\mimo_project\circuit_ocr'
TEX_PATH = f'{DATASET_DIR}/arxiv_template/template.tex'
OUT_PATH = f'{DATASET_DIR}/blind_review_report.md'

def main():
    if not os.path.exists(TEX_PATH):
        print(f"Error: {TEX_PATH} not found!")
        return

    print("Reading LaTeX manuscript...")
    with open(TEX_PATH, 'r', encoding='utf-8') as f:
        tex_content = f.read()

    prompt = (
        "You are an independent, blind peer reviewer for a top-tier EDA/VLM conference (e.g., DAC, ICCAD, NeurIPS). "
        "Your task is to write a highly professional, rigorous, and critical peer review of the following LaTeX manuscript. "
        "Since you are a blind reviewer, you do not know the prior conversation history; evaluate the manuscript solely on its content, logic, and methodology. "
        "\n\n"
        "Here are the links provided by the authors for verification (you should comment on the completeness of this open-source package):\n"
        "- Dataset Repo (mixed): https://github.com/ZhangJ83/circuit_ocr_dataset_final\n"
        "- Synthetic Dataset: https://github.com/ZhangJ83/circuit-ocr-dataset\n"
        "- Code & Report Repo: https://github.com/ZhangJ83/circuit-ocr-paddle\n"
        "- Online Demo: https://huggingface.co/spaces/yingchu83/CircuitOCR\n"
        "- LoRA weights: https://huggingface.co/yingchu83/CircuitOCR-lora\n"
        "\n\n"
        "Please structure your review as follows:\n"
        "1. SUMMARY OF THE PAPER: Briefly summarize the goals, methodology (V5 Golden dataset, LLM-Only LoRA, token shift/BPE boundary fixes), and main results.\n"
        "2. MAIN STRENGTHS: What are the key contributions? (Open sourcing, modality collapse analysis, low-VRAM training, etc.)\n"
        "3. WEAKNESSES & CONCERNS: Be highly critical. Look for potential limitations, such as constraints on 8GB VRAM consumer GPUs, the lack of complete topological netlist evaluation (only string-level NED evaluated), the small test size, or details in the ablation studies.\n"
        "4. CLARIFYING QUESTIONS FOR AUTHORS: List 3-4 specific technical questions.\n"
        "5. OVERALL RATING: Choose one: Strong Accept, Accept, Weak Accept, Weak Reject, Reject. Explain your decision.\n"
        "\n\n"
        f"--- LATEX MANUSCRIPT CONTENT ---\n{tex_content}\n"
    )

    payload = {
        "model": "qwen3.5:4b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 32768
        }
    }

    print("Sending request to local Ollama service...")
    req = urllib.request.Request(
        'http://127.0.0.1:11434/api/generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            review_text = res_data.get('response', '')
            
            # Write to blind_review_report.md
            with open(OUT_PATH, 'w', encoding='utf-8') as out_f:
                out_f.write(review_text)
            
            print(f"Blind review successfully generated and saved to: {OUT_PATH}")
    except Exception as e:
        print("Error calling Ollama:", e)

if __name__ == '__main__':
    main()
