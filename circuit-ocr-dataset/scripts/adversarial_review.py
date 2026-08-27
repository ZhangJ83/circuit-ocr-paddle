import json
import urllib.request
import os

DATASET_DIR = r'G:\mimo_project\circuit_ocr'
TEX_PATH = f'{DATASET_DIR}/arxiv_template/template.tex'
TRAIN_PATH = f'{DATASET_DIR}/circuit-ocr-dataset/scripts/train_llm_v8_fixed.py'
EVAL_PATH = f'{DATASET_DIR}/circuit-ocr-dataset/scripts/eval_benchmark.py'
DEMO_PATH = f'{DATASET_DIR}/hf_space/app.py'
OUT_PATH = f'{DATASET_DIR}/adversarial_review_report.md'
LOG_PATH = f'{DATASET_DIR}/adversarial_review.log'

def log(msg):
    with open(LOG_PATH, 'a', encoding='utf-8') as lf:
        lf.write(msg + '\n')
    try:
        print(msg.encode('gbk', errors='replace').decode('gbk'))
    except Exception:
        pass

def main():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
        
    log("Reading full, untruncated source files...")
    try:
        with open(TEX_PATH, 'r', encoding='utf-8') as f:
            tex_content = f.read()
        with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
            train_code = f.read()
        with open(EVAL_PATH, 'r', encoding='utf-8') as f:
            eval_code = f.read()
        with open(DEMO_PATH, 'r', encoding='utf-8') as f:
            demo_code = f.read()
    except Exception as e:
        log(f"Failed to read files: {e}")
        return

    repo_structure = ""
    for root, dirs, files in os.walk(f'{DATASET_DIR}/circuit-ocr-dataset/scripts'):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
        for file in files[:40]:
            repo_structure += f"- scripts/{file}\n"
        break

    prompt = (
        "You are an extremely harsh, adversarial, and critical peer reviewer (Meta-Reviewer / Reviewer 2) for a top-tier EDA/VLM conference. "
        "Your goal is to find as many critical flaws, inconsistencies, bugs, and logical gaps as possible to justify REJECTING this paper. "
        "Do not be polite or encouraging. Focus entirely on uncovering issues by auditing the paper text AND the full provided source code (training, evaluation, and demo).\n\n"
        
        "--- PROJECT CONTEXT & COMPLETE SOURCE FILES ---\n\n"
        
        "### 1. REPOSITORY STRUCTURE:\n"
        f"{repo_structure}\n\n"
        
        "### 2. FULL PAPER MANUSCRIPT (template.tex):\n"
        f"{tex_content}\n\n"
        
        "### 3. FULL CORE TRAINING CODE (train_llm_v8_fixed.py):\n"
        f"{train_code}\n\n"
        
        "### 4. FULL CORE EVALUATION CODE (eval_benchmark.py):\n"
        f"{eval_code}\n\n"
        
        "### 5. FULL DEMO APPLICATION CODE (hf_space/app.py):\n"
        f"{demo_code}\n\n"
        
        "--- CRITICAL AUDIT INSTRUCTIONS ---\n"
        "Examine the following specific areas and report all weaknesses:\n"
        "1. PAPER VS CODE INCONSISTENCY: Are there claims in the paper (e.g., hyperparameter values, learning rates, epochs, dataset sizes, model names, target modules) that do not match the actual values in train_llm_v8_fixed.py, eval_benchmark.py, or app.py?\n"
        "2. ENVIRONMENT DEPENDENCY: Auditing the hardcoded Windows paths (E:\\080000software\\..., F:/hf_cache) vs the open-source claim. Does it make reproducing the training/evaluation extremely difficult for other Linux-based researchers?\n"
        "3. CODE BUGS & SAFETY: Audit train_llm_v8_fixed.py and eval_benchmark.py for code issues (e.g., memory leaks, hardcoded path dependencies, error handling failures, token masking bugs, token shift/BPE boundary issues, or OOM risks on 8GB VRAM).\n"
        "4. DEMO CODE AUDIT: Check hf_space/app.py. Does the Gradio interface implement proper preprocessing? Does it use the correct weights? Are there fallback risks?\n"
        "5. METHODOLOGICAL HOLES: What are the fundamental flaws in the scientific methodology? (e.g., validation set size, lack of standard baseline comparison, evaluation metrics weaknesses, lack of topological validation).\n\n"
        "Format your output as a formal conference review report with headings: [1. SUMMARY OF REJECTION GROUND], [2. CRITICAL INCONSISTENCIES], [3. CODE AUDIT & BUGS], [4. DATASET & METHODOLOGY WEAKNESSES], [5. RECOMMENDATION & RATING]."
    )

    payload = {
        "model": "qwen3.5:4b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 28000 # Large context window to load all full files safely
        }
    }

    log("Sending request to local Ollama service...")
    req = urllib.request.Request(
        'http://127.0.0.1:11434/api/generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_data = json.loads(res_body)
            review_text = res_data.get('response', '')
            log(f"Received response of length {len(review_text)}")
            if not review_text.strip():
                log(f"Empty response! Keys: {list(res_data.keys())}")
                if 'error' in res_data:
                    log(f"Ollama error: {res_data['error']}")
            else:
                with open(OUT_PATH, 'w', encoding='utf-8') as out_f:
                    out_f.write(review_text)
                log(f"Adversarial review successfully generated and saved to: {OUT_PATH}")
    except Exception as e:
        log(f"Error calling Ollama: {e}")

if __name__ == '__main__':
    main()
