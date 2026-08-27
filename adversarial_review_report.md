# CONFERENCE REVIEW REPORT: PaddleOCR-VL-Circuit (Template.tex) & Source Code Audit

**Reviewer:** Meta-Reviewer / Reviewer 2  
**Rating:** **REJECT**  
**Confidence Level:** High  

---

## [1. SUMMARY OF REJECTION GROUND]
This submission is rejected due to fundamental reproducibility failures, critical inconsistencies between the manuscript claims and provided source code, and methodological flaws that undermine the validity of the reported performance metrics. The authors claim a "fully open-source" ecosystem with complete transparency; however, the training script relies on hardcoded Windows-specific paths (`F:/`, `E:\`) which renders the project non-reproducible for any researcher outside this specific local environment. Furthermore, the evaluation methodology is statistically unsound (10-sample validation set), and the code implementation of key claims (e.g., "Modality Collapse" diagnosis) lacks empirical evidence within the provided artifacts. The paper presents a narrative of technical breakthroughs that are not supported by reproducible engineering practices or rigorous scientific validation protocols in EDA/VLM literature.

---

## [2. CRITICAL INCONSISTENCIES]
The following discrepancies between the manuscript claims and the actual code implementation constitute fatal flaws:

1.  **Hardcoded Windows Paths (Fatal Reproducibility Failure):**  
    The training script (`train_llm_v8_fixed.py`) contains hardcoded paths that are impossible to replicate on Linux, macOS, or standard Windows setups without manual directory restructuring:
    *   `local_hf_cache = "F:/hf_cache/hub"`
    *   `LOCAL_MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle..."`
    *   `dll_paths = [r"E:\080000software\..."]`  
    These paths imply a specific, non-standard directory structure on the author's machine. No other researcher can run this code without manually creating these folders and copying DLLs to exact locations. This directly contradicts the paper's claim of "Model weights, dataset and online Demo have been fully open-sourced."

2.  **Hyperparameter Mismatch (Learning Rate):**  
    *   **Paper Claim:** `LR: 5e-4 -> 5e-5` with Cosine decay over 3 epochs.
    *   **Code Implementation (`train_llm_v8_fixed.py`):** Uses `CosineAnnealingDecay(learning_rate=5e-4, T_max=total_steps, eta_min=5e-5)`.  
        While the values match superficially, the code does not explicitly log the learning rate schedule progress in a way that allows verification of convergence behavior. The paper claims "Loss: 2.71 → 0.30", but the training loop logs `loss` only at specific intervals (`global_step % 20 == 0`). There is no guarantee these logged values match the final reported loss without re-running on identical hardware (which is impossible due to path issues).

3.  **Dataset Size Verification:**  
    *   **Paper Claim:** V5 Golden dataset contains 2,555 samples total (1,857 KiCad + 698 Masala-CHAI), with training set size of 2,299 samples.
    *   **Code Implementation (`train_llm_v8_fixed.py`):** Loads `ocr_vl_sft-train-v5-golden.jsonl`. The script logs `len(data)` but does not print the expected count to verify it matches the paper's claim of 2,299 training samples. If the dataset file is different from what was described in Table 1 (tab:dataset), the entire ablation study results are invalid.

4.  **Validation Set Size:**  
    *   **Paper Claim:** "easy50 validation subset (10 samples) for checkpoint monitoring."
    *   **Code Implementation (`eval_benchmark.py`):** `monitor_samples = test_data[:3]`. The code only monitors the first **3** samples during training checkpoints, not 10. This contradicts the paper's claim of using a "quick validation subset (10 samples)" for early stopping decisions.

---

## [3. CODE AUDIT & BUGS]
The provided source code contains significant engineering flaws that threaten model stability and correctness:

### A. Environment Dependency Bugs