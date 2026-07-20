# Circuit OCR Handover Document (V2)

## 📌 当前状态 (Current Status)
1. **环境升级**: 已将 Paddle 升级至 `paddlepaddle-gpu==3.1.0`（源于 cu126 镜像），解决了 Paddle 3.0b2 在 Windows 上 LoRA 推理 (`model.generate()`) 时 segfault 崩溃的 Bug。
2. **重训练 V5 正在运行**: 
   - 训练脚本：[train_llm_v5.py](file:///G:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts/train_llm_v5.py)
   - 目标：进行 **LLM 层的 LoRA 微调 (r=8, alpha=16)**，而将 Projector (`mlp_AR`) 和 VisionEncoder (`visual`) 保持 **100% 冻结**。
   - 目前状态：正在后台执行 (Task PID: 43656)。已完成 440/928 步，Loss 已显著下降至 **0.0004**。大约在 22:18-22:20 左右训练完成。

---

## 🔍 关键技术发现 (Key Findings)
- **V2 / V3 模态塌缩 (Modality Collapse) 根因**: 
  - 之前微调 Projector 时，由于数据集规模较小（1,857 样本），微调将 Projector 的预训练对齐权重破坏了。导致任何图像的视觉 Token 都被映射为 Out-of-Distribution 的垃圾向量。
  - LLM 接收到这些垃圾向量后，注意力机制陷入退化状态，因而产生了 `λλλλλ`、`助助助`、`11111` 等重复字符。
  - **之前的 0.8003 NED 指标是虚假的**：由于生成的重复序列极短且带格式（与 ground truth 的 `\n` 对齐），导致 Levenshtein 距离算出来虚低，实际上模型丧失了 OCR 能力。
- **V5 架构升级**: 
  - 冻结 `mlp_AR`，确保预训练的视觉特征空间不受污染（保护 OCR 基础能力）。
  - 只微调 LLM 层的 self-attention，采用 **Manual Token Concatenation** (手动拼接)，在 Tokenizer 上强行对齐 Prompt 与 Label（完美规避 boundary 处的特殊 token 错位 bug，使训练与推理一致）。
  - 学习率安全下调为 `2e-5` (Cosine 衰减至 `2e-6`)，防过拟合与塌缩。

---

## 🚀 接下来要做的事情 (Next Steps)

### Step 1: 检查 V5 训练进度与日志
在新的窗口中，运行以下命令实时查看后台训练任务的日志：
```powershell
# 查看最后 50 行日志
Get-Content -Path "C:\Users\zzz\.gemini\antigravity-ide\brain\c5f210ce-f1b9-48a0-9b22-c636d51c8121\.system_generated\tasks\task-526.log" -Tail 50 -Wait
```
训练完毕后，会在终端打印 `Training complete!`，并在 `G:/mimo_project/circuit_ocr/circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v5/` 留下 `lora_s200.pdparams`, `lora_s400.pdparams`, `lora_s600.pdparams`, `lora_s800.pdparams` 和 `lora_projector_v5_final_fp16.pdparams`。

### Step 2: 评估 V5 检查点以寻找最佳步数
运行我们编写的专属评估脚本：
```powershell
E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe -u G:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts/eval_all_checkpoints_v5.py
```
这会在 `easy50` 测试集上运行所有 v5 检查点并计算真实的 NED 分数。运行完毕后会在终端输出 `BEST: sXXX with NED=0.XXXX`，并把结果存入 `checkpoint_eval_results_v5.json`。
*(由于 V5 使用了对齐的视觉编码，请确认此时预测出来的文字已经能够真正进行电路符号 OCR)*

### Step 3: 全量评估
选定最佳 Checkpoint (例如 `s800` 或 `final`) 后，使用 `eval_benchmark.py` 对全量测试集和退化测试集进行评估：
```powershell
# 1. 评估 easy100
E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe G:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts/eval_benchmark.py --model_type paddleocr-vl --model_name_or_path "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27" --paddle_lora_dir "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/lora_v5_eval" --data_path "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/ocr_vl_sft-test-easy100.jsonl" --output_path "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/results_v5_best_easy100.jsonl"

# 2. 评估 full523
E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe G:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts/eval_benchmark.py --model_type paddleocr-vl --model_name_or_path "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27" --paddle_lora_dir "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/lora_v5_eval" --data_path "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/ocr_vl_sft-test-full523.jsonl" --output_path "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/results_v5_best_full523.jsonl"

# 3. 评估 degraded
E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe G:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts/eval_benchmark.py --model_type paddleocr-vl --model_name_or_path "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27" --paddle_lora_dir "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/lora_v5_eval" --data_path "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/ocr_vl_sft-test-degraded.jsonl" --output_path "G:/mimo_project/circuit_ocr/circuit-ocr-dataset/results_v5_best_degraded.jsonl"
```

### Step 4: 提交准备与清理
1. 导出所选出的最佳 LoRA 权重参数，用于准备 HuggingFace 部署或报告编写。
2. 运行退化样本数据脚本扩充数据集（按需）。
3. 整理 Arxiv 的报告和 README 描述。

---

## 🛠 环境参数与避坑指南 (Environment & Anti-Traps)
- **Python 环境**: `E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe`
- **模型本地路径 (MODEL_PATH)**: `F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27`
- **避坑1**: 评估必须使用 `model.config._attn_implementation = "flashmask"` 和 `model.visual.config._attn_implementation = "flashmask"`，不能使用默认 of eager attention，因为 Paddle 3.1.0 的 eager attention 底层有 `expand()` 参数不匹配 bug 导致崩溃。我们的 `train_llm_v5.py` 与 `eval_benchmark.py` 已默认包含此设置。
- **避坑2**: 绝对不要在推理评估时带 `--manual_decode` 参数，因为新环境已天然支持 `model.generate()` 不崩溃，并且 manual_decode 极易引发低质重复生成。
- **避坑3**: LoRA 的 merging 参数比例 alpha/r 必须一致。当前微调时设定 `r=8, lora_alpha=16`，使得其 scaling 因子为 `2.0`。在 `eval_benchmark.py` 中有硬编码 of `LORA_SCALE = 2.0`，因此此配置与其能够完美无缝融合。
