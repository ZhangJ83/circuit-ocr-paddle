<div align="center">

# CircuitOCR
### 基于 PaddleOCR-VL 的电路原理图 OCR 与元件信息识别

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hackathon 10th](https://img.shields.io/badge/PaddleOCR%20Challenge-Finals%20Top%207-FF4444?logo=baidu&logoColor=white)](https://github.com/PaddlePaddle/PaddleOCR/issues/17858)
[![Base Model: PaddleOCR-VL-0.9B](https://img.shields.io/badge/Base%20Model-PaddleOCR--VL--0.9B-blue)](https://github.com/PaddlePaddle/PaddleOCR)
[![Fine-Tuning: LoRA (r=16)](https://img.shields.io/badge/Fine--Tuning-LoRA%20(r%3D16)-green)]()
[![HuggingFace Space](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://huggingface.co/spaces/yingchu83/CircuitOCR)
[![HuggingFace Models](https://img.shields.io/badge/Weights-CircuitOCR--lora-purple)](https://huggingface.co/yingchu83/CircuitOCR-lora)

<p align="center">
  <a href="https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf">📄 中文技术报告 (51页)</a> •
  <a href="https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf">📄 English Report (43p)</a> •
  <a href="https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/slides/beamer_slides.pdf">🎞️ Beamer 演示文稿</a> •
  <a href="https://huggingface.co/spaces/yingchu83/CircuitOCR">🎮 在线 Demo</a> •
  <a href="https://huggingface.co/yingchu83/CircuitOCR-lora">🏋️ 模型权重</a> •
  <a href="https://github.com/ZhangJ83/circuit_ocr_dataset_final">📦 数据集</a>
</p>

</div>

---

> 🏆 **赛事荣誉 | Award & Recognition**
>
> 本项目参与百度主办的 **[第十届飞桨黑客松 · PaddleOCR 全球衍生模型挑战赛 (Hackathon 10th)](https://github.com/PaddlePaddle/PaddleOCR/issues/17858)**，在长尾 OCR 衍生微调赛道中完成全流程开发并进入决赛，获得 **决赛第 7 名（Top 10 获奖衍生模型）**。

---

## 项目背景与说明

电路原理图包含密集的电气符号、小字号参数与引脚编号，通用 OCR 模型（如 PaddleOCR 通用版、Tesseract 等）在此类图像上容易失效。

本项目基于 **PaddleOCR-VL-0.9B**（NaViT 视觉编码器 + ERNIE-4.5-0.3B 语言模型），在单卡 RTX 4060（8GB 显存）环境下，使用 LoRA 微调与合成数据预训练方法，探索电路图文字和元件标号的识别。

当前模型定位于研究原型，主要作为辅助人工标注的参考工具。

---

## 实验结果与对比

### 1. 主模型测试集表现

在 30 样本测试集上的评测结果如下：

| 模型 | 训练数据与配置 | CompF1 (元件标号) | LineAcc (行匹配率) | NED ↓ | 说明 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PaddleOCR-VL-0.9B 基座** | 无微调 (Zero-Shot) | 0.000 | 0.000 | 0.944 | 输出无关的通用文本或模板 |
| **v1 (exp6)** | 1,500 样本 (含 20% 合成文字) | 0.119 | 0.033 | 0.946 | 具备基础标号识别能力 |
| **v2 (Phase 1 预训练)** | 5,000 张合成 KiCad 原理图 | **0.304** | **0.040** | **0.942** | 当前测试集中表现较好的版本 |

### 2. 检查点消融评测（V10-Fixed，测试集 44 样本）

使用 `eval_benchmark_v3.py` 评测不同训练步数检查点的结果：

| Checkpoint | ExactMatch | CompF1 | CompPrec | CompRec | TokenRec | NED ↓ | RepRate ↓ | Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base 基座** | 0% | 0.0455 | 0.0455 | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
| **S400** | 0% | 0.1820 | 0.1862 | 0.2501 | 0.1302 | 0.8298 | 20.5% | 95.5% |
| **S600** | 0% | **0.2061** | 0.2024 | **0.3114** | **0.1540** | **0.8031** | **15.9%** | **90.9%** |
| **S800** | 0% | 0.2080 | 0.2862 | 0.1996 | 0.1191 | 0.8063 | 40.9% | 93.2% |

- S600 步为该组实验中指标平衡较好的检查点。
- S800 步时重复输出比例上升至 40.9%，出现过拟合。
- ExactMatch 均为 0%，表明模型尚无法完整无误地还原整张复杂图纸。

### 3. 失败实验记录

- **V11（正则化 + 合成 V4 数据）**：增加 Dropout 与 Label Smoothing，并引入 Synthetic-V4 数据后，测试集重复率上升至 84.1%，CompF1 下降至 0.060。
- **V12（两阶段单独训练视觉 LoRA）**：冻结语言模型并单独训练视觉编码器 LoRA（学习率 1e-4），导致输出变为无效字符。

---

## 典型识别案例

以下选取技术报告中的 2 个实际测试样本，展示基座模型、v1 与 v2 的完整输出对比：

### 样本 1：微控制器与外设电路 (Sample #0，1446.png，17 个元件)

#### 标注真值 (Ground Truth，前 22 行)
```text
H101
6.00mm
H102
6.00mm
H103
6.00mm
+5V
USB In
3.3V Regulator
Debug
PWR_FLAG
H104
6.00mm
H105
6.00mm
+3.3V
SWD Connector
C101
10uF
Mounting Holes
SWD
J102
USB-C
```

#### 各模型实际输出对比

| 模型 | 输出文本 | 输出分析 |
| :--- | :--- | :--- |
| **基座模型 (Zero-Shot)** | ```text<br>Service \| Name / Service Name \| SERVING<br>Status \| Status \| Status<br>Version \| Version \| Version<br>ID \| ID \| ID<br>Created \| Created \| Created<br>Updated \| Updated \| Updated<br>``` | 输出预训练语料中的通用 API 表格模板，与电路图内容无关。<br>(CompF1 = 0.000) |
| **v1 (exp6, 1.5k 混合)** | ```text<br>1<br>2<br>3<br>4<br>5<br>...<br>23<br>``` | 发生局部模态塌缩，输出机械递增的纯数字序列，未读取图纸文字。<br>(CompF1 = 0.000) |
| **v2 (Phase 1 预训练)** | ```text<br>H101<br>6.00mm<br>H102<br>6.00mm<br>H103<br>6.00mm<br>+5V<br>USB In<br>3.3V Regulator<br>Debug<br>PWR_FLAG<br>H104<br>6.00mm<br>H105<br>6.00mm<br>+3.3V<br>SWD Connector<br>C101<br>10uF<br>Mounting Holes<br>SWD<br>J102<br>USB-C<br>A1 GND<br>A12 GND<br>A4 VBUS<br>A5 CC1<br>``` | 完整匹配 GT 前 21 行内容，正确读取定位孔标号及尺寸（`H101~H105`、`6.00mm`）、电容（`C101 10uF`）、接口（`J102 USB-C`）及网络标号。<br>(CompF1 ≈ 1.0) |

---

### 样本 2：电源输入与开关接口电路 (Sample #1，0385.png，22 个元件)

#### 标注真值 (Ground Truth，前 18 行)
```text
PWR_FLAG
+12V_In
PWR_FLAG
+12V
DC1
DC-005-5A-2.0
1
2
3
SW1
SW_SPDT
1
2
3
SW2
SW_SPDT
1
2
```

#### 各模型实际输出对比

| 模型 | 输出文本 | 输出分析 |
| :--- | :--- | :--- |
| **基座模型 (Zero-Shot)** | ```text<br>This image is a graphic design and does not contain any chart<br>or data that can be extracted into tabular format.<br>The image appears to be a technical drawing or diagram<br>with various labels and annotations.<br>``` | 输出一段图像通用说明文本，未能解析出具体文字与元件信息。<br>(CompF1 = 0.000) |
| **v1 (exp6, 1.5k 混合)** | ```text<br>Model \| Model Name / T-SNE \| T-SNE / PCA \| PCA<br>Data \| Data \| Data<br>Analysis \| Analysis \| Analysis<br>Result \| Result \| Result<br>Output \| Output \| Output<br>``` | 输出机器学习名词表格，未转入电路领域的实体识别。<br>(CompF1 = 0.000) |
| **v2 (Phase 1 预训练)** | ```text<br>PWR_FLAG<br>+12V_In<br>PWR_FLAG<br>+12V<br>DC1<br>DC-005-5A-2.0<br>1<br>2<br>3<br>SW1<br>SW_SPDT<br>1<br>2<br>3<br>SW2<br>SW_SPDT<br>1<br>2<br>3<br>SW3<br>SW_SPDT<br>``` | 前 20 行与真值一致，准确识别出电源接口型号 `DC-005-5A-2.0`、开关类型 `SW_SPDT` 及对应引脚序号 `1, 2, 3`。<br>(CompF1 ≈ 1.0) |

---

## 训练问题排查与修复

在微调过程中，主要处理了以下 6 个技术问题：

1. **因果标签偏移（Causal Token Double-Shift）**：模型内部在计算损失时已包含标签错位处理，若训练脚本再次手动错位会导致监督信号偏移 2 位。
2. **分词边界合并（BPE Boundary Merging）**：直接拼接 Prompt 与 Label 字符串会导致分词器在连接处合并字符，改用分别 Tokenize 后拼接 Token ID 的方式。
3. **Paddle 3.1.0 权重加载**：部分版本下 `model.set_state_dict()` 对 LoRA 返回 `None`，改为通过 `LoRAModel` 遍历参数并使用 `p.set_value()` 注入。
4. **投影层（Projector）冻结**：小样本微调中，解冻 `mlp_AR` 投影层容易导致语言模型退化为输出高频数字，因此保持投影层冻结。
5. **数据混合防塌缩**：在电路图训练集中混入 20% 纯文字图片（阻容表、引脚定义），促使模型依赖视觉输入。
6. **参数转换精度**：统一 LoRA 权重在 float32 与 bfloat16 之间的转换逻辑，避免截断误差。

---

## 数据集说明

数据集独立仓库：[`ZhangJ83/circuit_ocr_dataset_final`](https://github.com/ZhangJ83/circuit_ocr_dataset_final)

- **样本规模**：共 1,820 张（训练集 1,520 张：含 1,200 张 KiCad 图、300 张合成文字图、20 张拍照图；验证集 150 张；测试集 150 张）。
- **标注流程**：采用 7 步检查流程（Schema 检查 $\to$ 文件与路径有效性 $\to$ 控制字符过滤 $\to$ 标号格式正则检查 $\to$ 单位符号标准化 $\to$ KiCad 网表交叉对比 $\to$ 10% 人工抽检）。

---

## 快速使用

### 1. 安装环境

```bash
git clone https://github.com/ZhangJ83/circuit-ocr-paddle.git
cd circuit-ocr-paddle
pip install -r requirements.txt
```

### 2. 模型加载与推理

```python
import paddle
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

# 加载基座模型
model_name = "PaddlePaddle/PaddleOCR-VL"
processor = AutoProcessor.from_pretrained(model_name)
base_model = AutoModelForConditionalGeneration.from_pretrained(model_name, dtype="bfloat16")

# 配置 LoRA 目标层
TARGETS = [
    r'model\.layers\..*q_proj', r'model\.layers\..*k_proj',
    r'model\.layers\..*v_proj', r'model\.layers\..*o_proj',
    r'model\.layers\..*linear_1', r'model\.layers\..*linear_2'
]
lora_config = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
model = LoRAModel(base_model, lora_config)

# 加载 LoRA 权重
lora_state = paddle.load("checkpoints/lora_v2_phase1.pdparams")
model_lora_params = {k: p for k, p in model.named_parameters() if 'lora_' in k}
for k, v in lora_state.items():
    if k in model_lora_params:
        model_lora_params[k].set_value(paddle.cast(v, model_lora_params[k].dtype))

model.eval()

# 推理
image = Image.open("examples/demo_circuit.png").convert("RGB")
inputs = processor(images=image, text="Recognize all circuit components and values in this schematic.", return_tensors="pd")
with paddle.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### 3. 运行测试集评测

```bash
python circuit-ocr-dataset/scripts/eval_benchmark_v3.py \
    --lora_checkpoint checkpoints/lora_s600.pdparams \
    --test_jsonl circuit-ocr-dataset/ocr_vl_sft-test-easy50-pure.jsonl \
    --output_file eval_results.json
```

---

## 引用

```bibtex
@misc{zhang2026circuitocr,
  title={CircuitOCR: LoRA Fine-Tuning PaddleOCR-VL for Circuit Schematic Understanding},
  author={Jianning Zhang},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/ZhangJ83/circuit-ocr-paddle}}
}
```

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。
