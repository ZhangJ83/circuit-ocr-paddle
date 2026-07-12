# CircuitOCR 可用性深度审核报告

**审核日期：** 2026-07-12  
**审核方法：** 逐文件代码审查 + API端点实测 + 数据分布量化分析 + 用户旅程追踪  
**核心结论：该项目在可用性上存在系统性缺陷——从 Demo 到代码到模型到数据到权重，每个用户接触点均不可用或严重受损。**

---

## 一、可用性总览：五个用户接触点，五个全部失败

| 用户接触点 | 预期行为 | 实际行为 | 可用性 |
|:---|:---|:---|:---:|
| **Demo** | 上传图片 → 实时OCR识别 | Gradio API崩溃（Python TypeError），Inference返回静态错误字符串 | ❌ |
| **代码** | clone → pip install → 运行 | 109个脚本含487个硬编码Windows路径，非Windows用户无法运行 | ❌ |
| **模型** | 输入电路图 → 输出网表 | ExactMatch=0%，joint_f1=0.019，87%的元件值错误 | ❌ |
| **数据** | 下载 → 使用 | test-v4含31%合成数据（违规），examples.json首条为空 | ❌ |
| **权重** | 下载 → load → 推理 | HF上仍是旧版塌缩权重，最优S600权重未上传；加载需8个monkey-patch | ❌ |

---

## 二、Demo可用性：完全损坏

### 2.1 Gradio API崩溃（实测确认）

对 `https://yingchu83-CircuitOCR.hf.space/gradio_api/info` 的HTTP请求返回**Python Traceback错误**而非正常JSON：

```
TypeError: argument of type 'bool' is not iterable
  File "gradio_client/utils.py", line 880, in get_type
    if "const" in schema:
```

**这意味着：** Gradio后端API已完全崩溃。前端HTML是构建缓存残留，但任何API调用（包括加载Examples、Benchmark数据）都会失败。用户看到的页面可能不完整或无法交互。

### 2.2 requirements.txt不完整

Space的 `requirements.txt` 仅包含两行：
```
pillow
numpy
```

缺少 `gradio`、`huggingface_hub`、`paddlepaddle`、`paddleformers` 等核心依赖。Space依赖HuggingFace默认环境中的预装Gradio，但版本不匹配（`gradio==5.3.0` 与 `gradio_client` 内部API不兼容）导致了上述崩溃。

### 2.3 Inference Tab：功能性欺骗

`app.py:46` 的 Inference Tab 按钮点击后仅返回硬编码字符串：
```python
"⚠️ Live inference unavailable on CPU-only free tier.\n"
"This is a research prototype — see Examples tab for\n"
"pre-computed results, or run locally with GPU."
```

**用户上传任何图片都会得到相同的错误提示。** 这不是"模型推理失败"，而是"根本没有尝试推理"。按照OpenAI/Anthropic的产品标准，这属于**功能性欺骗**——UI暗示用户可以上传图片获得结果，但实际无法执行。

### 2.4 Examples数据质量问题

`examples.json` 中9条示例，**第1条完全为空**（无image、无verdict、无note），会导致Demo在渲染时出现空元素。

其余8条示例中：4条标注为🔴 FAILURE，3条🟡 PARTIAL，0条完全正确。虽然诚实性值得肯定，但用户看到的是"这个模型几乎对所有示例都失败了"。

---

## 三、代码可用性：无法复现

### 3.1 硬编码路径（487处，109个脚本）

所有核心脚本均硬编码了Windows特定路径，**非作者本人环境无法运行**：

```
F:/hf_cache/hub                                — HF模型缓存
F:/paddle_cache                                — Paddle缓存
E:\080800software\080900_Miniconda\...          — Python环境
E:\080800software\080900_Miniconda\...\torch\lib — CUDA DLL
```

**影响范围：**

| 脚本 | 硬编码路径数 | 重要程度 |
|:---|:---:|:---|
| `train_llm_v11_regularized.py` | 16 | 🔴 核心训练 |
| `eval_benchmark_v2.py` | 12 | 🔴 核心评估 |
| `eval_capability_win.py` | 11 | 🔴 核心评估 |
| `train_llm_v10_fixed.py` | 9 | 🔴 Phase 1最优训练 |
| `train_llm_v12_stage2_vision_lora.py` | 9 | 🔴 Phase 3训练 |
| `eval_benchmark.py` | 8 | 🔴 通用评估 |
| `eval_benchmark_v3.py` | 7 | 🔴 Phase 1正式评估 |
| 其余102个脚本 | 1-8 | 各种 |

**即使Windows用户，如果路径不同（如D盘而非F盘），也无法运行。**

### 3.2 Monkey-Patch依赖（8个独立补丁）

运行核心评估脚本需要8个兼容性补丁（[eval_benchmark.py:51-170](circuit-ocr-dataset/scripts/eval_benchmark.py#L51-L170)）：

| # | 补丁 | 原因 |
|---|------|------|
| 1 | `paddle.LongTensor = paddle.Tensor` | Paddle 3.1.0移除了LongTensor |
| 2 | `flex_checkpoint` dummy module | Paddle 3.1.0 API变更 |
| 3 | `PySafeSlice.shape` | safetensors兼容性 |
| 4 | `LocalSharedLayerDesc→SharedLayerDesc` | Paddle 3.0 rc/beta缺失 |
| 5 | `swiglu` 自定义实现 | Paddle版本缺失 |
| 6 | `fused_rms_norm` 别名 | Paddle API变更 |
| 7 | `get_flags` 字符串参数兼容 | Paddle API变更 |
| 8 | `reshape` API参数顺序 | Paddle 3.1.0变更 |

**这些补丁表明：项目依赖于一个极其脆弱的PaddlePaddle环境。作者在Windows/WSL2上经过大量调试才使代码运行，但任何稍有差异的环境（不同Paddle版本、不同Python版本、Linux原生）都会导致新的兼容性问题。**

### 3.3 依赖声明不完整

`requirements.txt` 缺少关键依赖：
- `paddleformers`（训练/推理核心库）——未列出
- `gradio`（Demo）——未在requirements.txt中列出
- `python-Levenshtein`（NED计算）——未列出
- `Levenshtein`（评估脚本直接import）——未列出

### 3.4 Quick Start不可执行

README中的Quick Start：
```bash
pip install paddlepaddle-gpu paddleformers gradio pillow
cd circuit-ocr-dataset/scripts
python eval_benchmark_v3.py \
    --data_path ../ocr_vl_sft-test-easy50-pure.jsonl \
    --lora_checkpoint ../PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v10_fixed/lora_s600.pdparams
```

**按此操作执行的结果：**
1. `paddleformers` 在PyPI上不存在（PaddlePaddle生态的私有包）→ **安装失败**
2. 即使安装成功，`eval_benchmark_v3.py` 第34行硬编码 `MODEL_PATH = r"F:\hf_cache\..."` → **路径不存在，崩溃**
3. 即使路径修复，还需配置8个monkey-patch的环境变量 → **环境不匹配，崩溃**

---

## 四、模型可用性：输出不可用

### 4.1 核心性能指标

| 指标 | 最优值 (S600) | 实际含义 |
|:---|:---:|:---|
| ExactMatch | **0%** | 无法完整重建任何一张电路图的网表 |
| CompF1 | 0.2061 | 仅识别~31%的元件标号（如R1、C2） |
| joint_f1 | **0.019** | 仅~2%的（元件标号, 参数值）对完全正确 |
| value_acc | 0.133 | **87%的元件参数值是错误的** |
| TokenRec | 0.1540 | 仅15%的token能匹配到ground truth |

### 4.2 实际输出质量（来自Demo Examples）

| 示例 | 预测 | 问题 |
|:---|:---|:---|
| Flicker-noise | (Pro micro + R2/100k + GND) 重复6次 | 🔴 模板重复塌缩 |
| digital_simple | AMS1111电压稳压器模式重复4次 | 🔴 完整模板幻觉 |
| analog_simple | 仅输出"1\n2" | 🔴 完全塌缩 |
| sot2dip | 识别J1,J2,J3但数值错误 | 🟡 部分可用 |
| cat-pcb | 正确识别BT1但幻觉BT2/BT3 | 🟡 部分可用 |

**结论：模型的输出对任何实际应用场景（辅助标注、自动网表提取、电路验证）均不可用。**

### 4.3 最优权重未发布

HF Model仓库 (`yingchu83/CircuitOCR-lora`) 上仍是旧版权重：

| HF上的权重 | 版本 | 状态 |
|:---|:---|:---|
| `lora_projector_r16_fp16.pdparams` | V3 Full LoRA | ❌ 已塌缩（diversity 4%） |
| `lora_projector_only_fp16.pdparams` | V3 Projector-only | ❌ 性能差 |
| `lora_best_v8_fixed_fp16.pdparams` | V8 | ⚠️ 旧版 |
| `lora_v9_pure_final_fp16.pdparams` | V9 | ⚠️ 旧版 |

**V10-Fixed S600（Phase 1最优，CompF1=0.2061）的5个checkpoint（各11MB）仅存在于本地，未上传到HuggingFace。** 用户下载到的权重是已知塌缩或性能较差的旧版本。

---

## 五、数据可用性：合规问题 + 质量缺陷

### 5.1 测试集含合成数据（违规）

`ocr_vl_sft-test-v4.jsonl`（155样本）中，**48个样本（31.0%）来自合成数据**：
```
data/synthetic_v3/synth_v3_0417.png
data/synthetic_v3/synth_v3_0071.png
data/synthetic_v3/synth_v3_0060.png
...（共48个）
```

这直接违反比赛规定"测试集中不能出现合成数据"。虽然当前Phase 1使用的`easy50-pure`不含合成数据，但`test-v4.jsonl`仍存在于仓库中，可能被误用。

### 5.2 训练集合成数据比例过高

| 训练集版本 | 总样本 | 合成数据 | 占比 |
|:---|:---:|:---:|:---:|
| V11 | 3,839 | 2,000 | **52.1%** |
| V9-Pure | 1,554 | 457 | 29.4% |
| V5-Golden | 2,299 | 457 | 19.9% |

V11训练集中超过一半是合成数据，这是导致其塌缩（CompF1=0.0604, RepRate=84.1%）的重要原因之一。

### 5.3 测试集缺乏真实场景多样性

测试集 `data/test/` 的2,327个文件中：
- PNG: 1,134（96.5%）— 以EDA工具渲染图为主
- JPG/JPEG: 29（2.5%）— 极少数真实照片

**缺少：** 手机拍照屏幕（摩尔纹）、纸质原理图扫描、不同光照条件、工程现场照片、倾斜/透视变形等真实场景。组委会明确要求"带有拍照、倾斜、光照变化、屏幕拍照等不同真实场景情况"。

### 5.4 examples.json数据缺陷

- 第1条示例完全为空（无image、verdict、note）
- 所有示例的image路径为 `./data/test/...`，这些路径仅在Demo的Docker环境中有效，本地无法查看

---

## 六、权重可用性：加载流程极其复杂

### 6.1 非标准加载流程

加载LoRA权重需要**9步操作**，且使用非标准API：

```python
# 步骤1-3：环境补丁
paddle.LongTensor = paddle.Tensor  # 补丁1
sys.modules['paddle.distributed.flex_checkpoint'] = dummy  # 补丁2

# 步骤4：加载基座模型
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format='naive',
    low_cpu_mem_usage=True, dtype="bfloat16")

# 步骤5：应用LoRA wrapper（非标准！）
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
model = LoRAModel(model, lc)

# 步骤6：加载checkpoint
lora_state = paddle.load("lora_s600.pdparams")

# 步骤7-8：手动逐参数设置（不能用set_state_dict！）
model_lora_params = {k: p for k, p in model.named_parameters() if 'lora_' in k}
for ckpt_key, ckpt_value in lora_state.items():
    p = model_lora_params[ckpt_key]
    ckpt_tensor = paddle.cast(ckpt_value, p.dtype)  # float16精度损失
    p.set_value(ckpt_tensor)  # 步骤9

# ❌ model.set_state_dict(lora_state) — 静默失败，返回None！
```

**对比标准HuggingFace/PyTorch LoRA加载：**
```python
# 标准流程（2步）
model = PeftModel.from_pretrained(base_model, "user/lora-weights")
result = model.generate(image)
```

### 6.2 权重版本混乱

存在多个版本的LoRA权重，但缺少明确的版本对照表：

| 位置 | 版本 | 状态 |
|:---|:---|:---|
| `hf_model/` (准备推送) | V3 Full LoRA + Projector-only | ❌ 旧版/塌缩 |
| `hf_model_clone/` (已推送) | V8 + V9 + V3 | ❌ 旧版 |
| `checkpoints_v10_fixed/` | **V10 S600最优** | ✅ 但未推送HF |
| `checkpoints_v11_regularized/` | V11 | ❌ 塌缩 |
| `checkpoints_v12_stage2/` | V12 | ❌ 塌缩 |
| `lora_weights_f32.pdparams` | 未知版本 | ❓ 无文档 |

---

## 七、组委会反馈与可用性问题的关联

| 组委会反馈 | 可用性根因 | 严重程度 |
|:---|:---|:---:|
| "demo目前处于无法使用的状态" | Gradio API崩溃（TypeError）+ Inference Tab无实时推理 | 🔴 致命 |
| "没有看到训练、测试代码与脚本" | 代码存在但**硬编码路径导致无法运行**，可能组委会尝试运行失败后得出此结论 | 🔴 致命 |
| "测试集中不能出现合成数据" | `test-v4.jsonl`含31%合成数据，仍存在于仓库中 | 🔴 违规 |
| "请收集真实场景图像" | 测试集96.5%为PNG渲染图，缺少真实拍照/扫描场景 | 🟡 严重 |

---

## 八、按可用性维度重新评分

| 可用性维度 | 评分 | 依据 |
|:---|:---:|:---|
| **Demo可用性** | **1/10** | Gradio API崩溃，Inference返回静态错误，无实时推理 |
| **代码可复现性** | **1/10** | 487个硬编码路径，8个monkey-patch，非作者环境无法运行 |
| **模型输出可用性** | **1/10** | ExactMatch=0%，87%值错误，输出不可用于任何实际任务 |
| **数据合规性** | **3/10** | 测试集含合成数据（违规），训练集合成过半 |
| **权重可用性** | **2/10** | 最优权重未发布，加载需9步非标准流程 |
| **文档可用性** | **4/10** | Quick Start不可执行，但README中英文覆盖较全 |
| **综合可用性** | **2/10** | 项目在工程完整性上达标，但在用户实际使用层面完全不可用 |

---

## 九、修复路线图（按可用性优先级）

### P0：让Demo可用（1-2天）

1. **修复Gradio版本兼容性**：将 `sdk_version: 5.3.0` 降级到 `4.44.0` 或升级到 `5.9.0` 以修复 `json_schema_to_python_type` 的bool类型崩溃
2. **补充requirements.txt**：添加 `gradio`、`huggingface_hub`、`python-Levenshtein` 等实际依赖
3. **修复examples.json**：移除空的第一条记录，确保所有image路径在Space环境中可访问
4. **部署实时推理**：申请HuggingFace GPU Space（T4 $0.60/h），或使用CPU推理（接受较慢速度但至少可运行）

### P1：让代码可复现（3-5天）

5. **移除所有硬编码路径**：使用环境变量 `PADDLE_MODEL_PATH`、`HF_HOME` 等替代
6. **编写环境配置脚本**：`setup_env.sh` 自动检测并配置PaddlePaddle环境
7. **Docker化**：提供Dockerfile，确保一键可运行
8. **修复Quick Start**：使README中的命令可以实际执行

### P2：让数据合规（1-2天）

9. **移除或标记废弃** `ocr_vl_sft-test-v4.jsonl`
10. **收集真实场景测试集**：50-100张照片（手机拍摄屏幕、纸质扫描、工程现场），人工标注+交叉验证
11. **在README中明确标注**：哪些测试集可用于正式评估，哪些已废弃

### P3：让权重可用（1-2天）

12. **上传V10-Fixed S600到HuggingFace**：替换当前旧版/塌缩权重
13. **更新模型卡**：包含完整的性能指标、加载方式、使用限制
14. **提供简化加载脚本**：封装9步操作为 `load_circuit_ocr()` 函数

---

## 十、最终结论

**该项目在"工程完整性"（代码量、实验设计、数据工程）上表现扎实，但在"可用性"上存在系统性崩溃。** 五个用户接触点（Demo、代码、模型、数据、权重）全部不可用或严重受损。

**核心矛盾：** 项目在README中声称"Quick Start: 3行命令即可运行"，但实际上：
- Demo API已崩溃（Gradio版本不兼容）
- 代码487个硬编码路径导致非作者环境无法运行
- 模型输出ExactMatch=0%无法用于任何实际任务
- 数据含违规合成数据
- 最优权重未发布到HuggingFace

**组委会的4条反馈实质上都指向同一个根因：可用性。** 修复Demo崩溃、移除硬编码路径、清理合成数据、收集真实场景测试集，这四项是让项目从"不可用"到"可演示"的最低门槛。

**建议：** 在可用性问题解决之前，不要在论文/实验层面继续投入。一个ExactMatch=0%但能正常运行的Demo，比一个CompF1=0.5但无人能运行的Demo更有价值。

---

*审核人：Claude Code (DeepSeek-V4-Pro) | 审核日期：2026-07-12 | 实测：Demo API状态、代码路径分析、数据分布量化、权重版本比对*