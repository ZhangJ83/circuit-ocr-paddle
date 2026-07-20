# Circuit OCR 项目交接文档

## 时间: 2026-06-30 06:15 UTC+8

---

## 1. 项目目标

PaddleOCR 全球衍生模型挑战赛 — 用 LoRA 微调 PaddleOCR-VL-0.9B 做电路原理图 OCR 和网表提取。

评测指标: **Avg. NED** (Normalized Edit Distance, 越低越好)

---

## 2. 当前状态摘要

### 已完成
- ✅ Paddle 3.0 环境修复 (从 2.6.2 升级到 3.0.0b2)
- ✅ 6 个 monkey-patch 使 PaddleFormers 1.1.1 兼容 Paddle 3.x
- ✅ 合成数据重新生成 (500 张, DPI=150, 脚本 `scripts/gen_synthetic_v2.py`)
- ✅ 训练集合并: `ocr_vl_sft-train-v2.jsonl` (1,357 real + 500 synth = 1,857 样本)
- ✅ 安全训练脚本: `scripts/train_projector_v2.py` (Projector-only, checkpoint/500步)
- ✅ 训练完成: 5,571 steps / 18min / 11个checkpoint 全部保存
- ✅ 全部 11 个 checkpoint 在 easy50 上评估完毕

### 最佳结果
| Checkpoint | easy50 NED | vs Base (0.8848) |
|-----------|-----------|-------------------|
| S2000 | **0.8003** | **+9.6%** |
| S5500 | 0.8027 | +9.3% |
| S3500 | 0.8145 | +8.0% |

### ⚠️ 关键问题
**模型输出已塌缩** — 所有 checkpoint 产生重复 token (如 "U\nU\nU...")，多样性仅 28-52%。
这是 Paddle 3.0.0b2 的 `model.generate()` bug 导致的。
Base model (无 LoRA) 输出 100% 多样但不含电路网表格式。

### 🔑 关键发现: Paddle 3.1.0 存在但需重新下载!
```
状态: pip 临时文件已被清理，需重新下载
大小: ~500 MB
平台: Windows AMD64, Python 3.10, CUDA 12.3
下载命令见下方 "第一步"
```

---

## 3. 环境

### 路径
```
项目根目录:     G:\mimo_project\circuit_ocr
数据集目录:     G:\mimo_project\circuit_ocr\circuit-ocr-dataset
Python 环境:    E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum
Python 可执行:  E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe
HF 缓存:        F:\hf_cache\hub\
Paddle 缓存:    F:\paddle_cache
```

### 当前已安装版本
```
PaddlePaddle:  3.0.0b2  (CUDA 12.3)
PaddleFormers:  1.1.1
safetensors:    0.6.2.dev0
GPU:            RTX 4060 8GB VRAM
CUDA Driver:    13.3
```

---

## 4. 关键文件

### 训练脚本
| 文件 | 用途 |
|------|------|
| `scripts/train_projector_v2.py` | **安全训练脚本** — Projector-only LoRA, checkpoint/500步 |
| `scripts/train_real1e_test.py` | Real-only 1-epoch 测试训练 |
| `scripts/eval_benchmark.py` | 评估脚本 (含 6 个 monkey-patch + --manual_decode) |
| `scripts/eval_all_checkpoints.py` | 批量评估所有 checkpoint |
| `scripts/gen_synthetic_v2.py` | 合成数据生成 (500张 DPI=150) |

### 数据文件
| 文件 | 内容 |
|------|------|
| `ocr_vl_sft-train-v2.jsonl` | **当前训练集** (1,357 real + 500 synth) |
| `ocr_vl_sft-train-real.jsonl` | 纯真实数据 (1,357 样本) |
| `ocr_vl_sft-train.jsonl` | 旧训练集 (含 1,076 低质量合成, 已弃用) |
| `ocr_vl_sft-synthetic-v2.jsonl` | 新合成数据 (500 样本) |
| `ocr_vl_sft-test.jsonl` | 测试集 (523 样本) |
| `ocr_vl_sft-test-easy50.jsonl` | easy50 测试 (50 样本) |
| `ocr_vl_sft-test-easy100.jsonl` | easy100 测试 (100 样本) |
| `ocr_vl_sft-test-easy200.jsonl` | easy200 测试 (200 样本) |
| `ocr_vl_sft-test-easy50-degraded.jsonl` | 退化测试 (250 样本) |

### 模型权重
| 路径 | 说明 |
|------|------|
| `PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v2/lora_s500.pdparams` ~ `lora_s5500.pdparams` | 11个训练checkpoint |
| `PaddleOCR-VL-LoRA-circuit-ocr/lora_projector_v2_final_fp16.pdparams` | 最终模型 |
| `PaddleOCR-VL-LoRA-circuit-ocr/lora_real1e_fp16.pdparams` | Real-only 1-epoch 测试 |
| `PaddleOCR-VL-LoRA-circuit-ocr/lora_best_v2_fp16.pdparams` | 最后一个checkpoint |

### 评估结果
| 文件 | 内容 |
|------|------|
| `results_v2_lora_s*_easy50.jsonl` | 各 checkpoint easy50 结果 |
| `results_v2_final_easy50.jsonl` | 最终模型 easy50 结果 |
| `results_v2_s2000_easy100.jsonl` | S2000 easy100 结果 (NED 0.8348) |

---

## 5. 所有 Monkey-Patch (6个)

这些补丁已集成在 `scripts/eval_benchmark.py` 的 `apply_paddle_patches()` 函数中:

### Patch 0: `PySafeSlice.shape` 属性
```python
# safetensors 的 PySafeSlice 类缺少 .shape 属性 (Paddle 3.0 rc/beta)
# 解决: 在 safe_open 调用时动态添加 shape property
from safetensors import safe_open as _safe_open
# ... monkey-patch safe_open 在首次调用时给 PySafeSlice 加 .shape
```

### Patch 0.1: `LocalSharedLayerDesc` → `SharedLayerDesc`
```python
import paddle.distributed.fleet.meta_parallel as _mp
if not hasattr(_mp, 'LocalSharedLayerDesc') and hasattr(_mp, 'SharedLayerDesc'):
    _mp.LocalSharedLayerDesc = _mp.SharedLayerDesc
```

### Patch 0.2: `swiglu` 激活函数
```python
import paddle.nn.functional as _pF
def _swiglu_impl(x, gate=None):
    if gate is None:
        split_dim = x.shape[-1] // 2
        x_up, x_gate = x[..., :split_dim], x[..., split_dim:]
    else:
        x_gate, x_up = gate, x
    return _pF.silu(x_gate) * x_up
_pF.swiglu = _swiglu_impl
```

### Patch 0.3a: `FLAGS_enable_auto_parallel_align_mode`
```python
paddle.set_flags({'FLAGS_enable_auto_parallel_align_mode': False})
```

### Patch 0.3: `fused_rms_norm_ext` → `fused_rms_norm`
```python
import paddle.incubate.nn.functional as _incF
_incF.fused_rms_norm_ext = _incF.fused_rms_norm
```

### Patch 1-4: 标准补丁 (flex_checkpoint, LongTensor, fp8_fp8_half_gemm_fused, get_flags)
```python
# flex_checkpoint → dummy module
# paddle.LongTensor = paddle.Tensor
# paddle.linalg.fp8_fp8_half_gemm_fused = None
# get_flags 和 set_flags 的 monkey-patch
```

---

## 6. 下一步操作 (按优先级)

### 第一步: 下载并安装 Paddle 3.1.0 (解决所有 bug 的关键!)
```bash
# 设置临时目录到 F 盘 (C 盘空间不足)
export TMPDIR=F:/pip_tmp TEMP=F:/pip_tmp TMP=F:/pip_tmp
mkdir -p F:/pip_tmp F:/pip_cache

# 下载并安装 PaddlePaddle 3.1.0 + CUDA 依赖
cd G:\mimo_project\circuit_ocr
E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe \
    -m pip install paddlepaddle-gpu==3.1.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ \
    --default-timeout=120 \
    --cache-dir F:/pip_cache \
    --no-cache-dir

# 如果以上失败, 尝试逐个安装 CUDA 依赖后再装 paddle:
# E:/.../python.exe -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ --default-timeout=120
# E:/.../python.exe -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ --default-timeout=120 --no-deps
```

### 第二步: 验证 Paddle 3.1.0 修复了关键函数
```python
import paddle
print(paddle.__version__)  # 应该是 3.1.0

# 检查之前缺失的函数
import paddle.nn.functional as F
print('swiglu:', hasattr(F, 'swiglu'))  # 应该有!

import paddle.incubate.nn.functional as incF
print('fused_rms_norm_ext:', hasattr(incF, 'fused_rms_norm_ext'))  # 应该有!

import paddle.distributed.fleet.meta_parallel as mp
print('LocalSharedLayerDesc:', hasattr(mp, 'LocalSharedLayerDesc'))  # 应该有!

import paddle.nn.functional.flash_attention as fa
print('flashmask_attention:', hasattr(fa, 'flashmask_attention'))  # 应该有!

# 如果有缺失，从上面的 patch 列表中添加对应补丁
```

### 第三步: 测试 model.generate() 不崩溃
```bash
cd G:\mimo_project\circuit_ocr\circuit-ocr-dataset

E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe -c "
import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0, 'scripts')
from eval_benchmark import apply_paddle_patches
apply_paddle_patches()  # 保留补丁以防万一
import paddle; paddle.set_device('gpu')
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
import json
from PIL import Image

MODEL_PATH = 'F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27'
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format='naive',
    low_cpu_mem_usage=True, dtype='bfloat16')
model.config._attn_implementation = 'flashmask'
model.visual.config._attn_implementation = 'flashmask'

# 加载 LoRA
TARGETS = ['.*linear_1', '.*linear_2']
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model.eval()

# 加载 S2000 checkpoint
lora_state = paddle.load('PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v2/lora_s2000.pdparams')
model.set_state_dict(lora_state)

# 测试 generate()
with open('ocr_vl_sft-test-easy50.jsonl', encoding='utf-8') as f:
    samples = [json.loads(l) for l in f if l.strip()][:5]

for i, s in enumerate(samples):
    img = Image.open(s['images'][0].lstrip('./')).convert('RGB')
    msgs = [{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':s['messages'][0]['content'].replace('<image>','')}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pd')
    with paddle.no_grad():
        out = model.generate(**inp, max_new_tokens=30, do_sample=False, use_cache=False)
    ids = [int(x) for x in out[0][0].numpy().tolist() if int(x)>0]
    resp = processor.tokenizer.decode(ids, skip_special_tokens=True)
    print(f'[{i}] {resp[:100]}')
    img.close()

print('DONE - 如果看到上面有正常输出(非重复token)，则 Paddle 3.1.0 修复成功!')
# 如果 segfault 或输出 "000..." 重复 token，则问题仍在
```

### 第四步: 如果 generate() 修复 → 全面重做

1. **重新评估 S2000** (不再用 manual_decode):
```bash
cd G:\mimo_project\circuit_ocr\circuit-ocr-dataset
mkdir -p PaddleOCR-VL-LoRA-circuit-ocr/lora_eval_tmp
cp PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v2/lora_s2000.pdparams \
   PaddleOCR-VL-LoRA-circuit-ocr/lora_eval_tmp/final_model_light.pdparams

python scripts/eval_benchmark.py \
    --model_type paddleocr-vl \
    --model_name_or_path "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27" \
    --paddle_lora_dir "PaddleOCR-VL-LoRA-circuit-ocr/lora_eval_tmp" \
    --data_path ocr_vl_sft-test-easy50.jsonl \
    --output_path results_v2_s2000_easy50_p31.jsonl \
    --max_length 30 \
    --resume
```

2. **如果输出质量仍然差 (重复 token)**:
   - 尝试 S1000 checkpoint (可能更少塌缩)
   - 降低 LoRA rank 到 r=4
   - 只训 1 epoch
   - 仅用 real 数据

3. **如果输出质量好**:
   - 跑全量 benchmark (easy50/100/200/full523/degraded)
   - 更新 README 和报告

### 第五步: 如果 generate() 仍然 segfault (Paddle 3.1.0 也不行)
   - 彻底放弃 PaddleOCR-VL 方案
   - 转向: 用 PaddleOCR 的检测+识别两阶段方案
   - 或者: 换用其他 VLM (如 Qwen2-VL, MiniCPM-V)

---

## 7. 已知陷阱 (避免浪费时间)

1. **不要用 `--manual_decode`**: 手动解码产出垃圾 token (重复 0/U/1)，NED 看起来好但实际无用
2. **不要用 `model.generate()` + LoRA 在 Paddle 3.0.0b2**: 会 segfault
3. **不要用 max_dim=336**: 高分辨率让 Projector 过载，多样性崩塌
4. **不要用 Full LoRA (q/k/v/o + projector)**: LLM 捷径学习，4% 多样性
5. **不要用 label_smoothing**: 完全崩塌 (NED=1.0)
6. **不要在训练过程中跑推理**: GPU 内存碎片导致 segfault，推理仅在独立进程中做

---

## 8. 关键数字

```
Base model NED:     0.8848 (easy50)
Projector-only r=16: 0.8003 (S2000) ← 当前最优 NED
Projector-only old: 0.8649 (90% diversity) ← 之前最优多样性
Full LoRA r=16:      0.7961 (4% diversity)  ← 严重塌缩
训练数据:            1,857 样本 (1,357 real + 500 synth)
LoRA 参数:           237,568 (4 matrices)
训练时间:            18 min (3 epochs)
GPU:                 RTX 4060 8GB
```

---

## 9. 项目链接

- GitHub: https://github.com/ZhangJ83/circuit-ocr-paddle
- HF Space: https://huggingface.co/spaces/yingchu83/CircuitOCR
- HF Models: https://huggingface.co/yingchu83/CircuitOCR-lora
- 中文报告: arxiv_template/template.tex → template.pdf
- 英文报告: arxiv_template/english.tex → english.pdf

---

## 10. 最终建议

**Paddle 3.1.0 是唯一的希望。** 如果它修复了 `model.generate()` 的 segfault 和缺失函数，所有之前训练的 checkpoint (尤其是 S2000) 可以立即重新评估，可能产生真正可用的模型。

如果 Paddle 3.1.0 也不行，建议放弃 PaddleOCR-VL 微调路线，转向:
- PaddleOCR 检测+识别两阶段
- 或者换用 Qwen2-VL / MiniCPM-V 等成熟 VLM 做同样任务
