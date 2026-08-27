"""Generate a single comprehensive Gemini batch-check prompt with all 50 samples."""
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUT_DIR = PROJECT_DIR / 'output' / 'gemini_check'
IMAGES_DIR = OUT_DIR / 'images'

header = """你是电路原理图 OCR 数据质量审核专家。你的工作目录下有 50 张电路原理图 PNG。

下面有 50 组数据，每组包含一张图片的路径和对应的 GT（ground truth）标注文本。请【逐组】对照图片检查 GT 质量，然后汇总统计。

## 检查标准（每组 4 类问题）

**A. 漏标（图有，GT无）**
图中可见但 GT 中没出现的文字。特别关注：
- 边框参考网格数字（1,2,3...）和字母（A,B,C...）
- 右下角标题栏各字段（公司名、日期、版本号等）
- 元件标号（R1, C2, U3 等）和参数值（10k, 100nF 等）
- 连线上的网络标签
- 原理图内文字注释

**B. 幻觉（GT有，图无）**
GT 中有但图中看不到/不可见的文字。特别关注：
- 隐藏的引脚号/引脚名
- KiCad 内部标记（#PWR, #FLG）
- 空字段标签（如只有 "Title:" 而无内容，且图中也看不到 "Title:" 字样）

**C. 匹配错误（两方都有但内容不对）**
- 元件标号与参数值配对是否正确（R1→10k，不是 R1→100nF）
- 引脚号与引脚名对应正确
- 特殊字符正确（上划线用 Unicode U+0305，下划线为字面 _）
- 希腊字母/单位（Omega, mu, +/-）

**D. 排序问题**
- 输出顺序是否大致从上到下、从左到右
- 同元件的标号-值-引脚是否聚集

---

## 输出格式

每组输出一个 JSON：

{
  "image": "01.png",
  "verdict": "pass|fail|borderline",
  "summary": "中文一句话",
  "issues": [
    {"type": "missing|hallucination|mismatch|sorting",
     "severity": "critical|major|minor",
     "gt_line": "GT原文(如有)",
     "image_text": "图中文字(如有)",
     "note": "说明"}
  ]
}

全部 50 组检查完毕后输出汇总：

总检查数: 50
pass: N, fail: N, borderline: N
各类问题: missing N, hallucination N, mismatch N, sorting N
严重度: critical N, major N, minor N
需要修复的文件: (列出图片名)

---

"""

# Build the full prompt
parts = [header]

for i in range(1, 51):
    img_abs = (IMAGES_DIR / f"{i:02d}.png").as_posix()
    gt_abs = OUT_DIR / f"{i:02d}.txt"

    gt_raw = gt_abs.read_text(encoding='utf-8')
    # Remove comment header lines (starting with #)
    gt_lines = []
    for line in gt_raw.split('\n'):
        if not line.startswith('#'):
            gt_lines.append(line)
    gt_content = '\n'.join(gt_lines).strip()

    parts.append(f"## 第{i}组\n")
    parts.append(f"图片路径: `{img_abs}`\n")
    parts.append(f"GT标注文本:\n")
    parts.append("```")
    parts.append(gt_content)
    parts.append("```\n")

# Write
output_path = OUT_DIR / 'gemini_batch_check.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(parts))

size_mb = output_path.stat().st_size / 1024 / 1024
print(f"Generated: {output_path}")
print(f"Size: {size_mb:.2f} MB")
print(f"Samples: 50")

# Verify first few entries
content = output_path.read_text(encoding='utf-8')
first_gt_start = content.find('## 第1组')
first_gt_end = content.find('## 第2组')
print(f"Sample 1 chunk: {len(content[first_gt_start:first_gt_end])} chars")
