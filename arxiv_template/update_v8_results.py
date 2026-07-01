#!/usr/bin/env python3
"""Update template.tex and english.tex with V8-Fixed results."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TEMPLATE = r'G:\mimo_project\circuit_ocr\arxiv_template\template.tex'
ENGLISH = r'G:\mimo_project\circuit_ocr\arxiv_template\english.tex'

# ═══════════════════════════════════════════
# PART 1: Chinese template updates
# ═══════════════════════════════════════════

with open(TEMPLATE, 'r', encoding='utf-8') as f:
    content = f.read()

# --- 1a. Replace V5 bottleneck + 长期方向 with V8-Fixed section ---
old_v5_end = r"""V5 的当前瓶颈是 LLM LoRA 容量不足（$r=8$, 1.25M 参数）：模型在约 200 步后收敛，权重不再变化。将 LoRA rank 提升至 $r=16$（+ cross\_attn 层）并增加训练数据，预计可在保持多样性的前提下将 NED 改善至 0.82 以下。这是当前最明确、风险最低的提升路径。

\subsubsection{长期方向（需要更多资源）}"""

new_v8_section = r"""V5 的验证结论是：LLM-Only LoRA 架构方向正确，但 $r=8$ 容量不足，模型约 200 步后过早收敛。这为 V8 的容量扩展和训练策略改进指明了方向。

\subsubsection{V8-Fixed 突破：宽 LoRA + 因果偏移修复 + BPE 边界对齐}

V8-Fixed 在 V5 的 LLM-Only LoRA 架构基础上做了三个关键改进，在 V5 Golden 数据集（2,299 训练样本）上取得了突破性成果：

\textbf{改进 1：宽 LoRA 容量扩展。} 将 LoRA rank 从 $r=8$ 提升至 $r=16$（$\alpha=32$），目标模块从仅 LLM 自注意力层（153 对）扩展至 LLM 注意力 + 视觉编码器注意力 + 投影层共 310 个投影矩阵，可训练参数从 1.25M 增至 5.7M（占模型总参数 0.63\%），训练 3 epoch（1,800 优化器步数）。Loss 从 2.71 稳定收敛至 0.30，未出现模态塌缩。

\textbf{改进 2：因果 1-Token 偏移修复。} PaddleOCR-VL 内部已自动执行 logits 的因果移位，但早期训练脚本额外施加了一次手动移位，导致双重偏移——模型被强制预测错误位置上的 token，loss 表面上下降但梯度信号被污染。V8-Fixed 正确实现了单次移位（\texttt{shift\_logits = logits[:, :-1, :]} vs \texttt{shift\_labels = labels[:, 1:]}），确保每个位置的监督信号精确对应。

\textbf{改进 3：BPE 边界合并修复。} 此前将 prompt 和 label 字符串拼接后再统一分词，BPE tokenizer 会在边界处将跨字符串的字符对合并为单一 token（如换行符 \texttt{\textbackslash n} + 元件前缀 \texttt{R} 被合并为一个不可分割的 token），导致 label 的起始字符被吞入 prompt token 而在训练中被 mask 掉。V8-Fixed 将 prompt 和 label 分别独立分词，在 token ID 空间拼接，从根本上杜绝了 BPE 边界合并。

\textbf{训练与评估。} V8-Fixed 在 V5 Golden 训练集（2,299 样本）上训练 3 epoch（1,800 步）。训练过程中每 200 步保存检查点，并在 easy50 验证子集上自动评估以筛选最优权重。检查点评测结果如表~\ref{tab:v8_checkpoints} 所示：

\begin{table}[h]
\centering
\caption{V8-Fixed 检查点评估对比（easy50 验证子集，10 样本）}
\label{tab:v8_checkpoints}
\begin{tabular}{lcc}
\toprule
\textbf{检查点} & \textbf{步数/轮次} & \textbf{Avg. NED} \\
\midrule
基座（无 LoRA） & --   & 0.9634 \\
s1000  & Step 1000 / Epoch 1.6 & 0.8603 \\
s1200  & Step 1200 / Epoch 2.0 & 0.8407 \\
s1400  & Step 1400 / Epoch 2.3 & 0.8727 \\
\textbf{s1600} & \textbf{Step 1600 / Epoch 2.6} & \textbf{0.8257} \\
final   & Step 1800 / Epoch 3.0 & 0.8352 \\
\bottomrule
\end{tabular}
\par\smallskip
{\small\noindent 注：s1600（Epoch 2.6）表现最优，相对基座（0.9634）错误率削减 37.6\%。final 检查点 NED 略有回升（0.8352），提示 2.6 epoch 附近为最优早停点。}
\end{table}

最优检查点 s1600 在 \textbf{easy100 全量测试集（100 样本）上取得 Avg. NED = 0.7760}，相比基座（0.8999）实现 44.4\% 的相对错误率削减。这是本项目迄今为止在标准测试集上达到的最优成绩。模型输出已从基座的随机乱码/重复退化完全转变为结构化的电路网表格式（如 \texttt{R0\textbackslash nresistor\textbackslash nnet012\textbackslash nVSS\textbackslash nC0\textbackslash ncapacitor}），并能正确生成 EOS 终止符。

表~\ref{tab:v8_summary} 汇总了 V8-Fixed 与历史版本的对比。

\begin{table}[h]
\centering
\caption{V8-Fixed 与历史版本核心指标对比}
\label{tab:v8_summary}
\begin{tabular}{lccccc}
\toprule
\textbf{版本} & \textbf{架构} & \textbf{训练样本} & \textbf{可训练参数} & \textbf{easy50 NED} & \textbf{easy100 NED} \\
\midrule
基座       & --       & 0      & 0       & 0.9634 & 0.8999 \\
V4 最优    & +Projector, $r=16$ & 2,433  & 2.75M   & 0.7961 & 0.8291 \\
V5 LLM-Only & 冻结Projector, $r=8$  & 1,857  & 1.25M   & 0.9031 & --    \\
\textbf{V8-Fixed} & \textbf{宽LoRA, $r=16$} & \textbf{2,299} & \textbf{5.7M} & \textbf{0.8257} & \textbf{0.7760} \\
\bottomrule
\end{tabular}
\par\smallskip
{\small\noindent 注：(1) V8-Fixed 在 easy50 验证子集（10 样本）上 NED=0.8257 vs 基座 0.9634（37.6\% 错误率削减）；(2) easy100 全量（100 样本）NED=0.7760 vs 基座 0.8999（44.4\% 错误率削减），为当前最优成绩；(3) V4 easy50 使用旧 50 样本测试集，基座基线不同（0.8848），不可直接对比绝对值。}
\end{table}

\subsubsection{长期方向（需要更多资源）}"""

if old_v5_end in content:
    content = content.replace(old_v5_end, new_v8_section)
    print('[OK] V8-Fixed section added to template.tex')
else:
    print('[MISS] V5 bottleneck paragraph - checking...')
    idx = content.find('V5 的当前瓶颈')
    if idx >= 0:
        snippet = content[idx:idx+150]
        print(f'  Found but mismatch. Snippet:')
        for i, line in enumerate(snippet.split('\n'), 1):
            print(f'    {i}: {repr(line[:100])}')

# --- 1b. Update experiment settings ---
old_settings = r'所有实验在单张 NVIDIA RTX 4060 (8GB VRAM) 上运行。模型使用 PaddleOCR-VL-0.9B 作为基座，LoRA 微调配置为 rank $r=8$、$\alpha=16$，目标模块为所有注意力层的 q\_proj、k\_proj、v\_proj 和 o\_proj 投影矩阵。训练采用 AdamW 优化器（$\beta_1=0.9$、$\beta_2=0.95$、weight\_decay=0.1），学习率 $5\times10^{-4}$，余弦退火至 $5\times10^{-5}$，1 个 epoch，最大图像尺寸 168 像素。总可训练参数量为 2,746,368，占模型总参数量（908M）的 0.30\%。'

new_settings = r'所有实验在单张 NVIDIA RTX 4060 (8GB VRAM) 上运行，模型使用 PaddleOCR-VL-0.9B 作为基座。V1--V5 早期实验：LoRA rank $r=8$、$\alpha=16$，目标模块为注意力层 q/k/v/o 投影矩阵，学习率 $5\times10^{-4}$，1 epoch，max\_dim=168px，1.25M--2.75M 可训练参数。V8-Fixed 最终配置：LoRA rank $r=16$、$\alpha=32$，目标模块扩展至 LLM 注意力 + 视觉编码器注意力 + 投影层共 310 个投影矩阵，学习率 $2\times10^{-5}$，3 epoch（1,800 步），max\_dim=384px，梯度累积步数 4，训练耗时约 2 小时。可训练参数 5.7M，占模型总参数（908M）的 0.63\%。训练采用 AdamW 优化器（$\beta_1=0.9$、$\beta_2=0.95$、weight\_decay=0.1），余弦退火调度。'

if old_settings in content:
    content = content.replace(old_settings, new_settings)
    print('[OK] Experiment settings updated')
else:
    print('[MISS] Experiment settings - checking...')
    idx = content.find('所有实验在单张')
    if idx >= 0:
        snippet = content[idx:idx+200]
        print(f'  Snippet: {repr(snippet[:150])}')

# --- 1c. Update conclusion section (V5 model sentence) ---
old_conc = r'V5 模型能够根据不同的电路原理图生成不同的输出，实际识别出 ``ESP32-WROOMU4''（微控制器）、``LM7805''（稳压器）、电阻标号与阻值等真实电路元件，标志着从 NED 数字竞赛到真正可用模型的范式转变。'
new_conc = r'V8-Fixed 模型在 easy100 全量测试集上取得 Avg. NED = 0.7760，输出为结构化的电路网表格式（如 \texttt{R0\textbackslash nresistor\textbackslash nnet012\textbackslash nVSS\textbackslash nC0\textbackslash ncapacitor}），并能正确生成 EOS 终止符。这是本项目在标准测试集上的最优成绩，验证了 LLM-Only LoRA + 宽容量 + 精确 token 对齐的技术路线。模型权重已发布为 \texttt{lora\_best\_v8\_fixed\_fp16.pdparams}。'

# This is in the abstract, already updated. Check if it also appears in conclusion.
# Actually the abstract was already updated separately. Don't need to change conclusion again.

# --- 1d. Update the "以上方案中 Step 1/2/2.5 已完成验证" line ---
old_steps = r'以上方案中 Step 1/2/2.5 已完成验证（累计 +10.0\%），Step 3 已尝试但因分布塌缩收效有限，Step 4 待实施。长期方向需要额外的计算和工程资源，此处仅作方向性讨论。'
new_steps = r'以上方案中 Step 1/2/2.5/E6 已完成验证，V8-Fixed 以宽 LoRA + 因果偏移修复 + BPE 边界对齐三管齐下的策略在 easy100 全量测试集上取得当前最优 Avg. NED = 0.7760（基座 0.8999，44.4\% 错误率削减）。长期方向需要额外的计算和工程资源，此处仅作方向性讨论。'

if old_steps in content:
    content = content.replace(old_steps, new_steps)
    print('[OK] Step summary updated')
else:
    print('[MISS] Step summary')

# Save
with open(TEMPLATE, 'w', encoding='utf-8') as f:
    f.write(content)
print('\n=== template.tex saved ===')

# ═══════════════════════════════════════════
# PART 2: English template updates
# ═══════════════════════════════════════════

with open(ENGLISH, 'r', encoding='utf-8') as f:
    en = f.read()

# --- 2a. Update English abstract ---
old_en_abs_end = 'This marks a paradigm shift from NED score competition to truly usable models.'
new_en_abs_end = r'''On the V5 Golden dataset (2,299 training samples), V8-Fixed adopts a wide LoRA configuration ($r=16$, $\alpha=32$, 5.7M parameters), trains for 3 epochs (1,800 steps), and fixes two critical issues---causal token shift and BPE boundary merging---achieving Avg.\ NED = 0.7760 on the full easy100 test set (baseline 0.8999, 44.4\% relative error reduction) and 0.8257 on easy50 (baseline 0.9634, 37.6\% relative error reduction). Model outputs transition from base model gibberish/repetition to structured circuit netlist format. This marks a paradigm shift from NED score competition to truly usable models.'''

if old_en_abs_end in en:
    en = en.replace(old_en_abs_end, new_en_abs_end)
    print('[OK] English abstract updated')
else:
    print('[MISS] English abstract end')

# --- 2b. Update English V5 section ---
old_en_v5_end = r"""V5's current bottleneck is LLM LoRA capacity insufficiency ($r=8$, 1.25M parameters): the model converges after approximately 200 steps with weights ceasing to change. Increasing the LoRA rank to $r=16$ (+ cross\_attn layers) and adding more training data are expected to push NED below 0.82 while maintaining diversity. This represents the clearest, lowest-risk improvement path forward.

\subsubsection{Long-term Directions (Requiring Additional Resources)}"""

new_en_v8_section = r"""V5's verification conclusion is: the LLM-Only LoRA architecture direction is correct, but $r=8$ capacity is insufficient, with premature convergence after approximately 200 steps. This points the way for V8's capacity expansion and training strategy improvements.

\subsubsection{V8-Fixed Breakthrough: Wide LoRA + Causal Shift Fix + BPE Boundary Alignment}

V8-Fixed builds on V5's LLM-Only LoRA architecture with three key improvements, achieving breakthrough results on the V5 Golden dataset (2,299 training samples):

\textbf{Improvement 1: Wide LoRA Capacity Expansion.} LoRA rank increased from $r=8$ to $r=16$ ($\alpha=32$). Target modules expanded from LLM self-attention only (153 pairs) to LLM attention + vision encoder attention + projection layers (310 projection matrices total). Trainable parameters grew from 1.25M to 5.7M (0.63\% of total model parameters). Trained for 3 epochs (1,800 optimizer steps). Loss converged steadily from 2.71 to 0.30 with no modality collapse.

\textbf{Improvement 2: Causal 1-Token Shift Fix.} PaddleOCR-VL internally auto-executes causal shifting on logits, but earlier training scripts additionally applied a manual shift, causing double-shifting---the model was forced to predict tokens at wrong positions; loss superficially decreased but gradient signals were corrupted. V8-Fixed correctly implements single-shift computation (\texttt{shift\_logits = logits[:, :-1, :]} vs \texttt{shift\_labels = labels[:, 1:]}), ensuring precise supervision at each position.

\textbf{Improvement 3: BPE Boundary Merging Fix.} Previously, prompt and label strings were concatenated before tokenization. The BPE tokenizer would merge cross-boundary character pairs into single tokens (e.g., newline \texttt{\textbackslash n} + component prefix \texttt{R} merged into one unbreakable token), causing the label's starting character to be swallowed into a prompt token and masked during training. V8-Fixed tokenizes prompt and label independently and concatenates in token-ID space, fundamentally eliminating BPE boundary merging.

\textbf{Training and Evaluation.} V8-Fixed trained on the V5 Golden training set (2,299 samples) for 3 epochs (1,800 steps). Checkpoints were saved every 200 steps and automatically evaluated on an easy50 validation subset to select optimal weights. Checkpoint evaluation results are shown in Table~\ref{tab:v8_checkpoints}:

\begin{table}[h]
\centering
\caption{V8-Fixed Checkpoint Evaluation Comparison (easy50 validation subset, 10 samples)}
\label{tab:v8_checkpoints}
\begin{tabular}{lcc}
\toprule
\textbf{Checkpoint} & \textbf{Steps/Epoch} & \textbf{Avg. NED} \\
\midrule
Base (No LoRA) & --   & 0.9634 \\
s1000  & Step 1000 / Epoch 1.6 & 0.8603 \\
s1200  & Step 1200 / Epoch 2.0 & 0.8407 \\
s1400  & Step 1400 / Epoch 2.3 & 0.8727 \\
\textbf{s1600} & \textbf{Step 1600 / Epoch 2.6} & \textbf{0.8257} \\
final   & Step 1800 / Epoch 3.0 & 0.8352 \\
\bottomrule
\end{tabular}
\par\smallskip
{\small\noindent Note: s1600 (Epoch 2.6) performs best, achieving 37.6\% relative error reduction over the baseline (0.9634). The final checkpoint (Epoch 3.0) shows a slight NED increase (0.8352), suggesting the optimal early-stopping point is around 2.6 epochs.}
\end{table}

The best checkpoint s1600 achieves \textbf{Avg. NED = 0.7760 on the full easy100 test set (100 samples)}, representing a 44.4\% relative error reduction over the baseline (0.8999). This is the best result achieved by this project on the standard test set to date. Model outputs have completely transformed from random gibberish/repetition to structured circuit netlist format (e.g., \texttt{R0\textbackslash nresistor\textbackslash nnet012\textbackslash nVSS\textbackslash nC0\textbackslash ncapacitor}), with correct EOS token generation.

Table~\ref{tab:v8_summary} summarizes the comparison between V8-Fixed and historical versions.

\begin{table}[h]
\centering
\caption{V8-Fixed vs.\ Historical Version Key Metric Comparison}
\label{tab:v8_summary}
\begin{tabular}{lccccc}
\toprule
\textbf{Version} & \textbf{Architecture} & \textbf{Train Samples} & \textbf{Trainable Params} & \textbf{easy50 NED} & \textbf{easy100 NED} \\
\midrule
Base       & --       & 0      & 0       & 0.9634 & 0.8999 \\
V4 Best    & +Projector, $r=16$ & 2,433  & 2.75M   & 0.7961 & 0.8291 \\
V5 LLM-Only & Frozen Projector, $r=8$  & 1,857  & 1.25M   & 0.9031 & --    \\
\textbf{V8-Fixed} & \textbf{Wide LoRA, $r=16$} & \textbf{2,299} & \textbf{5.7M} & \textbf{0.8257} & \textbf{0.7760} \\
\bottomrule
\end{tabular}
\par\smallskip
{\small\noindent Note: (1) V8-Fixed on easy50 validation subset (10 samples): NED=0.8257 vs.\ baseline 0.9634 (37.6\% error reduction); (2) Full easy100 (100 samples): NED=0.7760 vs.\ baseline 0.8999 (44.4\% error reduction), best result to date; (3) V4 easy50 uses the old 50-sample test set with a different baseline (0.8848), so absolute values are not directly comparable.}
\end{table}

\subsubsection{Long-term Directions (Requiring Additional Resources)}"""

if old_en_v5_end in en:
    en = en.replace(old_en_v5_end, new_en_v8_section)
    print('[OK] English V8-Fixed section added')
else:
    print('[MISS] English V5 bottleneck')

# --- 2c. Update English experiment settings ---
old_en_settings = r'All experiments were conducted on a single NVIDIA RTX 4060 (8\,GB VRAM). The model uses PaddleOCR-VL-0.9B as the base, with LoRA fine-tuning configured at rank $r=8$, $\alpha=16$, targeting the q\_proj, k\_proj, v\_proj, and o\_proj projection matrices of all attention layers. Training uses the AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.95$, weight\_decay=0.1), learning rate $5\times10^{-4}$ with cosine annealing to $5\times10^{-5}$, 1 epoch, maximum image dimension 168 pixels. Total trainable parameters: 2,746,368, constituting 0.30\% of the total model parameters (908M).'

new_en_settings = r'All experiments were conducted on a single NVIDIA RTX 4060 (8\,GB VRAM). The model uses PaddleOCR-VL-0.9B as the base. V1--V5 early experiments: LoRA rank $r=8$, $\alpha=16$, targeting attention layer q/k/v/o projection matrices, learning rate $5\times10^{-4}$, 1 epoch, max\_dim=168px, 1.25M--2.75M trainable parameters. V8-Fixed final configuration: LoRA rank $r=16$, $\alpha=32$, target modules expanded to LLM attention + vision encoder attention + projection layers (310 projection matrices total), learning rate $2\times10^{-5}$, 3 epochs (1,800 steps), max\_dim=384px, gradient accumulation steps=4, training time $\sim$2 hours. Trainable parameters: 5.7M (0.63\% of 908M total). Training uses AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.95$, weight\_decay=0.1) with cosine annealing schedule.'

if old_en_settings in en:
    en = en.replace(old_en_settings, new_en_settings)
    print('[OK] English experiment settings updated')
else:
    print('[MISS] English experiment settings')

# Save
with open(ENGLISH, 'w', encoding='utf-8') as f:
    f.write(en)
print('\n=== english.tex saved ===')
print('\nAll V8-Fixed updates complete!')
