#!/usr/bin/env python3
"""Sync english.tex with latest Chinese changes."""
path = r'G:\mimo_project\circuit_ocr\arxiv_template\english.tex'
with open(path, encoding='utf-8') as f:
    content = f.read()

replacements = [
    # Abstract: add 3-epoch + hardware note
    ('adding Projector to LoRA targets further reduces Avg.\\ NED to 0.8271 (+7.0\\%), validating the bottleneck hypothesis. We also outline a performance improvement roadmap, including near-term actionable enhancements such as Projector LoRA extension, multi-epoch training, and diverse decoding, as well as long-term exploration directions such as multi-task decoupling and simulation-based verification.',
     'adding Projector to LoRA targets further reduces Avg.\\ NED to 0.8271 (+7.0\\%). 3-epoch training further lowers it to 0.8197 (+7.8\\%), validating both the Projector bottleneck hypothesis and the gain from multi-epoch training. The current absolute score is constrained by laptop GPU compute; this paper contributes primarily methodological validation and bottleneck analysis.'),

    # Performance section opening
    ('adding Projector LoRA to cumulative improvement elevates to 7.0\\% (0.8271), validating the effectiveness of the bottleneck analysis. The following sections summarize the identified bottlenecks, validated improvements, and future directions.',
     'adding Projector LoRA to cumulative improvement reaches 7.0\\% (0.8271), and 3-epoch training further achieves 7.8\\% (0.8197). The following sections summarize the identified bottlenecks, validated improvements, and future directions.'),

    # Step 1
    ('Step 1: Unlock the Projector bottleneck.',
     'Step 1: Unlock the Projector bottleneck (Validated).'),

    # Step 2
    ('Step 2: Multi-epoch training. Increase the number of epochs under the current configuration (1 $\\to$ 3--5), using cosine annealing learning rate to complete a full schedule within each epoch. Expected training time increase of 3--5$\\times$ (5--8 hours), still within acceptable range for a single GPU. Experiment E2 has already ruled out the pure-resolution hypothesis; more epochs allow the model to more fully learn the circuit vocabulary distribution in the training data.',
     'Step 2: Multi-epoch training (Validated). Increasing epochs from 1 to 3 (7,299 steps, 77 min) on top of Projector LoRA achieved Avg.\\ NED = 0.8197 on easy50, a further 0.9\\% improvement over 1-epoch (0.8271) and a cumulative 7.8\\% over baseline. Diminishing returns suggest limited marginal benefit from further epoch increases, but multi-epoch training still contributes positively to mitigating class collapse.'),

    # Empirical analysis
    ('After Projector LoRA, the 7.0\\% improvement confirms the effectiveness of the bottleneck analysis, though the absolute NED (0.8271) remains far from practical usability. Ongoing 3-epoch training is expected to further improve performance.',
     'After 3-epoch Projector LoRA training, the NED further decreased from 0.8271 to 0.8197 (cumulative +7.8\\% vs. baseline), confirming the gain from multi-epoch training. The 3-epoch improvement over 1-epoch is 0.9\\%, with diminishing returns. The optimal NED of 0.8197 remains far from practical usability (NED $<$ 0.5), severely constrained by laptop GPU compute (see Section 6.3). On larger GPUs, Flash Attention and higher max\\_length/max\\_dim are expected to yield significantly better absolute scores.'),

    # Summary
    ('Original LoRA fine-tuning on 2,433 samples took 100 minutes; all 306 LoRA matrices (153 pairs $\\times$ 2) had their weights updated normally. Adding Projector LoRA further reduced NED to 0.8271 (+7.0\\%), validating the critical role of the vision-language projection layer.',
     'Original LoRA (q/k/v/o) achieved Avg.\\ NED 0.8554 (+3.8\\%) on easy50. Adding Projector LoRA further reduced NED to 0.8271 (+7.0\\%). 3-epoch training achieved 0.8197 (+7.8\\%), validating both the Projector bottleneck and multi-epoch gains. Class collapse at low sample counts was mitigated by full training.'),

    # E4 table row
    ('E4: LoRA layers   \\& Add Projector layers  \\& OOM, incomplete \\& Projector is key bottleneck \\\\',
     'E4: LoRA layers   \\& Add Projector layers  \\& Validated (NED 0.8271, +7.0\\%) \\& Projector is key bottleneck \\\\'),

    # E3 table row
    ('E3: Epoch count   \\& 1 vs.\\ 3 epochs      \\& OOM, incomplete \\& Needs further verification \\\\',
     'E3: Epoch count   \\& 1 vs.\\ 3 epochs      \\& Validated (3-epoch NED 0.8197) \\& Multi-epoch has positive gain \\\\'),

    # Limitations
    ('Limited training scale: Full LoRA fine-tuning was completed on 2,433 training samples (100 minutes), achieving Avg.\\ NED 0.8554 on the easy50 test set (baseline 0.8895, +3.8\\% improvement). Due to 8\\,GB VRAM constraints, inference max\\_length is only 30 tokens, and 3/50 samples failed due to OOM.',
     'Severe compute constraints on experimental results: All experiments were conducted on a single laptop RTX 4060 (8\\,GB VRAM), which severely constrained outcomes: (a) Flash Attention unavailable, requiring manual fallback with ~10x slower inference; (b) max\\_length limited to 30 tokens, far insufficient for complete SPICE netlists (100--500 tokens); (c) max\\_dim limited to 168px, causing loss of fine text details; (d) batch\\_size=1 with no gradient accumulation; (e) Windows + PaddlePaddle 2.6.2 bfloat16 issues added debugging overhead. These constraints resulted in optimal NED of 0.8197 (+7.8\\%). On GPUs with larger VRAM, significantly better absolute scores are expected. This paper contributes primarily methodological validation and bottleneck analysis, not absolute scores.'),

    # Future work
    ('(1) implementing Projector LoRA extension and verifying its NED gain;',
     '(1) benchmarking on additional test tiers (easy100, easy200, full523);'),
    ('(2) completing multi-epoch training and benchmarking on additional test tiers (easy100, easy200, full523);',
     '(2) implementing the topology evaluation protocol (component F1, pin accuracy);'),
    ('(3) implementing the topology evaluation protocol (component F1, pin accuracy) to supplement pure text NED;',
     '(3) introducing diverse decoding strategies (top-p/top-k) to increase output diversity;'),
    ('(4) exploring long-term directions such as multi-task decoupling and simulator-based verification on GPUs with larger VRAM.',
     '(4) exploring higher Projector LoRA ranks ($r=16/32$) and long-term directions on GPUs with larger VRAM.'),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f'OK: {old[:50]}...')
    else:
        print(f'MISS: {old[:50]}...')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
