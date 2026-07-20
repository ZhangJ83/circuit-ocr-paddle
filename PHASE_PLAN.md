# Circuit OCR — Four-Phase Execution Plan

> **Last updated:** 2026-07-04 20:00 | **Author:** ZhangJ83 | **Model:** PaddleOCR-VL-0.9B
> **VRAM budget:** 8 GB (RTX 4060) | **Framework:** Paddle 3.1.0 | **Base model: FIXED (cannot change)**

---

## Phase 1 Retrospective (COMPLETED)

### What Was Achieved

| Metric | Baseline (zero-shot) | V10-Fixed S600 (best) | Delta |
|--------|---------------------|----------------------|-------|
| CompF1 | 0.000 | 0.206 | +0.206 |
| TokenRec | -- | 0.154 | -- |
| NED | -- | 0.803 | -- |
| RepRate | -- | 15.9% | -- |
| Diversity | -- | 90.9% | -- |
| joint_f1 (refdes+value) | -- | 0.019 | -- |

**Key achievements:**
- Fine-tuned PaddleOCR-VL-0.9B with LoRA (vision r=16 + LLM r=16 + projector, 5.7M / 908M = 0.63% params)
- Complete training+evaluation pipeline on Paddle 3.1.0
- S600 is best checkpoint; S800 shows overfitting
- Training time: ~43 min on RTX 4060 (1165 steps, 3 epochs)

### Critical Bottleneck

**Value reading: joint_f1=0.019 vs CompF1=0.206 — a 10x gap.** The model can locate ~20% of components but almost never reads their values correctly. The vision encoder (even with LoRA r=16) can't reliably extract small value text from circuit symbols.


---

## Phase 2 Retrospective (COMPLETED — FAILED)

### Strategy

Fix eval bugs, improve synthetic data (gen_synthetic_v4.py: 1500 images, random wiring), add regularization (dropout=0.1, label_smoothing=0.05, data augmentation, early stopping), train V11 on 3054 samples.

### Results

| Checkpoint | CompF1 | TokenRec | RepRate | NED | Diversity |
|------------|--------|----------|---------|-----|-----------|
| **V10-Fixed S600** | **0.206** | 0.154 | 15.9% | 0.803 | 90.9% |
| V11 S200 | 0.052 | 0.067 | 43.2% | 0.893 | 70.5% |
| V11 S400 | 0.069 | 0.063 | 63.6% | 0.894 | 47.7% |
| V11 S600 | 0.060 | 0.058 | 84.1% | 0.917 | 50.0% |
| V11 S800 | 0.042 | 0.050 | 93.2% | 0.911 | 47.7% |

Training: 800/2290 steps (early stopped), 107 min. RepRate monotonically increased → mode collapse.

### Root Cause

Synthetic-V4 images have fundamentally different visual distribution from real schematics. The model oscillated between two domains → mode collapse. **Synthetic data is not a viable path forward.** All future phases use V9-Pure only (1554 samples).


---

## Phase 3 Plan: Two-Stage Vision+LLM Training (NEXT →)

### Key Discovery

V10-Fixed S600 checkpoint inspection reveals:
- **162 vision LoRA keys** (27 layers × 3 attention matrices × 2 lora_A/B) were trained alongside LLM LoRA
- Vision LoRA was trained at r=16 simultaneously with LLM LoRA r=16
- The CompF1=0.206 result already includes vision LoRA contribution
- Simply "adding vision LoRA" won't help — it's already there

### Hypothesis: Competing Gradients

When vision LoRA and LLM LoRA are trained simultaneously:
1. LLM learns to "compensate" for poor vision features (memorizing patterns)
2. Vision LoRA never gets a clear gradient signal because LLM adapts around it
3. This explains why S800 overfits: LLM memorizes, vision doesn't improve

**Fix:** Two-stage training — train LLM first (Stage 1), then freeze LLM and train vision only (Stage 2).

### Stage 1: LLM LoRA Warmup (ALREADY DONE)

Use V10-Fixed S600 as the starting point. This provides trained LLM LoRA (r=16) + projector LoRA. The vision LoRA weights from V10-Fixed will be **discarded** (they were undertrained due to competing gradients).

### Stage 2: Vision LoRA Only (NEW)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Vision targets** | `model.visual.vision_model.encoder.layers.{0-26}.self_attn.{q,k,v,o}_proj` | 27 layers × 4 matrices |
| **Vision rank** | r=4 | Lower than V10-Fixed's r=16 → less overfitting on 1554 samples |
| **Vision alpha** | 8 | alpha/r = 2.0, same scaling as LLM |
| **Vision LR** | 1e-4 | Higher LR for vision-only training (clearer gradient signal) |
| **LLM LoRA** | Frozen (V10-Fixed S600 weights) | Prevents LLM from adapting around vision |
| **Projector** | Frozen (V10-Fixed S600 weights) | Keeps vision→LLM alignment |
| **Data** | V9-Pure only (1554) | No synthetic |
| **Resolution** | 448px (attempt), fallback 384px | Higher res = smaller text readable |
| **Epochs** | 5 | More epochs since only vision params train |
| **Checkpoint** | Every 200 steps | Standard |

### VRAM Budget

| Component | 384px | 448px |
|-----------|-------|-------|
| Base model (bf16) | ~1.82 GB | ~1.82 GB |
| LLM LoRA r=16 (frozen) | ~11 MB | ~11 MB |
| Vision LoRA r=4 (trainable) | ~3 MB | ~3 MB |
| Optimizer (Adam, vision only) | ~12 MB | ~12 MB |
| Visual tokens | ~182 | ~342 |
| Activations | ~2.5 GB | ~4.5 GB |
| **Total** | **~4.4 GB** ✅ | **~6.4 GB** ⚠️ |

### Script: `train_llm_v12_stage2_vision_lora.py`

Key implementation steps:
1. Load base model, apply LoRA with full TARGETS (vision + LLM + projector)
2. Load V10-Fixed S600 checkpoint → all LoRA weights initialized
3. **Re-initialize vision LoRA weights** to random (discard V10-Fixed vision LoRA)
4. **Freeze** LLM LoRA params + projector LoRA params (`requires_grad=False`)
5. Train only vision LoRA params at higher LR (1e-4)
6. Same quick_inference + compute_comp_f1 validation
7. Save vision LoRA weights separately; merge with V10-Fixed LLM weights for eval

### Resolution Strategy

1. **First attempt:** 448px. If OOM → fallback to 384px.
2. At 384px, try gradient checkpointing for vision encoder if Paddle supports it.
3. If neither works, train at 384px with batch=1, grad_accum=8 (more steps per update).

### Expected Outcomes

| Metric | V10-Fixed S600 | Phase 3 Target | Rationale |
|--------|---------------|----------------|-----------|
| CompF1 | 0.206 | 0.30-0.40 | Better vision features → more refdes identified |
| joint_f1 | 0.019 | 0.05-0.15 | Better vision → values become readable |
| RepRate | 15.9% | < 15% | Clearer visual signal → less hallucination |

### Decision Gates

| Outcome | CompF1 | Action |
|---------|--------|--------|
| **Success** | > 0.30 | Proceed to Phase 4: full fine-tune + higher resolution |
| **Marginal** | 0.20-0.30 | Try vision MLP LoRA (fc1/fc2), or increase vision rank to r=8 |
| **Stagnation** | ≈ 0.20 (no improvement) | Vision LoRA not the bottleneck → Phase 3B: data quality |
| **Degradation** | < 0.15 | Approach failing → Phase 3B: post-processing pipeline |

### Phase 3B (Fallback): Data-Centric + Engineering Approach

If vision LoRA retraining doesn't help:

1. **Data audit:** Manually review 100 training samples for annotation quality (especially values)
2. **Targeted augmentation:** Stronger rotation (±10°), brightness/contrast variation, JPEG compression artifacts
3. **Post-processing pipeline:**
   - Component database matching (known refdes prefixes: R/C/L/D/Q/U/J/Y/F)
   - Value validation (numeric + unit pattern check)
   - Netlist syntax correction (line-by-line format enforcement)
4. **Accept current ceiling:** Package V10-Fixed S600 as the best model, document limitations

### Estimated Time

- Stage 2 training: ~45-90 min (fewer trainable params, more epochs)
- Evaluation: ~45 min per checkpoint
- Total Phase 3: ~3-5 hours


---

## Phase 4 Plan: Full Fine-Tune + Higher Resolution (FUTURE)

Condition: Phase 3 CompF1 > 0.30

### Approach

- Unfreeze last 6 vision encoder layers (full fine-tune, not LoRA)
- Increase resolution to 512px
- Mixed precision training with gradient checkpointing
- Estimated VRAM: ~7.5 GB (tight but possible)

### Target

| Metric | Phase 3 | Phase 4 Target |
|--------|---------|----------------|
| CompF1 | 0.30-0.40 | 0.50+ |
| joint_f1 | 0.05-0.15 | 0.20+ |
| ExactMatch | 0% | 2-5% |
