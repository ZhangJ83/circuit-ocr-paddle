"""PaddleOCR-VL LoRA Training — Fixed Script (train_lora_final.py).

Key differences from train_lora_pf.py:
  - max_dim 168 (faster training, still readable)
  - 1 epoch, grad_accum=1
  - Uses all 2433 training samples by default
  - No silent try/except in compute_loss — errors surface immediately
  - Manual CE loss with shift-logits (no shape mismatch)
  - LoRA r=8, alpha=16 on q/k/v/o projections
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "HF_HOME": "F:/hf_cache/hub",
    "PADDLE_HOME": "F:/paddle_cache",
    "HF_HUB_CACHE": "F:/hf_cache/hub",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "FLAGS_allocator_strategy": "auto_growth",
})

# Allow importing apply_paddle_patches from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
import platform
if platform.system() == "Windows":
    DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
    MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
else:
    DATASET_DIR = "/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset"
    MODEL_PATH = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
LOG_FILE = f"{OUTPUT_DIR}/training_lora_final.log"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Manual cross-entropy loss (no shape mismatch — avoids Paddle 2.6.2 crash)
# ---------------------------------------------------------------------------
def compute_loss(model, processor, sample, max_dim: int = 168):
    """Compute CE loss for one training sample using manual shift-logits.

    Tokenizes the full (image + query + label) sequence, then builds a label
    tensor where prompt tokens are -100 (ignored) and assistant tokens keep
    their ids.  Logits are shifted by 1 so token[t] predicts token[t+1].
    """
    from PIL import Image
    from io import BytesIO

    query = sample["messages"][0]["content"]
    label = sample["messages"][1]["content"]

    # Resolve image path
    img_path = sample["images"][0]
    if not img_path.startswith("/"):
        img_path = f"{DATASET_DIR}/{img_path.lstrip('./')}"
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load & resize image (max_dim controls training speed vs. readability)
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Re-encode to JPEG to strip metadata (in-memory round-trip)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    image = Image.open(buf)

    # --- Tokenize prompt-only (to find the cut point) ---
    prompt_msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query.replace("<image>", "")},
        ],
    }]
    prompt_inputs = processor.apply_chat_template(
        prompt_msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pd",
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # --- Truncate label to control VRAM (long labels = long sequence = O(n²) attention) ---
    max_label_chars = 200
    if len(label) > max_label_chars:
        label = label[:max_label_chars]

    # --- Tokenize full sequence (prompt + label) ---
    full_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query.replace("<image>", "")},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": label}]},
    ]
    full_inputs = processor.apply_chat_template(
        full_msgs,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pd",
    )

    # --- Build labels: -100 for prompt, real ids for assistant ---
    import paddle
    full_ids = full_inputs["input_ids"]
    labels = paddle.full_like(full_ids, -100, dtype=full_ids.dtype)
    labels[0, prompt_len:] = full_ids[0, prompt_len:]

    # --- Forward pass (no labels kwarg — avoids Paddle 2.6.2 crash) ---
    out = model(**full_inputs)
    logits = out[0] if isinstance(out, (tuple, list)) else out  # [1, seq_len, vocab]

    # --- Shift: logits[t] predicts token[t+1] ---
    shift_logits = paddle.cast(logits[:, :-1, :], "float32")  # paddle.cast preserves grad
    shift_labels = labels[:, 1:]        # [1, seq_len-1] (int64, required for CE)

    # --- Mask out ignored positions ---
    mask = paddle.cast((shift_labels != -100), "float32")
    shift_labels_clamped = paddle.where(
        shift_labels != -100,
        shift_labels,
        paddle.zeros_like(shift_labels),  # int64 zeros
    )

    ce = paddle.nn.functional.cross_entropy(
        shift_logits.reshape([-1, shift_logits.shape[-1]]),
        shift_labels_clamped.reshape([-1]),
        reduction="none",
    ).reshape(shift_labels.shape)

    loss = (ce * mask).sum() / mask.sum().clip(min=1)

    # Cleanup
    image.close()
    paddle.device.cuda.empty_cache()

    return loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="PaddleOCR-VL LoRA Training (fixed — errors surface)"
    )
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_eval_samples", type=int, default=20)
    ap.add_argument(
        "--data_size", type=int, default=None,
        help="Limit training samples (None = all 2433)",
    )
    ap.add_argument(
        "--max_dim", type=int, default=168,
        help="Max image dimension after resize (168 = fast, 224 = detailed)",
    )
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Compatibility patches — MUST run before importing PaddleFormers
    # -----------------------------------------------------------------------
    log("Applying Paddle compatibility patches...")
    apply_paddle_patches()
    log("[Patches] OK")

    import paddle
    paddle.set_device("gpu")
    log(f"GPU: {paddle.device.cuda.get_device_name(0)}")

    from paddleformers.transformers import (
        AutoModelForConditionalGeneration,
        AutoProcessor,
    )
    from paddleformers.peft.lora import LoRAConfig, LoRAModel

    # -----------------------------------------------------------------------
    # Load processor
    # -----------------------------------------------------------------------
    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # -----------------------------------------------------------------------
    # Load base model
    # -----------------------------------------------------------------------
    log("Loading base model...")
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        convert_from_hf=True,
        load_checkpoint_format="naive",
        low_cpu_mem_usage=True,
        dtype="float32",
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    log("Base model loaded")

    # -----------------------------------------------------------------------
    # Apply LoRA
    # -----------------------------------------------------------------------
    log(f"Applying LoRA: r={args.rank}, alpha={args.alpha}")
    lora_config = LoRAConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=[".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj"],
    )
    model = LoRAModel(model, lora_config)
    model.mark_only_lora_as_trainable()
    # Patch: save_pretrained needs model.model.full()
    if not hasattr(model.model, 'full'):
        def _full(state, structured_name='', prefix='', *args, **kwargs):
            for n, p in model.model.named_parameters(prefix=prefix):
                yield n, p
        model.model.full = _full
    model.train()

    total_params = sum(p.size for p in model.parameters())
    trainable_params = sum(
        p.size for p in model.parameters() if not p.stop_gradient
    )
    log(
        f"Params: Total={total_params:,}  "
        f"Trainable={trainable_params:,}  "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    train_path = f"{DATASET_DIR}/ocr_vl_sft-train.jsonl"
    eval_path = f"{DATASET_DIR}/ocr_vl_sft-test.jsonl"

    with open(train_path, encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    with open(eval_path, encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f if line.strip()]

    if args.data_size is not None and args.data_size < len(train_data):
        train_data = train_data[:args.data_size]

    eval_subset = eval_data[:args.max_eval_samples]
    log(
        f"Data: Train={len(train_data)}  "
        f"Eval={len(eval_subset)}  "
        f"max_dim={args.max_dim}"
    )

    # -----------------------------------------------------------------------
    # Optimizer & scheduler
    # -----------------------------------------------------------------------
    spe = max(1, len(train_data) // args.grad_accum)
    total_steps = args.epochs * spe
    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=total_steps, eta_min=5e-5
    )
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=[p for p in model.parameters() if not p.stop_gradient],
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
    )

    # -----------------------------------------------------------------------
    # Save initial LoRA weights for post-training comparison
    # -----------------------------------------------------------------------
    init_lora = {}
    for k, v in model.named_parameters():
        if "lora_A" in k or "lora_B" in k:
            init_lora[k] = v.numpy().copy()
    log(f"Saved {len(init_lora)} initial LoRA weights for comparison")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    history = []
    global_step = 0
    best_eval = float("inf")
    t0 = time.time()

    log("=" * 60)
    log(
        f"Training: r={args.rank}  alpha={args.alpha}  "
        f"epochs={args.epochs}  samples={len(train_data)}  "
        f"grad_accum={args.grad_accum}  max_dim={args.max_dim}  "
        f"lr={args.lr}"
    )
    log(f"Steps per epoch: {spe}  Total steps: {total_steps}")
    log("=" * 60)

    for epoch in range(args.epochs):
        ep_start = time.time()
        step_loss_accum = 0.0
        random.shuffle(train_data)
        log(f"[Epoch {epoch + 1}/{args.epochs}] Starting ({len(train_data)} samples)...")

        for idx, sample in enumerate(train_data):
            loss = compute_loss(model, processor, sample, args.max_dim)

            # Log first few sample losses for sanity check
            if idx < 3:
                log(f"  Sample {idx}: loss={loss.item():.6f}")

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            step_loss_accum += loss.item()

            # Step optimizer when grad_accum steps are reached (or at last sample)
            if (idx + 1) % args.grad_accum == 0 or (idx + 1) == len(train_data):
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                global_step += 1

                avg_loss = step_loss_accum / args.grad_accum
                step_loss_accum = 0.0

                elapsed_m = (time.time() - t0) / 60
                eta_m = (
                    (elapsed_m / global_step) * total_steps - elapsed_m
                    if global_step > 0
                    else 0.0
                )
                history.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(avg_loss),
                    "lr": optimizer.get_lr(),
                })

                log(
                    f"[E{epoch + 1} S{global_step:4d}/{total_steps}]  "
                    f"loss={float(avg_loss):.4f}  "
                    f"lr={optimizer.get_lr():.2e}  "
                    f"elapsed={elapsed_m:.0f}m  ETA={eta_m:.0f}m"
                )

                # ---- Evaluation (only at end to save VRAM) ----
                if global_step == total_steps:
                    model.eval()
                    eval_losses = []
                    for es in eval_subset:
                        eloss = compute_loss(model, processor, es, args.max_dim)
                        eval_losses.append(eloss.item())
                    model.train()

                    if eval_losses:
                        avg_eval = sum(eval_losses) / len(eval_losses)
                        best_marker = " (BEST!)" if avg_eval < best_eval else ""
                        log(f"  [Eval  S{global_step}] loss={avg_eval:.4f}{best_marker}")
                        if avg_eval < best_eval:
                            best_eval = avg_eval
                            lora_w = {k: v.numpy() for k, v in model.named_parameters() if 'lora' in k.lower()}; paddle.save(lora_w, f"{OUTPUT_DIR}/best_lora_final")

                # ---- Checkpoint (every 500 steps, save VRAM) ----
                if global_step % 500 == 0:
                    paddle.save({k: v.numpy() for k, v in model.named_parameters() if 'lora' in k.lower()}, f"{OUTPUT_DIR}/checkpoint_lora_final")
                    with open(f"{OUTPUT_DIR}/loss_history_final.json", "w") as fh:
                        json.dump(history, fh, indent=2)
                    log(f"  [Ckpt] Saved @ step {global_step}")

        ep_elapsed = (time.time() - ep_start) / 60
        log(f"[Epoch {epoch + 1}] Done in {ep_elapsed:.1f}m")

    total_time = (time.time() - t0) / 60
    log("=" * 60)
    log(f"Training complete!  Total: {total_time:.0f}m  Best eval: {best_eval:.4f}")
    log("=" * 60)

    # -----------------------------------------------------------------------
    # Save final model + history
    # -----------------------------------------------------------------------
    paddle.save({k: v.numpy() for k, v in model.named_parameters() if 'lora' in k.lower()}, f"{OUTPUT_DIR}/final_lora_final")
    with open(f"{OUTPUT_DIR}/loss_history_final.json", "w") as fh:
        json.dump(history, fh, indent=2)

    # -----------------------------------------------------------------------
    # Verify LoRA weights actually changed
    # -----------------------------------------------------------------------
    log("Verifying LoRA weight changes...")
    import numpy as np
    sd = {k: v for k, v in model.named_parameters() if k in init_lora}
    changed = 0
    unchanged_list = []
    for k in init_lora:
        if k in sd:
            diff = np.abs(sd[k].numpy() - init_lora[k]).max()
            if diff > 1e-8:
                changed += 1
            else:
                unchanged_list.append(k)
    log(f"  LoRA weights changed: {changed}/{len(init_lora)}")
    if changed > 0:
        log("SUCCESS: Training produced non-zero LoRA updates!")
    else:
        log("FAIL: All LoRA weights unchanged after training!")
        if unchanged_list:
            log(f"  Unchanged keys (first 5): {unchanged_list[:5]}")
    log("Done!")


if __name__ == "__main__":
    main()
