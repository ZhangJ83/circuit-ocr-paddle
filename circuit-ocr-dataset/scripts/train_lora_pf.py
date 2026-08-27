"""PaddleOCR-VL LoRA Training using PaddleFormers built-in LoRAModel."""
import os, sys, json, time, argparse, random
from pathlib import Path
from datetime import datetime

os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HOME": "F:/hf_cache/hub",
    "PADDLE_HOME": "F:/paddle_cache", "HF_HUB_CACHE": "F:/hf_cache/hub",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "FLAGS_allocator_strategy": "auto_growth",
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUTPUT_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"
LOG_FILE = f"{OUTPUT_DIR}/training_pf.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def compute_loss(model, processor, sample, max_dim=224):
    from PIL import Image
    from io import BytesIO
    query = sample["messages"][0]["content"]
    label = sample["messages"][1]["content"]
    img_path = sample["images"][0]
    if not img_path.startswith("/"):
        img_path = f"{DATASET_DIR}/{img_path.lstrip('./')}"
    if not Path(img_path).exists(): return None
    image = None
    try:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0)
        image = Image.open(buf)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query.replace("<image>", "")}
        ]}]
        # Tokenize the FULL message (prompt + label) so labels align with logits
        full_msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query.replace("<image>", "")}
        ]}, {"role": "assistant", "content": label}]
        full_inputs = processor.apply_chat_template(
            full_msgs, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pd"
        )
        # Build labels: -100 for prompt tokens, actual ids for response tokens
        prompt_inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pd"
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        full_ids = full_inputs["input_ids"]
        labels = paddle.full_like(full_ids, -100)  # -100 = ignore
        labels[0, prompt_len:] = full_ids[0, prompt_len:]

        # Manual loss: get logits, compute CE only on label positions
        # Avoid model(**inputs, labels=labels) which crashes Paddle 2.6.2 on Windows
        out = model(**full_inputs)  # no labels → returns (logits,)
        logits = out[0] if isinstance(out, (tuple, list)) else out  # [1, seq, vocab]

        # Shift: predict token[t] from logits[t], target is token[t+1]
        shift_logits = logits[:, :-1, :]  # [1, seq-1, vocab]
        shift_labels = labels[:, 1:]       # [1, seq-1]

        # Mask: only compute loss where labels != -100
        mask = (shift_labels != -100).astype('float32')
        shift_labels_clamped = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))

        ce = paddle.nn.functional.cross_entropy(
            shift_logits.reshape([-1, shift_logits.shape[-1]]),
            shift_labels_clamped.reshape([-1]),
            reduction='none'
        ).reshape(shift_labels.shape)
        loss = (ce * mask).sum() / mask.sum().clip(min=1)
        return loss
    except Exception as e:
        return None
    finally:
        if image is not None:
            try: image.close()
            except: pass
        import paddle; paddle.device.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_eval_samples", type=int, default=20)
    ap.add_argument("--data_size", type=int, default=523, help="523=full, 5=test")
    ap.add_argument("--max_dim", type=int, default=224)
    args = ap.parse_args()

    # Patches must be applied BEFORE importing PaddleFormers
    apply_paddle_patches()
    log("[Patches] OK")

    import paddle
    import numpy as np
    paddle.set_device("gpu")
    log(f"GPU: {paddle.device.cuda.get_device_name(0)}")

    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
    from paddleformers.peft.lora import LoRAConfig, LoRAModel

    # Load processor
    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Load base model
    log("Loading base model...")
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
        low_cpu_mem_usage=True, dtype="float32",
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    log("Base model loaded")

    # Apply LoRA
    log(f"Applying LoRA: r={args.rank}, alpha={args.alpha}")
    lora_config = LoRAConfig(
        r=args.rank, lora_alpha=args.alpha,
        target_modules=[".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj"],
    )
    model = LoRAModel(model, lora_config)
    model.mark_only_lora_as_trainable()
    model.train()

    total = sum(p.size for p in model.parameters())
    trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
    log(f"Params: Total={total:,} Trainable={trainable:,} ({100*trainable/total:.2f}%)")

    # Load data
    train_data = [json.loads(l) for l in open(f"{DATASET_DIR}/ocr_vl_sft-train.jsonl", encoding="utf-8") if l.strip()]
    eval_data = [json.loads(l) for l in open(f"{DATASET_DIR}/ocr_vl_sft-test.jsonl", encoding="utf-8") if l.strip()]
    if args.data_size < len(train_data):
        train_data = train_data[:args.data_size]
    log(f"Data: Train={len(train_data)} Eval={len(eval_data[:args.max_eval_samples])}")

    # Optimizer
    spe = max(1, len(train_data) // args.grad_accum)
    total_steps = args.epochs * spe
    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=total_steps, eta_min=5e-5
    )
    opt = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=[p for p in model.parameters() if not p.stop_gradient],
        weight_decay=0.1, beta1=0.9, beta2=0.95, epsilon=1e-8,
    )

    # Save initial LoRA weights to verify they change
    init_lora = {}
    for k, v in model.state_dict().items():
        if "lora_A" in k or "lora_B" in k:
            init_lora[k] = v.numpy().copy()
    log(f"Saved {len(init_lora)} initial LoRA weights for comparison")

    # Training loop
    hist = []; gs = 0; best_eval = float("inf"); t0 = time.time()
    log(f"{'='*50}")
    log(f"Training: r={args.rank} epochs={args.epochs} samples={len(train_data)} grad_accum={args.grad_accum}")
    log(f"{'='*50}")

    for ep in range(args.epochs):
        ep_start = time.time(); sl = 0.0; random.shuffle(train_data)
        log(f"[Epoch {ep+1}/{args.epochs}] Starting...")

        for i, s in enumerate(train_data):
            loss = compute_loss(model, processor, s, args.max_dim)
            if loss is None: continue
            if i < 3:
                log(f"  Sample {i}: loss={float(loss):.6f}")

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            sl += float(loss)

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_data):
                opt.step(); lr_scheduler.step(); opt.clear_grad(); gs += 1
                avg_loss = sl / args.grad_accum; sl = 0.0
                hist.append({"step": gs, "epoch": ep+1, "loss": float(avg_loss), "lr": opt.get_lr()})
                elapsed = (time.time() - t0) / 60
                eta = (elapsed / gs) * total_steps - elapsed if gs > 0 else 0
                log(f"[E{ep+1} S{gs:3d}/{total_steps}] loss={float(avg_loss):.4f} lr={opt.get_lr():.2e} {elapsed:.0f}m ETA={eta:.0f}m")

                # Eval every 50 steps
                if gs % 50 == 0:
                    model.eval()
                    els = []
                    for es in eval_data[:args.max_eval_samples]:
                        eloss = compute_loss(model, processor, es, args.max_dim)
                        if eloss is not None: els.append(float(eloss))
                    model.train()
                    if els:
                        ae = sum(els) / len(els)
                        log(f"  [Eval S{gs}] loss={ae:.4f}{' (BEST!)' if ae < best_eval else ''}")
                        if ae < best_eval:
                            best_eval = ae
                            model.save_pretrained(f"{OUTPUT_DIR}/best_lora_pf")

                # Checkpoint every 100 steps
                if gs % 100 == 0:
                    model.save_pretrained(f"{OUTPUT_DIR}/checkpoint_lora_pf")
                    json.dump(hist, open(f"{OUTPUT_DIR}/loss_history_pf.json", "w"), indent=2)
                    log(f"  [Ckpt] Saved @ S{gs}")

        log(f"[Epoch {ep+1}] Done in {(time.time()-ep_start)/60:.1f}m")

    total_time = (time.time() - t0) / 60
    log(f"{'='*50}")
    log(f"Training complete! {total_time:.0f}m Best eval: {best_eval:.4f}")
    log(f"{'='*50}")

    # Save final
    model.save_pretrained(f"{OUTPUT_DIR}/final_lora_pf")
    json.dump(hist, open(f"{OUTPUT_DIR}/loss_history_pf.json", "w"), indent=2)

    # Verify LoRA weights changed
    log("Verifying LoRA weight changes...")
    sd = model.state_dict()
    changed = 0
    for k in init_lora:
        if k in sd:
            diff = np.abs(sd[k].numpy() - init_lora[k]).max()
            if diff > 1e-8:
                changed += 1
    log(f"  LoRA weights changed: {changed}/{len(init_lora)}")
    if changed > 0:
        log("SUCCESS: Training produced non-zero LoRA updates!")
    else:
        log("FAIL: All LoRA weights unchanged after training!")
    log("Done!")

if __name__ == "__main__":
    main()
