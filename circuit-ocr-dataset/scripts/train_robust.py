"""Robust training script — survives problematic samples, accepts CLI args."""
import os, sys, json, time, random, argparse
from types import ModuleType
_dummy = ModuleType('dummy_flex_checkpoint')
_dummy.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
import numpy as np; from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = r"g:/mimo_project/circuit_ocr"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"

def log(m):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {m}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp")
    ap.add_argument("--max_dim", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--freeze_projector", type=int, default=1)
    args = ap.parse_args()

    OUTPUT_DIR = os.path.join(DATASET_DIR, "checkpoints", args.name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CHECKPOINT_STEPS = 200

    log(f"=== {args.name} === dim={args.max_dim} epochs={args.epochs} lr={args.lr:.0e} dropout={args.dropout} freeze_proj={args.freeze_projector}")

    # Load model
    log("Loading model...")
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
        low_cpu_mem_usage=True, dtype="bfloat16")
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
    lc = LoRAConfig(r=args.rank, lora_alpha=args.rank*2, target_modules=TARGETS, lora_dropout=args.dropout)
    model = LoRAModel(model, lc)
    model.mark_only_lora_as_trainable()
    if not hasattr(model.model, 'full'): model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    tp = [p for p in model.parameters() if not p.stop_gradient]
    log(f"Trainable: {sum(p.numel() for p in tp):,}")

    # Data
    with open(os.path.join(DATASET_DIR, "output", "train_v10fmt.jsonl"), encoding="utf-8") as f:
        all_data = [json.loads(l) for l in f if l.strip()]
    random.shuffle(all_data)
    split = int(len(all_data) * 0.9)
    train_data = all_data[:split]; val_data = all_data[split:]
    GRAD_ACCUM = 4; GRAD_CLIP = 1.0
    total_steps = args.epochs * len(train_data) // GRAD_ACCUM
    log(f"Train: {len(train_data)} Val: {len(val_data)} Steps: {total_steps}")

    cosine = paddle.optimizer.lr.CosineAnnealingDecay(args.lr, T_max=max(1, total_steps - 100), eta_min=args.lr/10)
    lrs = paddle.optimizer.lr.LinearWarmup(cosine, warmup_steps=100, start_lr=args.lr/10, end_lr=args.lr)
    opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

    monitor_samples = val_data[:3]
    model.train(); t0 = time.time(); gs = 0; el_acc = 0.0; opt.clear_grad()
    best_loss = float('inf'); skipped = 0

    def quick_inference(samples, max_tokens=60):
        preds = []
        for s in samples:
            try:
                img_path = s['images'][0]
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                if max(w, h) > args.max_dim:
                    scale = args.max_dim / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
                inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
                input_ids = inp["input_ids"]; attn = inp["attention_mask"]
                pv = inp.get("pixel_values"); igt = inp.get("image_grid_thw")
                gen = []
                with paddle.no_grad():
                    for _ in range(max_tokens):
                        out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pv, image_grid_thw=igt)
                        logits = out[0] if isinstance(out, (list, tuple)) else out.logits
                        ntl = logits[:, -1, :]
                        for tid in set(gen):
                            sc = float(ntl[0, tid]); ntl[0, tid] = sc * 1.1 if sc < 0 else sc / 1.1
                        nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
                        if nt == processor.tokenizer.eos_token_id: break
                        gen.append(nt)
                        input_ids = paddle.concat([input_ids, paddle.to_tensor([[nt]])], axis=1)
                        attn = paddle.concat([attn, paddle.ones([1, 1], dtype=attn.dtype)], axis=1)
                preds.append(processor.tokenizer.decode(gen, skip_special_tokens=True))
                img.close()
            except Exception as e:
                preds.append(f"[ERR:{str(e)[:30]}]")
        return preds

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        log(f"--- Epoch {epoch+1}/{args.epochs} ---")
        for idx, sample in enumerate(train_data):
            try:
                img_path = sample['images'][0]
                if not os.path.exists(img_path): continue
                image = Image.open(img_path).convert("RGB")
                w, h = image.size
                if max(w, h) > args.max_dim:
                    scale = args.max_dim / max(w, h); image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                buf = BytesIO(); image.save(buf, format="JPEG", quality=95); buf.seek(0); image = Image.open(buf)

                query = sample["messages"][0]["content"]
                label = sample["messages"][1]["content"]

                prompt_msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
                prompt_inputs = processor.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
                prompt_ids = prompt_inputs["input_ids"][0]; prompt_len = prompt_ids.shape[0]

                lt = processor.tokenizer(label, return_tensors="pd", padding=False, truncation=True, max_length=512)
                label_ids = lt["input_ids"][0]
                eos_t = paddle.to_tensor([processor.tokenizer.eos_token_id], dtype=label_ids.dtype)
                label_ids = paddle.concat([label_ids, eos_t], axis=0); label_len = label_ids.shape[0]

                full_ids = paddle.concat([prompt_ids, label_ids], axis=0).unsqueeze(0)
                full_mask = paddle.concat([prompt_inputs["attention_mask"][0], paddle.ones([label_len], dtype="int64")], axis=0).unsqueeze(0)
                labels = paddle.full([1, prompt_len + label_len], -100, dtype="int64")
                labels[0, prompt_len:] = label_ids

                out = model(input_ids=full_ids, attention_mask=full_mask,
                           pixel_values=prompt_inputs["pixel_values"], image_grid_thw=prompt_inputs.get("image_grid_thw"))
                logits = out[0] if isinstance(out, (tuple, list)) else out.logits
                shift_logits = paddle.cast(logits[:, :-1, :], "float32"); shift_labels = labels[:, 1:]
                mask = paddle.cast(shift_labels != -100, "float32")
                shift_labels_clean = paddle.where(shift_labels != -100, shift_labels, paddle.zeros_like(shift_labels))
                ce = paddle.nn.functional.cross_entropy(
                    shift_logits.reshape([-1, shift_logits.shape[-1]]),
                    shift_labels_clean.reshape([-1]), reduction="none").reshape(shift_labels.shape)
                loss = (ce * mask).sum() / mask.sum().clip(min=1)
                (loss / GRAD_ACCUM).backward(); el_acc += loss.item()
                image.close()

                if (idx + 1) % GRAD_ACCUM == 0 or idx == len(train_data) - 1:
                    paddle.nn.utils.clip_grad_norm_(tp, GRAD_CLIP)
                    opt.step(); lrs.step(); opt.clear_grad(); gs += 1

                    if gs % 20 == 0:
                        eta = (time.time()-t0)/max(1,gs)*(total_steps-gs)/60
                        avg_loss = el_acc/max(1, idx+1)
                        log(f"E{epoch+1}/{args.epochs} S{gs}/{total_steps} loss={avg_loss:.4f} lr={opt.get_lr():.2e} ETA={eta:.0f}m")

                    if gs % CHECKPOINT_STEPS == 0:
                        log(f"--- Checkpoint S{gs} ---")
                        model.eval()
                        lora_dict = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                        ckpt_path = os.path.join(OUTPUT_DIR, f"lora_s{gs}.pdparams")
                        paddle.save(lora_dict, ckpt_path)
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            paddle.save(lora_dict, os.path.join(OUTPUT_DIR, "best.pdparams"))
                        log(f"  Saved: {ckpt_path} ({len(lora_dict)} matrices) loss={loss.item():.4f}")
                        preds = quick_inference(monitor_samples)
                        for mi, pred in enumerate(preds[:2]):
                            ref = monitor_samples[mi]["messages"][1]["content"][:60]
                            log(f"  [{mi}] Pred: {pred[:60]}")
                            log(f"  [{mi}] Ref:  {ref}")
                        model.train()
            except Exception as e:
                skipped += 1
                if skipped <= 3: log(f"  SKIP sample {idx}: {str(e)[:60]}")
                try: opt.clear_grad()
                except: pass
                continue

    total_min = (time.time()-t0)/60
    log(f"DONE {total_min:.0f}m. Skipped {skipped}/{len(train_data)*args.epochs} samples. Best loss={best_loss:.4f}")
    log(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
