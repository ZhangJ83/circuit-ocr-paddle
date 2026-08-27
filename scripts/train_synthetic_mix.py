"""Train PaddleOCR-VL on mixed real + synthetic KiCad data.
Uses train_fast.py approach with manual CE loss.
"""
import os, sys, json, time, random, argparse, re
from types import ModuleType

# === CUDA/cuDNN DLL paths (for FlashAttention support) ===
for _dp in [
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin",
    r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\lib\site-packages\torch\lib",
]:
    if os.path.exists(_dp):
        os.environ["PATH"] = _dp + ";" + os.environ.get("PATH", "")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# === Paddle compatibility patches ===
_d = ModuleType('d'); _d.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _d)
sys.path.insert(0, 'circuit-ocr-dataset/scripts')
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()

import paddle; paddle.set_device("gpu")
import paddle.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
PROJECT_DIR = r"g:/mimo_project/circuit_ocr"
MAX_DIM = 320  # Lower from 384 to reduce VRAM
RANK = 16; ALPHA = 32; DROPOUT = 0.05
TARGETS = ['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj', '.*linear_1', '.*linear_2']

def log(msg):
    print(f"[TRAIN-{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def to_pd(d):
    o = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray): o[k] = paddle.to_tensor(v)
        elif isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], np.ndarray): o[k] = paddle.to_tensor(np.array(v))
            else: o[k] = v
        else: o[k] = v
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='synth_mix')
    ap.add_argument('--train_data', default=f'{PROJECT_DIR}/output/train_synthetic_mix.jsonl')
    ap.add_argument('--val_data', default=f'{PROJECT_DIR}/output/val_clean.jsonl')
    ap.add_argument('--test_data', default=f'{PROJECT_DIR}/output/test_clean.jsonl')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--output_dir', default=f'{PROJECT_DIR}/checkpoints/synthetic_mix')
    ap.add_argument('--checkpoint_steps', type=int, default=400)
    ap.add_argument('--n_val', type=int, default=10)
    ap.add_argument('--n_test', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    log(f"Loading data: {args.train_data}")
    with open(args.train_data, encoding='utf-8') as f:
        train_data = [json.loads(l) for l in f if l.strip()]
    random.shuffle(train_data)
    log(f"Train: {len(train_data)} samples")

    with open(args.val_data, encoding='utf-8') as f:
        val_data = [json.loads(l) for l in f if l.strip()][:args.n_val]
    with open(args.test_data, encoding='utf-8') as f:
        test_data = [json.loads(l) for l in f if l.strip()][:args.n_test]

    # Load model with LoRA
    log("Loading model...")
    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH, convert_from_hf=True, load_checkpoint_format='naive',
        low_cpu_mem_usage=True, dtype='bfloat16')
    model.config._attn_implementation = 'flashmask'
    model.visual.config._attn_implementation = 'flashmask'

    # Patch Tensor.expand for PyTorch compatibility
    _old_expand = paddle.Tensor.expand
    def _patched_expand(self, *args, **kwargs):
        if len(args) > 1:
            return _old_expand(self, list(args), **kwargs)
        return _old_expand(self, *args, **kwargs)
    paddle.Tensor.expand = _patched_expand

    for n, p in model.named_parameters():
        if "mlp_AR" in n or "projector" in n:
            p.stop_gradient = True

    lc = LoRAConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, target_modules=TARGETS)
    model = LoRAModel(model, lc)
    if not hasattr(model.model, 'full'):
        model.model.full = lambda *a, **kw: iter(model.model.named_parameters())

    tp = [p for p in model.parameters() if not p.stop_gradient]
    log(f"Trainable: {sum(p.numel() for p in tp):,}")

    # Training setup
    total_steps = len(train_data) * args.epochs
    cd = paddle.optimizer.lr.CosineAnnealingDecay(args.lr, T_max=max(1, total_steps - 100), eta_min=args.lr / 10)
    lrs = paddle.optimizer.lr.LinearWarmup(cd, warmup_steps=min(100, total_steps // 4), start_lr=args.lr / 10, end_lr=args.lr)
    opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

    best_loss = float('inf')
    gs = 0
    t0 = time.time()

    val_fixed = val_data[:args.n_val]

    import gc; gc.collect()
    log(f"Training: {total_steps} steps ({args.epochs} epochs × {len(train_data)} samples)")

    import gc; gc.enable()

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        el = 0.0

        for i, s in enumerate(train_data):
            # Memory safety: periodic cache cleanup
            if i % 50 == 0:
                gc.collect()

            try:
                ip = s["images"][0]
                if not os.path.exists(ip):
                    ip = ip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
                img = Image.open(ip).convert("RGB")
                w, h = img.size
                scale = MAX_DIM / max(w, h)
                if scale < 1:
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

                label = s["messages"][1]["content"]
                label_ids = proc.tokenizer.encode(label) + [proc.tokenizer.eos_token_id or 2]
                label_tensor = paddle.to_tensor(label_ids, dtype="int64")

                # Compute correct placeholder count from image features
                img_inputs = proc.image_processor(images=[img], return_tensors="np")
                igt = img_inputs["image_grid_thw"][0]
                n_patches = int(igt[1]) * int(igt[2])
                n_copies = max(1, n_patches // 4)
                prompt = ('<' + '|placeholder|' + '>') * n_copies + 'OCR:'
                inp = proc(text=[prompt], images=[img], return_tensors="np",
                          padding=True, max_length=2048, truncation=True)

                inp_pd = to_pd(inp)
                prompt_len = inp_pd["input_ids"].shape[1]
                inp_pd["input_ids"] = paddle.concat([inp_pd["input_ids"][0], label_tensor]).unsqueeze(0)
                inp_pd["labels"] = paddle.concat([paddle.full([prompt_len], -100, dtype="int64"), label_tensor]).unsqueeze(0)
                inp_pd["attention_mask"] = paddle.ones([1, inp_pd["input_ids"].shape[1]], dtype="int64")

                out = model(**inp_pd)
                loss_val = out[0] if isinstance(out, (list, tuple)) else out.loss
                loss_val.backward()
                paddle.nn.utils.clip_grad_norm_(tp, 1.0)
                opt.step(); lrs.step(); opt.clear_grad()
                gs += 1; el += loss_val.item()

                # Immediate cleanup
                del out, inp_pd, label_tensor, img_inputs
                img.close()
            except RuntimeError as e:
                # CUDA OOM or similar — recover gracefully
                gc.collect()
                log(f"  MEM SAFE {i}: {str(e)[:80]}")
                continue
            except Exception as e:
                if i < 3:
                    log(f"  Train err sample {i}: {str(e)[:80]}")
                continue

            if gs % 50 == 0 and gs > 0:
                eta = (time.time() - t0) / max(1, gs) * (total_steps - gs) / 60
                log(f"E{epoch+1}/{args.epochs} S{gs}/{total_steps} loss={el / max(1, i + 1):.4f} ETA={eta:.0f}m")

            if gs > 0 and gs % args.checkpoint_steps == 0:
                train_loss = el / max(1, i + 1)
                ld = {k: paddle.cast(p.detach(), "float16") for k, p in model.named_parameters() if 'lora_' in k}
                paddle.save(ld, os.path.join(args.output_dir, f"checkpoint_s{gs}.pdparams"))
                if train_loss < best_loss:
                    best_loss = train_loss
                    paddle.save(ld, os.path.join(args.output_dir, "best.pdparams"))
                    log(f"  BEST loss={best_loss:.4f}")

                # Validation
                model.eval()
                preds = []; refs = []
                with paddle.no_grad():
                    for vs in val_fixed:
                        try:
                            vip = vs["images"][0]
                            if not os.path.exists(vip):
                                vip = vip.replace("/root/circuit_ocr/", PROJECT_DIR + "/")
                            vimg = Image.open(vip).convert("RGB")
                            vw, vh = vimg.size
                            vscale = MAX_DIM / max(vw, vh)
                            if vscale < 1:
                                vimg = vimg.resize((int(vw * vscale), int(vh * vscale)), Image.LANCZOS)
                            vimg_inputs = proc.image_processor(images=[vimg], return_tensors="np")
                            vigt = vimg_inputs["image_grid_thw"][0]
                            vn_patches = int(vigt[1]) * int(vigt[2])
                            vn_copies = max(1, vn_patches // 4)
                            prompt = ('<' + '|placeholder|' + '>') * vn_copies + 'OCR:'
                            vinp = proc(text=[prompt], images=[vimg], return_tensors="np",
                                      padding=True, max_length=2048, truncation=True)
                            vinp_pd = to_pd(vinp)
                            actual_len = int(vinp_pd["attention_mask"].sum().numpy()[0])
                            vinp_pd["input_ids"] = vinp_pd["input_ids"][:, :actual_len]
                            vinp_pd["attention_mask"] = vinp_pd["attention_mask"][:, :actual_len]

                            from paddleformers.generation import GenerationConfig
                            gc = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2,
                                                  pad_token_id=0, use_cache=False)
                            out_gen = model.model.generate(**vinp_pd, generation_config=gc, max_new_tokens=256)
                            gen = out_gen[0].tolist()[0]
                            preds.append(proc.tokenizer.decode(gen, skip_special_tokens=True))
                            refs.append(vs["messages"][1]["content"])
                            vimg.close()
                        except Exception as e:
                            preds.append("[ERR]")
                            refs.append(vs["messages"][1]["content"])

                model.train()

                # Quick metrics
                from eval_metrics import compute_all
                m = compute_all(preds, refs, label=f"s{gs}")
                log(f"  Val: jf1={m['joint_f1']:.4f} CompF1={m['component_f1']:.4f} RepRate={m['repetition_rate']:.2%}")
                if preds and preds[0] != "[ERR]":
                    log(f"  Pred[0]: {preds[0][:80]}")
                    log(f"  Ref [0]: {refs[0][:80]}")

        log(f"Epoch {epoch+1}: {(time.time() - t0) / 60:.1f}min total")

    tt = (time.time() - t0) / 60
    log(f"DONE {tt:.1f}min. Best loss={best_loss:.4f}")

    # Final evaluation
    log("=== Final Evaluation ===")
    # Load best checkpoint
    best_path = os.path.join(args.output_dir, "best.pdparams")
    state = paddle.load(best_path)
    for k, p in model.named_parameters():
        if k in state:
            v = state[k]
            if p.dtype != v.dtype: v = paddle.cast(v, p.dtype)
            if list(p.shape) == list(v.shape): p.set_value(v)
    model.eval()

    preds = []; refs = []
    with paddle.no_grad():
        for i, s in enumerate(test_data):
            try:
                ip = s["images"][0].replace("/root/circuit_ocr/", PROJECT_DIR + "/")
                img = Image.open(ip).convert("RGB")
                w, h = img.size
                scale = MAX_DIM / max(w, h)
                if scale < 1:
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

                # Compute correct placeholder count from image features
                img_inputs = proc.image_processor(images=[img], return_tensors="np")
                igt = img_inputs["image_grid_thw"][0]
                n_patches = int(igt[1]) * int(igt[2])
                n_copies = max(1, n_patches // 4)
                prompt = ('<' + '|placeholder|' + '>') * n_copies + 'OCR:'
                inp = proc(text=[prompt], images=[img], return_tensors="np",
                          padding=True, max_length=2048, truncation=True)
                inp_pd = to_pd(inp)
                actual_len = int(inp_pd["attention_mask"].sum().numpy()[0])
                inp_pd["input_ids"] = inp_pd["input_ids"][:, :actual_len]
                inp_pd["attention_mask"] = inp_pd["attention_mask"][:, :actual_len]

                from paddleformers.generation import GenerationConfig
                gc = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2,
                                      pad_token_id=0, use_cache=False)
                out_gen = model.model.generate(**inp_pd, generation_config=gc, max_new_tokens=256)
                gen = out_gen[0].tolist()[0]
                preds.append(proc.tokenizer.decode(gen, skip_special_tokens=True))
                refs.append(s["messages"][1]["content"])
                img.close()
            except Exception as e:
                preds.append("[ERR]")
                refs.append(s["messages"][1]["content"])

            if (i + 1) % 10 == 0:
                log(f"  Eval {i+1}/{len(test_data)}")

    # Compute metrics
    import Levenshtein
    re_comp = re.compile(r'\b((?:LED|[RCDLQUJYF])\d+)\b')

    cf1s = []; jf1s = []
    for p, r in zip(preds, refs):
        pc = set(re_comp.findall(p)); rc = set(re_comp.findall(r))
        if not pc and not rc: cf1 = 1.0
        elif not pc or not rc: cf1 = 0.0
        else: tp = len(pc & rc); cf1 = 2*(tp/len(pc))*(tp/len(rc))/(tp/len(pc)+tp/len(rc)) if (tp/len(pc)+tp/len(rc)) > 0 else 0.0
        cf1s.append(cf1)

        def parse(t):
            ps = set()
            for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))', t):
                v = m.group(2).strip().rstrip(',').replace(' ', '').upper()
                if v and len(v) < 50: ps.add((m.group(1), v))
            return ps
        pp = parse(p); rp = parse(r)
        if not pp and not rp: jf1 = 1.0
        elif not pp or not rp: jf1 = 0.0
        else: tp = len(pp & rp); jf1 = 2*(tp/len(pp))*(tp/len(rp))/(tp/len(pp)+tp/len(rp)) if (tp/len(pp)+tp/len(rp)) > 0 else 0.0
        jf1s.append(jf1)

    avg_cf1 = np.mean(cf1s)
    avg_jf1 = np.mean(jf1s)
    ned = np.mean([Levenshtein.distance(p, r) / max(len(p), len(r), 1) for p, r in zip(preds, refs)])

    log(f"FINAL: CompF1={avg_cf1:.4f} JointF1={avg_jf1:.4f} NED={ned:.4f}")
    log(f"vs exp6: CompF1=0.1192 JointF1=0.0076 NED=0.9460")

    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({'CompF1': avg_cf1, 'JointF1': avg_jf1, 'NED': ned}, f, indent=2)

    return avg_cf1, avg_jf1

if __name__ == '__main__':
    main()
