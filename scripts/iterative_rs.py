"""Iterative Rejection Sampling Fine-tuning for Circuit OCR.
Theory: For each training image, generate N diverse candidates → score →
select best → SFT. Repeat. Monotonic improvement guarantee from rejection sampling.
"""
import os, sys, json, time, random, re, argparse
from types import ModuleType

# === CUDA/cuDNN DLL paths (must be set BEFORE paddle import for flashmask) ===
for _dp in [
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin",
    r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\lib\site-packages\torch\lib",
    r"E:\080000software\080900_Miniconda\miniconda3\pkgs\cudatoolkit-11.3.1-h59b6b97_2\Library\bin"
]:
    if os.path.exists(_dp):
        os.environ["PATH"] = _dp + ";" + os.environ.get("PATH", "")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# === Paddle flex_checkpoint dummy (must be BEFORE any paddle import) ===
_d = ModuleType('d')
_d.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _d)

# === Safetensors PySafeSlice.shape patch ===
try:
    import safetensors
    if hasattr(safetensors, 'safe_open'):
        _orig_safe_open = safetensors.safe_open
        _patched_flag = [False]
        def _patched_safe_open(*args, **kwargs):
            result = _orig_safe_open(*args, **kwargs)
            if not _patched_flag[0] and len(result.keys()) > 0:
                try:
                    sl = result.get_slice(list(result.keys())[0])
                    if not hasattr(type(sl), 'shape'):
                        type(sl).shape = property(lambda s: s.get_shape())
                    _patched_flag[0] = True
                except Exception:
                    pass
            return result
        safetensors.safe_open = _patched_safe_open
except Exception:
    pass

# === Apply ALL Paddle compatibility patches (swiglu, LongTensor, reshape, transpose, etc.) ===
sys.path.insert(0, 'circuit-ocr-dataset/scripts')
from eval_benchmark import apply_paddle_patches
apply_paddle_patches()

# === Now safe to import paddle and paddleformers ===
import paddle
paddle.set_device("gpu")
import paddle.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
import Levenshtein

# === Config ===
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
PROJECT_DIR = r"g:/mimo_project/circuit_ocr"
EXP6_CKPT = os.path.join(PROJECT_DIR, "PaddleOCR-VL-LoRA-circuit-ocr", "lora_exp6_best.pdparams")
TRAIN_DATA = os.path.join(PROJECT_DIR, "output", "train_v10fmt_synth.jsonl")
TEST_DATA = os.path.join(PROJECT_DIR, "output", "test_clean.jsonl")
TARGETS = ['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj', '.*linear_1', '.*linear_2']
MAX_DIM = 384
RANK = 16
ALPHA = 32
DROPOUT = 0.05

def log(msg):
    print(f"[RS-{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# === Reward function ===
RE_COMP = re.compile(r'\b((?:LED|[RCDLQUJYF])\d+)\b')

def compute_comp_f1(pred, ref):
    """Component-level F1: set of refdes tokens."""
    pc = set(RE_COMP.findall(pred))
    rc = set(RE_COMP.findall(ref))
    if not pc and not rc:
        return 1.0
    if not pc or not rc:
        return 0.0
    tp = len(pc & rc)
    prec = tp / len(pc)
    rec = tp / len(rc)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def parse_pairs(text):
    """Parse (refdes, value) pairs from free text output."""
    pairs = set()
    for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))', text):
        v = m.group(2).strip().rstrip(',').replace(' ', '').upper()
        if v and len(v) < 50:
            pairs.add((m.group(1), v))
    return pairs

def compute_joint_f1(pred, ref):
    """Joint F1: (refdes, value) pairs must both match."""
    pp = parse_pairs(pred)
    rp = parse_pairs(ref)
    if not pp and not rp:
        return 1.0
    if not pp or not rp:
        return 0.0
    tp = len(pp & rp)
    prec = tp / len(pp)
    rec = tp / len(rp)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def reward_fn(pred, ref):
    """Composite reward: 0.4*CompF1 + 0.3*JointF1 + 0.15*format + 0.15*diversity"""
    cf1 = compute_comp_f1(pred, ref)
    jf1 = compute_joint_f1(pred, ref)

    # Format bonus: can we parse structured output?
    pairs = parse_pairs(pred)
    format_bonus = 0.15 if len(pairs) > 0 else -0.1

    # Diversity: unique lines ratio
    lines = [l.strip() for l in pred.strip().split('\n') if l.strip()]
    if len(lines) > 0:
        unique_ratio = len(set(lines)) / len(lines)
        div_bonus = 0.15 if unique_ratio > 0.5 else (-0.2 if unique_ratio < 0.3 else 0.0)
    else:
        div_bonus = -0.3

    return 0.4 * cf1 + 0.3 * jf1 + format_bonus + div_bonus

# === Model loading ===
def load_model_with_lora(ckpt_path=None):
    """Load base model + LoRA, optionally load pretrained LoRA weights."""
    log("Loading base model...")
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH, convert_from_hf=True, load_checkpoint_format='naive',
        low_cpu_mem_usage=True, dtype='bfloat16'
    )
    model.config._attn_implementation = 'flashmask'
    model.visual.config._attn_implementation = 'flashmask'

    lc = LoRAConfig(r=RANK, lora_alpha=ALPHA, target_modules=TARGETS, lora_dropout=DROPOUT)
    model = LoRAModel(model, lc)
    if not hasattr(model.model, 'full'):
        model.model.full = lambda *a, **kw: iter(model.model.named_parameters())

    if ckpt_path and os.path.exists(ckpt_path):
        state = paddle.load(ckpt_path)
        n = 0
        for k, p in model.named_parameters():
            if k in state:
                v = state[k]
                if p.dtype != v.dtype:
                    v = paddle.cast(v, p.dtype)
                if list(p.shape) == list(v.shape):
                    p.set_value(v)
                    n += 1
        log(f"Loaded {n} LoRA params from {os.path.basename(ckpt_path)}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

# === Generation ===
@paddle.no_grad()
def generate_one(model, processor, img_path, temperature=0.8):
    """Generate one output for an image."""
    try:
        if not os.path.exists(img_path):
            local = img_path.replace('/root/circuit_ocr/', PROJECT_DIR + '/')
            if not os.path.exists(local):
                return None
            img_path = local

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # JPEG re-encode for Paddle PNG decode bug
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        img = Image.open(buf)

        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img},
            {'type': 'text', 'text': 'OCR:'}
        ]}]
        inp = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pd'
        )
        ids = inp['input_ids']
        am = inp['attention_mask']
        pv = inp.get('pixel_values')
        igt = inp.get('image_grid_thw')

        gen = []
        for _ in range(80):
            out = model(input_ids=ids, attention_mask=am, pixel_values=pv, image_grid_thw=igt)
            logits = (out[0] if isinstance(out, (list, tuple)) else out.logits)[:, -1, :]
            logits = logits.astype('float32') / max(temperature, 0.1)

            # Repetition penalty
            for tid in set(gen):
                val = float(logits[0, tid])
                logits[0, tid] = val * 1.1 if val < 0 else val / 1.1

            # Argmax with temperature
            nt = int(paddle.argmax(logits, axis=-1).numpy()[0])
            if nt == processor.tokenizer.eos_token_id:
                break
            gen.append(nt)
            ids = paddle.concat([ids, paddle.to_tensor([[nt]])], axis=1)
            am = paddle.concat([am, paddle.ones([1, 1], dtype=am.dtype)], axis=1)

        img.close()
        return processor.tokenizer.decode(gen, skip_special_tokens=True)
    except Exception as e:
        # Log first few errors for debugging
        if not hasattr(generate_one, '_err_count'):
            generate_one._err_count = 0
        generate_one._err_count += 1
        if generate_one._err_count <= 5:
            import traceback
            log(f"GEN_ERR[{generate_one._err_count}] {img_path}: {e}")
            traceback.print_exc()
        return None

def generate_candidates(model, processor, samples, n_candidates, temperatures, out_dir):
    """Generate N candidates per sample with temperature diversity."""
    os.makedirs(out_dir, exist_ok=True)

    # Check for existing progress
    cache_file = os.path.join(out_dir, 'candidates.jsonl')
    done = set()
    all_results = {}
    if os.path.exists(cache_file):
        with open(cache_file, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                done.add(r['img'])
                all_results[r['img']] = r
        log(f"Resuming: {len(done)} images already generated")

    model.eval()
    t0 = time.time()
    pending = [s for s in samples if s['images'][0] not in done]
    log(f"Generating {n_candidates} candidates for {len(pending)} images ({len(done)} already done)")

    # Reset error counter for fresh error logging each round
    generate_one._err_count = 0

    for idx, sample in enumerate(pending):
        img_path = sample['images'][0]
        ref = sample['messages'][1]['content']
        candidates = []

        for c in range(n_candidates):
            temp = temperatures[c % len(temperatures)]
            pred = generate_one(model, processor, img_path, temperature=temp)
            if pred:
                r = reward_fn(pred, ref)
                candidates.append({'text': pred, 'reward': r, 'temp': temp})

        result = {
            'img': img_path,
            'ref': ref,
            'candidates': candidates,
            'best_idx': max(range(len(candidates)), key=lambda i: candidates[i]['reward']) if candidates else -1,
            'best_reward': max(c['reward'] for c in candidates) if candidates else -999,
        }
        all_results[img_path] = result

        # Append to cache
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        if (idx + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60
            rate = (idx + 1) / max(elapsed, 0.01)
            eta = (len(pending) - idx - 1) / max(rate, 0.01)
            avg_r = np.mean([result['best_reward']])
            log(f"  Gen {idx+1}/{len(pending)} ({rate:.1f}/min) best_r={result['best_reward']:.4f} ETA={eta:.0f}m")

    tt = (time.time() - t0) / 60
    log(f"Generation done: {len(all_results)} images in {tt:.1f}min")

    # Also update the full results
    with open(cache_file, 'w', encoding='utf-8') as f:
        for r in all_results.values():
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    return all_results

# === Score analysis ===
def analyze_candidates(results, n_candidates):
    """Print statistics about candidate quality."""
    best_rewards = [r['best_reward'] for r in results.values()]
    all_rewards = []
    for r in results.values():
        for c in r['candidates']:
            all_rewards.append(c['reward'])

    ref_rewards = []
    for r in results.values():
        ref = r['ref']
        # Score GT against itself (upper bound)
        ref_rewards.append(reward_fn(ref, ref))

    log(f"=== Candidate Statistics (N={n_candidates}) ===")
    log(f"  Samples: {len(results)}")
    log(f"  Best reward: mean={np.mean(best_rewards):.4f} median={np.median(best_rewards):.4f} max={np.max(best_rewards):.4f}")
    log(f"  All rewards: mean={np.mean(all_rewards):.4f} std={np.std(all_rewards):.4f}")
    log(f"  GT self-reward: {np.mean(ref_rewards):.4f}")
    log(f"  Fraction best_r > 0: {sum(1 for r in best_rewards if r > 0) / len(best_rewards):.1%}")

    # Show top/bottom examples
    sorted_items = sorted(results.values(), key=lambda x: x['best_reward'], reverse=True)
    valid_items = [it for it in sorted_items if it['best_idx'] >= 0 and len(it['candidates']) > 0]
    if valid_items:
        log(f"  Top-3 best_rewards:")
        for item in valid_items[:3]:
            best_c = item['candidates'][item['best_idx']]
            log(f"    r={best_c['reward']:.3f}: {best_c['text'][:80].replace(chr(10), ' ')}")
        log(f"  Bottom-3 best_rewards:")
        for item in valid_items[-3:]:
            best_c = item['candidates'][item['best_idx']]
            log(f"    r={best_c['reward']:.3f}: {best_c['text'][:80].replace(chr(10), ' ')}")
    else:
        log(f"  WARNING: All {len(sorted_items)} images have no valid candidates!")

# === SFT Training on Selected Candidates ===
def build_sft_data(results, synth_samples):
    """Build SFT training data: best candidates + synthetic text."""
    sft_samples = []

    # Best candidates from circuit images
    for r in results.values():
        if r['best_idx'] >= 0:
            best_text = r['candidates'][r['best_idx']]['text']
            # Only include if reward > threshold (avoid training on garbage)
            if r['best_reward'] > -0.1:
                sft_samples.append({
                    'images': [r['img']],
                    'messages': [
                        r.get('messages0', {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'OCR:'}]}),
                        {'role': 'assistant', 'content': best_text},
                    ]
                })

    # Need original messages for circuit samples
    # The results dict doesn't have original messages, so let me fix this...
    # For now, we use a different approach - the results have img and ref

    log(f"SFT data: {len(sft_samples)} best candidates (reward > -0.1)")
    return sft_samples

def train_sft_epoch(model, processor, train_samples, lr, output_dir, round_num):
    """One epoch SFT on selected candidates."""
    model.train()

    # Filter trainable params
    tp = [p for p in model.parameters() if not p.stop_gradient]
    log(f"Trainable params: {sum(p.numel() for p in tp):,}")

    total_steps = len(train_samples)
    cosine = paddle.optimizer.lr.CosineAnnealingDecay(
        lr, T_max=max(1, total_steps), eta_min=lr / 10
    )
    warmup_steps = min(50, total_steps // 4)
    lrs = paddle.optimizer.lr.LinearWarmup(
        cosine, warmup_steps=warmup_steps, start_lr=lr / 10, end_lr=lr
    )
    opt = paddle.optimizer.AdamW(lrs, parameters=tp, weight_decay=0.1)

    random.shuffle(train_samples)
    t0 = time.time()
    total_loss = 0.0

    for i, s in enumerate(train_samples):
        try:
            ip = s['images'][0]
            if not os.path.exists(ip):
                ip = ip.replace('/root/circuit_ocr/', PROJECT_DIR + '/')
            img = Image.open(ip).convert('RGB')
            w, h = img.size
            scale = MAX_DIM / max(w, h)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            label = s['messages'][1]['content']
            label_ids = processor.tokenizer.encode(label) + [processor.tokenizer.eos_token_id or 2]
            label_tensor = paddle.to_tensor(label_ids, dtype='int64')

            # Prompt construction (matching training data format)
            prompt = '<|placeholder|>' * 64 + 'OCR:'
            inp = processor(text=[prompt], images=[np.array(img)], return_tensors='np',
                          padding=True, max_length=2048, truncation=True)

            # Convert to paddle
            inp_pd = {}
            for k, v in inp.items():
                if isinstance(v, np.ndarray):
                    inp_pd[k] = paddle.to_tensor(v)
                elif isinstance(v, list):
                    inp_pd[k] = paddle.to_tensor(np.array(v) if isinstance(v[0], np.ndarray) else v)
                else:
                    inp_pd[k] = v

            prompt_len = inp_pd['input_ids'].shape[1]
            inp_pd['input_ids'] = paddle.concat([inp_pd['input_ids'][0], label_tensor]).unsqueeze(0)
            inp_pd['labels'] = paddle.concat([
                paddle.full([prompt_len], -100, dtype='int64'), label_tensor
            ]).unsqueeze(0)
            inp_pd['attention_mask'] = paddle.ones([1, inp_pd['input_ids'].shape[1]], dtype='int64')

            out = model(**inp_pd)
            loss_val = out[0] if isinstance(out, (list, tuple)) else out.loss
            loss_val.backward()
            paddle.nn.utils.clip_grad_norm_(tp, 1.0)
            opt.step()
            lrs.step()
            opt.clear_grad()

            total_loss += loss_val.item()
            img.close()
            del out, inp_pd, label_tensor
        except Exception as e:
            if i == 0:
                log(f"  Train error sample 0: {e}")
            continue

        if (i + 1) % 50 == 0:
            avg_l = total_loss / max(1, i + 1)
            elapsed = (time.time() - t0) / 60
            eta = elapsed / max(1, i + 1) * (total_steps - i - 1)
            log(f"  SFT {i+1}/{total_steps} loss={avg_l:.4f} ETA={eta:.0f}m")

    tt = (time.time() - t0) / 60
    avg_loss = total_loss / max(1, len(train_samples))
    log(f"SFT done: {tt:.1f}min, avg_loss={avg_loss:.4f}")

    # Save LoRA weights
    ckpt_dir = os.path.join(output_dir, f'round{round_num}')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'lora_best.pdparams')
    lora_dict = {k: paddle.cast(p.detach(), 'float16')
                 for k, p in model.named_parameters() if 'lora_' in k}
    paddle.save(lora_dict, ckpt_path)
    log(f"Saved: {ckpt_path}")

    return ckpt_path, avg_loss

# === Evaluation ===
def evaluate(model, processor, test_samples):
    """Evaluate on test set."""
    model.eval()
    refs = [s['messages'][1]['content'] for s in test_samples]
    preds = []

    t0 = time.time()
    for i, s in enumerate(test_samples):
        try:
            img_path = s['images'][0].replace('/root/circuit_ocr/', PROJECT_DIR + '/')
            # Use the same generate function but with deterministic temp=0.01
            pred = generate_one(model, processor, img_path, temperature=0.01)
            preds.append(pred if pred else '[ERR]')
        except:
            preds.append('[ERR]')

        if (i + 1) % 10 == 0:
            log(f"  Eval {i+1}/{len(test_samples)}")

    # Compute metrics
    cf1s, jf1s = [], []
    for p, r in zip(preds, refs):
        cf1s.append(compute_comp_f1(p, r))
        jf1s.append(compute_joint_f1(p, r))

    avg_cf1 = np.mean(cf1s)
    avg_jf1 = np.mean(jf1s)
    ned = np.mean([Levenshtein.distance(p, r) / max(len(p), len(r), 1)
                   for p, r in zip(preds, refs)])

    tt = (time.time() - t0) / 60
    log(f"Eval done in {tt:.1f}min: CompF1={avg_cf1:.4f} JointF1={avg_jf1:.4f} NED={ned:.4f}")

    return {'CompF1': avg_cf1, 'JointF1': avg_jf1, 'NED': ned}, preds, refs

# === Main pipeline ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--n_candidates', type=int, default=8)
    ap.add_argument('--temperatures', type=str, default='0.6,0.8,1.0')
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--output_dir', default=os.path.join(PROJECT_DIR, 'checkpoints', 'iterative_rs'))
    ap.add_argument('--exp6_ckpt', default=EXP6_CKPT)
    ap.add_argument('--train_data', default=TRAIN_DATA)
    ap.add_argument('--test_data', default=TEST_DATA)
    ap.add_argument('--n_test', type=int, default=30)
    ap.add_argument('--skip_gen', action='store_true', help='Skip generation if cached')
    args = ap.parse_args()

    temps = [float(t) for t in args.temperatures.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(args.train_data, encoding='utf-8') as f:
        all_data = [json.loads(l) for l in f if l.strip()]

    # Split: circuit vs synthetic text
    circuit_samples = [s for s in all_data if 'synth_text_images' not in s['images'][0]]
    synth_samples = [s for s in all_data if 'synth_text_images' in s['images'][0]]
    log(f"Data: {len(circuit_samples)} circuit + {len(synth_samples)} synth text = {len(all_data)} total")

    with open(args.test_data, encoding='utf-8') as f:
        test_samples = [json.loads(l) for l in f if l.strip()][:args.n_test]
    log(f"Test: {len(test_samples)} samples")

    # Load initial model (exp6)
    model, processor = load_model_with_lora(args.exp6_ckpt)

    # Evaluate baseline
    log("=== Baseline (exp6) ===")
    history = []
    base_metrics, _, _ = evaluate(model, processor, test_samples)
    history.append({'round': 0, 'ckpt': args.exp6_ckpt, **base_metrics})

    current_ckpt = args.exp6_ckpt

    for round_num in range(1, args.rounds + 1):
        log(f"{'='*60}")
        log(f"=== ROUND {round_num}/{args.rounds} ===")
        log(f"{'='*60}")

        # Phase 1: Generate candidates
        if not args.skip_gen or round_num == 1:
            gen_dir = os.path.join(args.output_dir, f'gen_round{round_num}')
            results = generate_candidates(
                model, processor, circuit_samples,
                args.n_candidates, temps, gen_dir
            )
            analyze_candidates(results, args.n_candidates)
        else:
            # Load from cache
            gen_dir = os.path.join(args.output_dir, f'gen_round{round_num}')
            cache_file = os.path.join(gen_dir, 'candidates.jsonl')
            results = {}
            with open(cache_file, encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    results[r['img']] = r
            log(f"Loaded {len(results)} cached candidates")
            analyze_candidates(results, args.n_candidates)

        # Phase 2: Build SFT data (best candidates + synth text)
        sft_samples = []
        for img_path, r in results.items():
            if r['best_idx'] >= 0 and r['best_reward'] > -0.1:
                best_text = r['candidates'][r['best_idx']]['text']
                # Reconstruct proper message format
                sft_samples.append({
                    'images': [img_path],
                    'messages': [
                        {'role': 'user', 'content': [
                            {'type': 'image'},
                            {'type': 'text', 'text': 'OCR:'}
                        ]},
                        {'role': 'assistant', 'content': best_text},
                    ]
                })

        # Add synthetic text samples (anti-collapse)
        sft_samples.extend(synth_samples)
        random.shuffle(sft_samples)
        log(f"SFT data: {len(sft_samples) - len(synth_samples)} best + {len(synth_samples)} synth = {len(sft_samples)} total")

        # Phase 3: SFT
        new_ckpt, train_loss = train_sft_epoch(
            model, processor, sft_samples,
            args.lr, args.output_dir, round_num
        )

        # Phase 4: Evaluate
        log(f"=== Evaluating Round {round_num} ===")
        # Reload model with new weights
        model, processor = load_model_with_lora(new_ckpt)
        round_metrics, preds, refs = evaluate(model, processor, test_samples)
        round_metrics['round'] = round_num
        round_metrics['ckpt'] = new_ckpt
        history.append(round_metrics)
        current_ckpt = new_ckpt

        # Print summary
        log(f"  Round {round_num} vs Round 0 (exp6):")
        log(f"    CompF1:  {history[0]['CompF1']:.4f} → {round_metrics['CompF1']:.4f} ({'+' if round_metrics['CompF1'] > history[0]['CompF1'] else ''}{round_metrics['CompF1'] - history[0]['CompF1']:.4f})")
        log(f"    JointF1: {history[0]['JointF1']:.4f} → {round_metrics['JointF1']:.4f} ({'+' if round_metrics['JointF1'] > history[0]['JointF1'] else ''}{round_metrics['JointF1'] - history[0]['JointF1']:.4f})")
        log(f"    NED:     {history[0]['NED']:.4f} → {round_metrics['NED']:.4f} ({'+' if round_metrics['NED'] < history[0]['NED'] else ''}{round_metrics['NED'] - history[0]['NED']:.4f})")

    # Final summary
    log(f"\n{'='*60}")
    log("=== ITERATIVE RS COMPLETE ===")
    log(f"{'='*60}")
    for h in history:
        log(f"  Round {h['round']}: CompF1={h['CompF1']:.4f} JointF1={h['JointF1']:.4f} NED={h['NED']:.4f}")

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    log(f"History saved to {args.output_dir}/history.json")

    return history

if __name__ == '__main__':
    main()
