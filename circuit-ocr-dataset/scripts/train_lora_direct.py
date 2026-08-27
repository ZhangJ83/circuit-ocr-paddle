#!/usr/bin/env python3
"""
PaddleOCR-VL LoRA Fine-Tuning — Pure Python, No CLI
=====================================================
Bypasses paddleformers-cli (which requires Paddle 3.x APIs).
Uses paddleformers.transformers (proven working in eval_benchmark.py).

LoRA implementation: manual low-rank adaptation using PaddlePaddle.
"""

import os, sys, json, time, math, argparse
from pathlib import Path
from datetime import datetime

# === Environment ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "/mnt/f/hf_cache/hub"
os.environ["PADDLE_HOME"] = "/mnt/f/paddle_cache"
os.environ["HF_HUB_CACHE"] = "/mnt/f/hf_cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# CUDA libs for WSL
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/home/zzz/miniconda3/lib:/usr/lib/wsl/lib"

import paddle

# === Apply Paddle patches (same as eval_benchmark.py) ===
def apply_patches():
    from types import ModuleType
    import paddle

    # flex_checkpoint
    try:
        import paddle.distributed.flex_checkpoint.dcp.sharded_weight
    except (ImportError, ModuleNotFoundError, AttributeError):
        dummy = ModuleType('dummy')
        for func in ['build_sharded_state_dict', 'create_sharded_weight_with_new_local',
                      'reshape_sharded_weight', 'sharded_weight_parallel_cpu',
                      'save_state_dict', 'load_state_dict']:
            setattr(dummy, func, lambda *a, **kw: None)
        for mod in ['paddle.distributed.flex_checkpoint',
                     'paddle.distributed.flex_checkpoint.dcp',
                     'paddle.distributed.flex_checkpoint.dcp.sharded_weight']:
            sys.modules.setdefault(mod, dummy)

    paddle.float8_e4m3fn = paddle.float32
    paddle.float8_e5m2 = paddle.float32
    paddle.LongTensor = paddle.Tensor
    paddle.linalg.fp8_fp8_half_gemm_fused = None
    paddle.Tensor.long = lambda self: self.astype("int64")
    paddle.Tensor.float = lambda self: self.astype("float32")
    paddle.Tensor.half = lambda self: self.astype("float16")

    # reshape/view patch
    old_reshape = paddle.Tensor.reshape
    def patched_reshape(self, *args, **kwargs):
        if args:
            if isinstance(args[0], paddle.dtype):
                # .view(dtype) or .reshape(dtype) => cast, not reshape
                return self.astype(args[0])
            if len(args) > 1: new_shape = list(args)
            elif len(args) == 1 and (isinstance(args[0], int) or hasattr(args[0], '__index__')):
                new_shape = [int(args[0])]
            else: new_shape = args[0]
            return old_reshape(self, new_shape, **kwargs)
        return old_reshape(self, **kwargs)
    paddle.Tensor.reshape = patched_reshape
    paddle.Tensor.view = patched_reshape
    if not hasattr(paddle.Tensor, "repeat"):
        paddle.Tensor.repeat = paddle.Tensor.tile

    # transpose
    old_transpose = paddle.Tensor.transpose
    def patched_transpose(self, *args, **kwargs):
        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            dim0, dim1 = args[0], args[1]
            ndim = self.ndim
            if dim0 < 0: dim0 += ndim
            if dim1 < 0: dim1 += ndim
            perm = list(range(ndim))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            return old_transpose(self, perm, **kwargs)
        return old_transpose(self, *args, **kwargs)
    paddle.Tensor.transpose = patched_transpose

    # masked_scatter
    def patched_masked_scatter(self, mask, source):
        orig_shape = self.shape
        mask = mask.astype('bool')
        flat_self = self.flatten()
        flat_mask = mask.flatten()
        flat_source = source.flatten()
        indices = paddle.nonzero(flat_mask)
        scattered = paddle.scatter_nd(indices, flat_source, flat_mask.shape)
        out_flat = paddle.where(flat_mask, scattered, flat_self)
        return out_flat.reshape(orig_shape)
    paddle.Tensor.masked_scatter = patched_masked_scatter

    # get_flags / set_flags
    old_get_flags = paddle.base.framework.get_flags
    def patched_get_flags(flags):
        res = {}
        for f in flags:
            if f == "FLAGS_flash_attn_version": res[f] = 2
            else:
                try: res[f] = old_get_flags([f])[f]
                except Exception: res[f] = None
        return res
    paddle.base.framework.get_flags = patched_get_flags

    old_set_flags = paddle.set_flags
    def patched_set_flags(flags_dict):
        try:
            filtered = {k: v for k, v in flags_dict.items() if k != "FLAGS_flash_attn_version"}
            if filtered: old_set_flags(filtered)
        except Exception: pass
    paddle.set_flags = patched_set_flags

    # gelu
    old_gelu = paddle.nn.functional.gelu
    def patched_gelu(x, approximate=False, name=None):
        if isinstance(approximate, str): approximate = (approximate == 'tanh')
        return old_gelu(x, approximate, name)
    paddle.nn.functional.gelu = patched_gelu

    # tensor creation
    for name in ['empty', 'zeros', 'ones', 'arange', 'full', 'randn', 'rand']:
        if hasattr(paddle, name):
            old_func = getattr(paddle, name)
            def make_patched(old):
                def patched(*args, **kwargs):
                    kwargs.pop('device', None)
                    return old(*args, **kwargs)
                return patched
            setattr(paddle, name, make_patched(old_func))

    import paddle.nn.functional as pnf
    pnf.swiglu = lambda *args, **kwargs: None

    def fallback_fused_rms_norm_ext(x, weight, epsilon=1e-6):
        variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
        rsqrt = paddle.rsqrt(variance + epsilon)
        return (x * rsqrt * weight, rsqrt)
    import paddle.incubate.nn.functional as pinf
    pinf.fused_rms_norm_ext = fallback_fused_rms_norm_ext

    # flashmask_attention
    def fallback_flashmask_attention(q, k, v, startend_row_indices=None, causal=True):
        q_tr = q.transpose([0, 2, 1, 3])
        k_tr = k.transpose([0, 2, 1, 3])
        v_tr = v.transpose([0, 2, 1, 3])
        b, h_q, l_q, d = q_tr.shape
        _, h_k, l_k, _ = k_tr.shape
        if h_q != h_k:
            n_rep = h_q // h_k
            k_tr = paddle.tile(k_tr.reshape([b, h_k, 1, l_k, d]), [1, 1, n_rep, 1, 1]).reshape([b, h_q, l_k, d])
            v_tr = paddle.tile(v_tr.reshape([b, h_k, 1, l_k, d]), [1, 1, n_rep, 1, 1]).reshape([b, h_q, l_k, d])
        use_causal = causal and (l_q == l_k)
        attn_mask = None
        if causal and not use_causal:
            row_idx = paddle.arange(l_q, dtype='int32').reshape([1, 1, l_q, 1])
            col_idx = paddle.arange(l_k, dtype='int32').reshape([1, 1, 1, l_k])
            causal_bool = col_idx <= (l_k - l_q + row_idx)
            attn_mask = paddle.where(causal_bool,
                paddle.zeros([1, 1, l_q, l_k], dtype=q.dtype),
                paddle.full([1, 1, l_q, l_k], -1e9, dtype=q.dtype))
            if b > 1: attn_mask = paddle.tile(attn_mask, [b, 1, 1, 1])
        try:
            return paddle.nn.functional.scaled_dot_product_attention(
                q_tr, k_tr, v_tr, attn_mask=attn_mask, is_causal=use_causal, training=False
            ).transpose([0, 2, 1, 3])
        except Exception:
            scores = paddle.matmul(q_tr, k_tr.transpose([0, 1, 3, 2])) / (d ** 0.5)
            if attn_mask is not None: scores = scores + attn_mask
            if use_causal:
                grid_q = paddle.arange(l_q, dtype="int32").reshape([l_q, 1])
                grid_k = paddle.arange(l_k, dtype="int32").reshape([1, l_k])
                scores = paddle.where((grid_k - grid_q) <= (l_k - l_q), scores, paddle.to_tensor(-1e9, dtype=scores.dtype))
            return paddle.matmul(paddle.nn.functional.softmax(scores, axis=-1), v_tr).transpose([0, 2, 1, 3])

    import paddle.nn.functional.flash_attention as fa
    fa.flashmask_attention = fallback_flashmask_attention

    import paddle.incubate.tensor.manipulation as m
    m.create_async_load = lambda *args, **kwargs: None

    import paddle.distributed.fleet.meta_parallel as mp
    if hasattr(mp, 'SharedLayerDesc'):
        mp.LocalSharedLayerDesc = mp.SharedLayerDesc

    print("[Patches] Applied successfully.")


class LoRALinear(paddle.nn.Layer):
    """Manual LoRA layer for PaddlePaddle: W' = W + B @ A * (alpha / r)"""
    def __init__(self, base_linear: paddle.nn.Linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        # LoRA weights
        self.lora_A = paddle.create_parameter(
            shape=[in_features, rank],
            dtype=base_linear.weight.dtype,
            default_initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(rank))
        )
        self.lora_B = paddle.create_parameter(
            shape=[rank, out_features],
            dtype=base_linear.weight.dtype,
            default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        self.dropout = paddle.nn.Dropout(dropout)

        # Freeze base
        for p in self.base.parameters():
            p.stop_gradient = True

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.05, target_modules=None):
    """Replace target Linear layers with LoRALinear wrappers."""
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    replaced = 0
    for name, layer in list(model.named_sublayers(include_self=False)):
        # Only target language model layers, NOT vision encoder
        if 'visual' in name or 'vision' in name:
            continue
        if any(t in name for t in target_modules):
            if isinstance(layer, paddle.nn.Linear):
                # Get parent
                parent_name = '.'.join(name.split('.')[:-1])
                leaf_name = name.split('.')[-1]
                if parent_name:
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                else:
                    parent = model
                    leaf_name = name

                lora_layer = LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, leaf_name, lora_layer)
                replaced += 1

    print(f"[LoRA] Replaced {replaced} Linear layers with LoRA (r={rank}, alpha={alpha})")

    # Count params
    total = sum(p.size for p in model.parameters())
    trainable = sum(p.size for p in model.parameters() if not p.stop_gradient)
    print(f"[LoRA] Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    return model


def load_data(jsonl_path, limit=None):
    """Load training/eval data."""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if limit:
        samples = samples[:limit]
    return samples


def compute_loss(model, processor, sample, generation_config, max_seq_len=2048):
    """Teacher-forcing loss on a single sample."""
    import paddle
    from PIL import Image
    from io import BytesIO

    query = sample["messages"][0]["content"]
    label = sample["messages"][1]["content"]
    image_path = sample["images"][0]

    # Resolve image relative to dataset directory
    dataset_dir = "/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset"
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = Path(dataset_dir) / image_path.lstrip('./')
    if not img_path.exists():
        img_path = Path(dataset_dir) / 'data' / 'train' / Path(image_path).name
        if not img_path.exists():
            img_path = Path(dataset_dir) / 'data' / 'test' / Path(image_path).name

    try:
        image = Image.open(str(img_path)).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query.replace("<image>", "")},
            ],
        }]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pd"
        )

        # Tokenize label
        label_tokens = processor.tokenizer(label, return_tensors="pd", padding=False,
                                           truncation=True, max_length=max_seq_len)
        label_ids = label_tokens["input_ids"][0]

        # Forward pass with teacher forcing
        outputs = model(**inputs, labels=label_ids.unsqueeze(0))
        loss = outputs.loss

        try: image.close()
        except: pass
        paddle.device.cuda.empty_cache()
        return loss

    except Exception as e:
        print(f"  [WARN] Loss computation failed for {img_path}: {e}", file=sys.stderr)
        return None


def evaluate_model(model, processor, eval_samples, generation_config, max_new_tokens=1024):
    """Run inference on eval set, compute avg loss."""
    import paddle
    from PIL import Image
    from io import BytesIO

    total_loss = 0.0
    count = 0
    model.eval()

    for sample in eval_samples:
        loss = compute_loss(model, processor, sample, generation_config)
        if loss is not None:
            total_loss += float(loss)
            count += 1

    model.train()
    return total_loss / count if count > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./PaddleOCR-VL-LoRA-circuit-ocr')
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--max_eval_samples', type=int, default=50)
    default_train = '/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/ocr_vl_sft-train.jsonl'
    default_eval = '/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/ocr_vl_sft-test.jsonl'
    parser.add_argument('--train_data', type=str, default=default_train)
    parser.add_argument('--eval_data', type=str, default=default_eval)
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    # Apply patches
    apply_patches()

    import paddle
    paddle.set_device('gpu')
    print(f"[Init] GPU: {paddle.device.cuda.get_device_name(0)}")
    print(f"[Init] Paddle: {paddle.__version__}")

    # Load model
    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
    from paddleformers.generation import GenerationConfig

    model_path = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"
    print(f"[Init] Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path, convert_from_hf=True, load_checkpoint_format='naive',
        low_cpu_mem_usage=True, dtype="bfloat16"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    # Apply LoRA
    model = apply_lora_to_model(model, rank=args.rank, alpha=args.alpha)

    # Load data
    train_samples = load_data(args.train_data)
    eval_samples = load_data(args.eval_data, limit=args.max_eval_samples)
    print(f"[Data] Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    # Optimizer + scheduler
    trainable_params = [p for p in model.parameters() if not p.stop_gradient]
    lr_schedule = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=args.epochs * len(train_samples) // args.grad_accum, eta_min=5e-5
    )
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_schedule, parameters=trainable_params,
        weight_decay=0.1, beta1=0.9, beta2=0.95, epsilon=1e-8
    )

    gen_config = GenerationConfig(
        do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    loss_history = []
    global_step = 0
    best_eval_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"  LoRA Fine-Tuning Started — r={args.rank}, epochs={args.epochs}")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Grad accum: {args.grad_accum} → {len(train_samples)//args.grad_accum} steps/epoch")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        step_loss = 0.0

        # Shuffle
        import random
        random.shuffle(train_samples)

        model.train()
        for i, sample in enumerate(train_samples):
            loss = compute_loss(model, processor, sample, gen_config)
            if loss is None:
                continue

            loss = loss / args.grad_accum
            loss.backward()
            step_loss += float(loss) * args.grad_accum

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_samples):
                optimizer.step()
                lr_schedule.step()
                optimizer.clear_grad()
                global_step += 1

                loss_history.append({
                    'step': global_step,
                    'epoch': epoch + 1,
                    'loss': step_loss / args.grad_accum,
                    'lr': optimizer.get_lr(),
                    'time': time.time() - total_start,
                })

                current_lr = optimizer.get_lr()
                elapsed = time.time() - total_start
                eta_total = (elapsed / global_step) * (args.epochs * len(train_samples) // args.grad_accum)
                eta_remaining = eta_total - elapsed

                print(f"[E{epoch+1} S{global_step:3d}] loss={step_loss/args.grad_accum:.4f} "
                      f"lr={current_lr:.2e} elapsed={elapsed/60:.0f}m ETA={eta_remaining/60:.0f}m")

                # Eval
                if global_step % args.eval_steps == 0:
                    print(f"  [Eval @ step {global_step}] Running...")
                    eval_start = time.time()
                    eval_loss = evaluate_model(model, processor, eval_samples, gen_config)
                    eval_time = time.time() - eval_start
                    print(f"  [Eval @ step {global_step}] loss={eval_loss:.4f} ({eval_time:.0f}s)")

                    # Save best
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        ckpt_path = os.path.join(args.output_dir, 'best_model')
                        paddle.save({
                            'model_state_dict': model.state_dict(),
                            'step': global_step,
                            'eval_loss': eval_loss,
                        }, ckpt_path + '.pdparams')
                        print(f"  [Save] Best model @ step {global_step} (eval_loss={eval_loss:.4f})")

                    model.train()

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                    paddle.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': global_step,
                        'epoch': epoch + 1,
                        'loss_history': loss_history,
                    }, ckpt_path + '.pdparams')
                    print(f"  [Save] Checkpoint @ step {global_step}")

                step_loss = 0.0

        epoch_time = (time.time() - epoch_start) / 60
        print(f"\n[Epoch {epoch+1}] Done in {epoch_time:.1f}m\n")

    total_time = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Total time: {total_time:.0f}m ({total_time/60:.1f}h)")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    print(f"  Model saved to: {args.output_dir}")
    print(f"{'='*60}")

    # Save final model + loss history
    paddle.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pdparams'))
    with open(os.path.join(args.output_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)

    print("Done.")


if __name__ == '__main__':
    main()
