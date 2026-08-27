#!/usr/bin/env python3
"""
WSL GPU-mode PaddleOCR-VL Benchmark Script
===========================================
Runs PaddleOCR-VL-0.9B on GPU via conda CUDA 11.8 toolkit + cuDNN 8.9.
Requires: LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib
"""
import os, sys

# WSL GPU paths
os.environ["HF_HOME"] = "/mnt/f/hf_cache/hub"
os.environ["PADDLE_HOME"] = "/mnt/f/paddle_cache"
os.environ["HF_HUB_CACHE"] = "/mnt/f/hf_cache/hub"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# DO NOT set CUDA_VISIBLE_DEVICES="" — we want GPU

import argparse, json, time
from pathlib import Path
from PIL import Image
import Levenshtein


def apply_paddle_patches():
    """Apply Paddle compatibility patches for PaddlePaddle 2.6.x."""
    try:
        from types import ModuleType
        import paddle

        try:
            import paddle.distributed.flex_checkpoint.dcp.sharded_weight
        except (ImportError, ModuleNotFoundError, AttributeError):
            dummy = ModuleType('dummy')
            dummy.build_sharded_state_dict = lambda *a, **kw: None
            sys.modules.setdefault('paddle.distributed.flex_checkpoint', dummy)
            sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', dummy)
            sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', dummy)

        paddle.float8_e4m3fn = paddle.float32
        paddle.float8_e5m2 = paddle.float32
        paddle.LongTensor = paddle.Tensor
        paddle.linalg.fp8_fp8_half_gemm_fused = None
        paddle.Tensor.long = lambda self: self.astype("int64")
        paddle.Tensor.float = lambda self: self.astype("float32")
        paddle.Tensor.half = lambda self: self.astype("float16")

        old_reshape = paddle.Tensor.reshape
        old_view = paddle.Tensor.view
        def patched_view(self, *args, **kwargs):
            if args and isinstance(args[0], paddle.dtype):
                return old_view(self, *args, **kwargs)
            if args:
                if len(args) > 1:
                    new_shape = list(args)
                elif len(args) == 1 and (isinstance(args[0], int) or hasattr(args[0], '__index__')):
                    new_shape = [int(args[0])]
                else:
                    new_shape = args[0]
                return old_reshape(self, new_shape, **kwargs)
            return old_reshape(self, **kwargs)
        paddle.Tensor.reshape = patched_view
        paddle.Tensor.view = patched_view

        if not hasattr(paddle.Tensor, "repeat"):
            paddle.Tensor.repeat = paddle.Tensor.tile

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

        old_get_flags = paddle.base.framework.get_flags
        def patched_get_flags(flags):
            res = {}
            for f in flags:
                if f == "FLAGS_flash_attn_version":
                    res[f] = 2
                else:
                    try:
                        res[f] = old_get_flags([f])[f]
                    except Exception:
                        res[f] = None
            return res
        paddle.base.framework.get_flags = patched_get_flags

        old_set_flags = paddle.set_flags
        def patched_set_flags(flags_dict):
            try:
                filtered = {k: v for k, v in flags_dict.items() if k != "FLAGS_flash_attn_version"}
                if filtered:
                    old_set_flags(filtered)
            except Exception:
                pass

        old_gelu = paddle.nn.functional.gelu
        def patched_gelu(x, approximate=False, name=None):
            if isinstance(approximate, str):
                approximate = (approximate == 'tanh')
            return old_gelu(x, approximate, name)
        paddle.nn.functional.gelu = patched_gelu

        def patch_creation_func(func_name):
            old_func = getattr(paddle, func_name)
            def patched(*args, **kwargs):
                kwargs.pop('device', None)
                return old_func(*args, **kwargs)
            setattr(paddle, func_name, patched)
        for name in ['empty', 'zeros', 'ones', 'arange', 'full', 'randn', 'rand']:
            if hasattr(paddle, name):
                patch_creation_func(name)

        import paddle.nn.functional as pnf
        pnf.swiglu = lambda *args, **kwargs: None

        def fallback_fused_rms_norm_ext(x, weight, epsilon=1e-6):
            variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
            rsqrt = paddle.rsqrt(variance + epsilon)
            normalized = x * rsqrt * weight
            return (normalized, rsqrt)

        import paddle.incubate.nn.functional as pinf
        pinf.fused_rms_norm_ext = fallback_fused_rms_norm_ext

        def fallback_flashmask_attention(q, k, v, startend_row_indices=None, causal=True):
            q_tr = q.transpose([0, 2, 1, 3])
            k_tr = k.transpose([0, 2, 1, 3])
            v_tr = v.transpose([0, 2, 1, 3])
            b, h_q, l_q, d = q_tr.shape
            _, h_k, l_k, _ = k_tr.shape
            if h_q != h_k:
                n_rep = h_q // h_k
                k_tr = k_tr.reshape([b, h_k, 1, l_k, d])
                k_tr = paddle.tile(k_tr, [1, 1, n_rep, 1, 1])
                k_tr = k_tr.reshape([b, h_q, l_k, d])
                v_tr = v_tr.reshape([b, h_k, 1, l_k, d])
                v_tr = paddle.tile(v_tr, [1, 1, n_rep, 1, 1])
                v_tr = v_tr.reshape([b, h_q, l_k, d])
            attn_mask = None
            use_causal = False
            if startend_row_indices is not None:
                if startend_row_indices.shape[-1] == 1:
                    startend_row_indices = startend_row_indices.squeeze(-1)
                if startend_row_indices.ndim == 3:
                    se = startend_row_indices
                    mask = paddle.full([b, 1, l_q, l_k], -1e9, dtype=q.dtype)
                    if se.shape[-1] == 2:
                        starts, ends = se[..., 0], se[..., 1]
                        pos = paddle.arange(l_k, dtype='int32').reshape([1, 1, 1, l_k])
                        valid = (pos >= starts.unsqueeze(-1)) & (pos < ends.unsqueeze(-1))
                        mask = paddle.where(valid, paddle.zeros_like(mask), mask)
                    elif se.shape[-1] == 4:
                        for slot in range(2):
                            s = se[..., slot * 2]
                            e = se[..., slot * 2 + 1]
                            pos = paddle.arange(l_k, dtype='int32').reshape([1, 1, 1, l_k])
                            valid = (pos >= s.unsqueeze(-1)) & (pos < e.unsqueeze(-1))
                            mask = paddle.where(valid, paddle.zeros_like(mask), mask)
                    attn_mask = mask
                    causal = False
            elif causal and l_q != l_k:
                row_idx = paddle.arange(l_q, dtype='int32').reshape([1, 1, l_q, 1])
                col_idx = paddle.arange(l_k, dtype='int32').reshape([1, 1, 1, l_k])
                causal_bool = col_idx <= (l_k - l_q + row_idx)
                attn_mask = paddle.where(
                    causal_bool,
                    paddle.zeros([1, 1, l_q, l_k], dtype=q.dtype),
                    paddle.full([1, 1, l_q, l_k], -1e9, dtype=q.dtype)
                )
                if b > 1:
                    attn_mask = paddle.tile(attn_mask, [b, 1, 1, 1])
            else:
                use_causal = causal
            try:
                out_tr = paddle.nn.functional.scaled_dot_product_attention(
                    q_tr, k_tr, v_tr, attn_mask=attn_mask, is_causal=use_causal, training=False
                )
            except Exception:
                scores = paddle.matmul(q_tr, k_tr.transpose([0, 1, 3, 2])) / (d ** 0.5)
                if attn_mask is not None:
                    scores = scores + attn_mask
                if use_causal:
                    grid_q = paddle.arange(l_q, dtype="int32").reshape([l_q, 1])
                    grid_k = paddle.arange(l_k, dtype="int32").reshape([1, l_k])
                    tril_mask = (grid_k - grid_q) <= (l_k - l_q)
                    scores = paddle.where(tril_mask, scores, paddle.to_tensor(-1e9, dtype=scores.dtype))
                attn_weights = paddle.nn.functional.softmax(scores, axis=-1)
                out_tr = paddle.matmul(attn_weights, v_tr)
            return out_tr.transpose([0, 2, 1, 3])

        import paddleformers.nn.attention.eager_attention as ea
        old_repeat_kv = ea.repeat_kv
        def patched_repeat_kv(hidden_states, n_rep):
            batch, num_key_value_heads, slen, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
            return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])
        ea.repeat_kv = patched_repeat_kv

        old_expand = paddle.Tensor.expand
        def patched_expand(self, *args, **kwargs):
            if len(args) > 1:
                shape = list(args)
                return old_expand(self, shape, **kwargs)
            return old_expand(self, *args, **kwargs)
        paddle.Tensor.expand = patched_expand

        import paddle.nn.functional.flash_attention as fa
        fa.flashmask_attention = fallback_flashmask_attention

        import paddle.incubate.tensor.manipulation as m
        m.create_async_load = lambda *args, **kwargs: None

        import paddle.distributed.fleet.meta_parallel as mp
        mp.LocalSharedLayerDesc = mp.SharedLayerDesc

        try:
            import numpy as np
            import tempfile
            from safetensors.numpy import save_file, safe_open
            tmp_path = tempfile.mktemp(suffix='.safetensors')
            save_file({'dummy': np.zeros((1,))}, tmp_path)
            with safe_open(tmp_path, framework='np') as f:
                PySafeSlice = type(f.get_slice('dummy'))
                setattr(PySafeSlice, 'shape', property(lambda self: self.get_shape()))
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: Failed to apply Paddle compatibility patches: {e}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(description="WSL GPU PaddleOCR-VL Benchmark")
    parser.add_argument("--model_type", type=str, default="paddleocr-vl")
    parser.add_argument("--model_name_or_path", type=str, default="PaddlePaddle/PaddleOCR-VL")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="results.jsonl")
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


def compute_metrics(predictions, references):
    total_ned = 0
    num_samples = len(predictions)
    if num_samples == 0:
        return 0.0
    for pred, ref in zip(predictions, references):
        dist = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            total_ned += dist / max_len
    return total_ned / num_samples


def save_incremental(output_path, sample):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def evaluate_paddleocr_vl(args):
    print("Applying Paddle compatibility patches...")
    apply_paddle_patches()

    print("Loading PaddleOCR-VL libraries...")
    import paddle
    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
    from paddleformers.generation import GenerationConfig

    # GPU mode
    device = "gpu"
    print(f"Setting Paddle device to: {device}")
    paddle.set_device(device)

    local_model_path = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"
    print(f"Loading model from: {local_model_path}")
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = AutoModelForConditionalGeneration.from_pretrained(
        local_model_path,
        convert_from_hf=True,
        load_checkpoint_format='naive',
        low_cpu_mem_usage=True,
        dtype="float16"  # GPU float16 for speed
    )
    model.config._attn_implementation = "eager"
    if hasattr(model, "visual") and hasattr(model.visual, "config"):
        model.visual.config._attn_implementation = "eager"
    model.eval()

    generation_config = GenerationConfig(
        do_sample=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True
    )

    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]

    # Resume support
    already_processed = set()
    if args.resume and Path(args.output_path).exists():
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if "images" in d:
                        already_processed.add(tuple(d["images"]))
        print(f"Resuming: {len(already_processed)} already processed")

    samples_to_run = [s for s in samples if tuple(s["images"]) not in already_processed]
    print(f"Loaded {len(samples)} total, {len(samples_to_run)} to process")

    results = []
    for i, sample in enumerate(samples_to_run):
        orig_idx = i + len(already_processed)
        start = time.time()
        image_path = sample["images"][0].replace("\\", "/")
        img_resolved_path = Path(args.data_path).parent / image_path
        if not img_resolved_path.exists():
            img_resolved_path = Path(args.data_path).parent / Path(image_path).name

        image = None
        try:
            image = Image.open(img_resolved_path).convert("RGB")
            query = sample["messages"][0]["content"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": query.replace("<image>", "")},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pd"
            )

            with paddle.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config,
                                        max_new_tokens=args.max_length)
                output_ids = outputs[0].tolist()[0]
                output_text = processor.decode(output_ids, skip_special_tokens=True)

            sample["prediction"] = output_text
            sample["label"] = sample["messages"][1]["content"]
            results.append(sample)
            save_incremental(args.output_path, sample)
            elapsed = time.time() - start
            print(f"[{orig_idx+1}/{len(samples)}] OK {img_resolved_path.name} {elapsed:.1f}s pred_len={len(output_text)}")
        except Exception as e:
            elapsed = time.time() - start
            print(f"[{orig_idx+1}/{len(samples)}] FAIL {img_resolved_path.name} {elapsed:.1f}s: {type(e).__name__}: {e}", file=sys.stderr)
            sample["prediction"] = ""
            sample["label"] = sample["messages"][1]["content"]
            results.append(sample)
            save_incremental(args.output_path, sample)
        finally:
            if image is not None:
                image.close()
            sys.stdout.flush()

    return results


def main():
    args = parse_args()
    start_time = time.time()

    output_file = Path(args.output_path)
    if output_file.exists() and not args.resume:
        output_file.unlink()

    results = evaluate_paddleocr_vl(args)

    # Compute metrics
    all_results = []
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    predictions = [res["prediction"] for res in all_results]
    references = [res["label"] for res in all_results]
    avg_ned = compute_metrics(predictions, references)

    print("\n" + "=" * 40)
    print("        Evaluation Report")
    print("=" * 40)
    print(f"Model Type: paddleocr-vl (WSL GPU)")
    print(f"Samples:    {len(all_results)}")
    print(f"Avg. NED:   {avg_ned:.4f} (Lower is better)")
    print("=" * 40)
    print(f"Results saved to: {output_file.absolute()}")
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
