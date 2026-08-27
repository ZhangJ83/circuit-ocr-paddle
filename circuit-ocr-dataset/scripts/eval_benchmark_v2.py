#!/usr/bin/env python3
"""
V2 Unified Evaluation Script (Phase 1)
======================================
Fixes vs V1 (eval_benchmark.py):
  1. UNIFIED INFERENCE: Always manual greedy decode with repetition_penalty=1.1
     — No more generate() vs manual fallback (eliminates OOM-dependent code paths)
  2. NEW METRICS:
     - exact_match_rate: pred == gt (primary metric)
     - component_f1: precision/recall/F1 on component refdes (R1, C2, U3, etc.)
     - token_recall: fraction of pred tokens present in gt
     - repetition_rate: fraction of samples with >=4 consecutive identical lines
     - NED: kept as reference (NOT primary)
  3. EARLY PATCH: flex_checkpoint dummy applied BEFORE paddleformers import
  4. Path resolution relative to data dir (no hardcoded absolute paths)

Usage:
  # Evaluate base model on full523:
  python eval_benchmark_v2.py --model_type paddleocr-vl \\
      --model_name_or_path "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27" \\
      --data_path ../ocr_vl_sft-test.jsonl

  # Evaluate V10-Fixed LoRA on full523:
  python eval_benchmark_v2.py --model_type paddleocr-vl-lora \\
      --model_name_or_path "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27" \\
      --paddle_lora_dir ../PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v10_fixed \\
      --data_path ../ocr_vl_sft-test.jsonl
"""

import os, sys, json, time, re, argparse
from pathlib import Path
from collections import Counter

# ── Early patch: flex_checkpoint for Paddle 3.1.0 compatibility ──
from types import ModuleType
_dummy_fc = ModuleType('dummy_flex_checkpoint')
_dummy_fc.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _dummy_fc)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _dummy_fc)

# Prepend matching CUDA/cuDNN DLL paths
dll_paths = [
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin",
    r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\lib\site-packages\torch\lib",
    r"E:\080000software\080900_Miniconda\miniconda3\pkgs\cudatoolkit-11.3.1-h59b6b97_2\Library\bin"
]
existing_dll_paths = [p for p in dll_paths if os.path.exists(p)]
if existing_dll_paths:
    os.environ["PATH"] = ";".join(existing_dll_paths) + ";" + os.environ.get("PATH", "")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
local_hf_cache = "F:/hf_cache/hub"
local_paddle_cache = "F:/paddle_cache"
if os.path.exists(local_hf_cache):
    os.environ["HF_HOME"] = local_hf_cache
    os.environ["HF_HUB_CACHE"] = local_hf_cache
if os.path.exists(local_paddle_cache):
    os.environ["PADDLE_HOME"] = local_paddle_cache

# Monkey-patch huggingface_hub
try:
    import huggingface_hub.constants
    if os.path.exists(local_hf_cache):
        huggingface_hub.constants.HF_HOME = "F:/hf_cache"
        huggingface_hub.constants.HF_HUB_CACHE = "F:/hf_cache/hub"
except Exception:
    pass

import Levenshtein
from PIL import Image
from io import BytesIO


# ==================== Paddle Patches (from eval_benchmark.py) ====================
def apply_paddle_patches():
    try:
        from types import ModuleType
        import paddle

        # Patch 0: PySafeSlice.shape
        try:
            from safetensors import safe_open as _safe_open
            _orig_safe_open = _safe_open
            def _patched_safe_open(*args, **kwargs):
                result = _orig_safe_open(*args, **kwargs)
                if len(result.keys()) > 0:
                    sl = result.get_slice(list(result.keys())[0])
                    if not hasattr(type(sl), 'shape'):
                        type(sl).shape = property(lambda self: self.get_shape())
                return result
            import safetensors
            safetensors.safe_open = _patched_safe_open
        except Exception:
            pass

        # Patch 0.1: LocalSharedLayerDesc
        try:
            import paddle.distributed.fleet.meta_parallel as _mp
            if not hasattr(_mp, 'LocalSharedLayerDesc') and hasattr(_mp, 'SharedLayerDesc'):
                _mp.LocalSharedLayerDesc = _mp.SharedLayerDesc
        except Exception:
            pass

        # Patch 0.2: swiglu
        try:
            import paddle.nn.functional as _pF
            if not hasattr(_pF, 'swiglu'):
                def _swiglu_impl(x, gate=None):
                    if gate is None:
                        split_dim = x.shape[-1] // 2
                        x_up, x_gate = x[..., :split_dim], x[..., split_dim:]
                    else:
                        x_gate, x_up = gate, x
                    return _pF.silu(x_gate) * x_up
                _pF.swiglu = _swiglu_impl
        except Exception:
            pass

        # Patch 0.3a: FLAGS_enable_auto_parallel_align_mode
        try:
            paddle.set_flags({'FLAGS_enable_auto_parallel_align_mode': False})
        except Exception:
            pass

        # Patch 0.3: fused_rms_norm_ext
        try:
            import paddle.incubate.nn.functional as _incF
            if not hasattr(_incF, 'fused_rms_norm_ext') and hasattr(_incF, 'fused_rms_norm'):
                _incF.fused_rms_norm_ext = _incF.fused_rms_norm
        except Exception:
            pass

        # flex_checkpoint already patched above

        paddle.LongTensor = paddle.Tensor
        paddle.linalg.fp8_fp8_half_gemm_fused = None
        paddle.Tensor.long = lambda self: self.astype("int64")
        paddle.Tensor.float = lambda self: self.astype("float32")
        paddle.Tensor.half = lambda self: self.astype("float16")

        # Patch reshape/view
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

        # Patch transpose
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

        # Patch masked_scatter
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

        # Patch get_flags
        old_get_flags = paddle.base.framework.get_flags
        def patched_get_flags(flags):
            if isinstance(flags, str):
                flags = [flags]
            res = {}
            for f in flags:
                if f == "FLAGS_flash_attn_version":
                    res[f] = 2
                elif f == "FLAGS_enable_auto_parallel_align_mode":
                    res[f] = False
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

        # Patch gelu
        old_gelu = paddle.nn.functional.gelu
        def patched_gelu(x, approximate=False, name=None):
            if isinstance(approximate, str):
                approximate = (approximate == 'tanh')
            return old_gelu(x, approximate, name)
        paddle.nn.functional.gelu = patched_gelu

        # Patch tensor creation
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
                    q_tr, k_tr, v_tr, attn_mask=attn_mask, is_causal=use_causal, training=False)
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

        import paddle.nn.functional.flash_attention as fa
        fa.flashmask_attention = fallback_flashmask_attention

        import paddle.incubate.tensor.manipulation as m
        m.create_async_load = lambda *args, **kwargs: None

        import paddle.distributed.fleet.meta_parallel as mp
        mp.LocalSharedLayerDesc = mp.SharedLayerDesc

        # Patch safetensors PySafeSlice
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


# ==================== Argument Parsing ====================
def parse_args():
    parser = argparse.ArgumentParser(description="Circuit VLM Benchmark V2 — Unified inference + new metrics")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["paddleocr-vl", "paddleocr-vl-lora"],
                        help="paddleocr-vl (base) or paddleocr-vl-lora (with LoRA)")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="HF model path (local snapshot)")
    parser.add_argument("--paddle_lora_dir", type=str, default=None,
                        help="LoRA weights directory or .pdparams file (for paddleocr-vl-lora)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Test dataset path (.jsonl)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output file (auto-generated if not specified)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max generation tokens")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty (default 1.1)")
    parser.add_argument("--unordered", action="store_true", default=False,
                        help="Sort lines before comparison")
    return parser.parse_args()


# ==================== NEW METRICS ====================
# Component refdes patterns: R1, C2, L3, D4, Q5, U6, J7, etc.
COMPONENT_PATTERN = re.compile(
    r'\b([RCLDQUJYTF]|[A-Z]{2,})\d+\b', re.IGNORECASE)

def extract_components(text):
    """Extract set of component refdes from text (e.g., {'R1', 'C2', 'U3'})."""
    return set(m.group(0) for m in COMPONENT_PATTERN.finditer(text))


def compute_exact_match(predictions, references):
    """Fraction of predictions that exactly match reference (after strip)."""
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions) if predictions else 0.0, matches


def compute_component_f1(predictions, references):
    """Component-level precision, recall, F1."""
    tp_total, fp_total, fn_total = 0, 0, 0
    for pred, ref in zip(predictions, references):
        pred_comps = extract_components(pred)
        ref_comps = extract_components(ref)
        tp = len(pred_comps & ref_comps)
        fp = len(pred_comps - ref_comps)
        fn = len(ref_comps - pred_comps)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total}


def compute_token_recall(predictions, references):
    """Fraction of pred tokens that appear in gt."""
    total_pred_tokens = 0
    matched_tokens = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        total_pred_tokens += len(pred_tokens)
        matched_tokens += len(pred_tokens & ref_tokens)
    return matched_tokens / total_pred_tokens if total_pred_tokens > 0 else 0.0


def compute_repetition_rate(predictions, min_repeat=4):
    """Fraction of samples with >= min_repeat consecutive identical lines."""
    count = 0
    for pred in predictions:
        lines = pred.strip().split('\n')
        if len(lines) < min_repeat:
            continue
        max_run = 1
        current_run = 1
        for i in range(1, len(lines)):
            if lines[i].strip() == lines[i-1].strip():
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        if max_run >= min_repeat:
            count += 1
    return count / len(predictions) if predictions else 0.0, count


def compute_ned(predictions, references, unordered=False):
    """Normalized Edit Distance (Levenshtein). Lower is better."""
    total_ned = 0
    for pred, ref in zip(predictions, references):
        if unordered:
            pred = "\n".join(sorted(pred.split("\n")))
            ref = "\n".join(sorted(ref.split("\n")))
        dist = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            total_ned += dist / max_len
    return total_ned / len(predictions) if predictions else 0.0


# ==================== Unified Manual Greedy Decoder ====================
def manual_greedy_decode_with_penalty(model, inputs, processor,
                                       max_new_tokens=256,
                                       repetition_penalty=1.1,
                                       eos_token_id=2):
    """UNIFIED inference: always manual greedy decode with repetition_penalty.

    No more model.generate() fallback — eliminates OOM-dependent code paths.
    """
    import paddle
    current_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    fwd_kwargs = {"use_cache": False}
    if pixel_values is not None:
        fwd_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        fwd_kwargs["image_grid_thw"] = image_grid_thw

    input_len = current_ids.shape[1]
    generated_ids = []

    for _ in range(max_new_tokens):
        fwd_kwargs["input_ids"] = current_ids
        outputs = model(**fwd_kwargs)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        next_token_logits = logits[0, -1, :]

        # Apply repetition_penalty
        if repetition_penalty != 1.0 and generated_ids:
            for tid in set(generated_ids):
                score = next_token_logits[tid].item()
                if score < 0:
                    next_token_logits[tid] = score * repetition_penalty
                else:
                    next_token_logits[tid] = score / repetition_penalty

        next_token_id = int(paddle.argmax(next_token_logits, axis=-1).item())

        if next_token_id == eos_token_id:
            break

        generated_ids.append(next_token_id)
        current_ids = paddle.concat(
            [current_ids, paddle.to_tensor([[next_token_id]], dtype=current_ids.dtype)],
            axis=1
        )

    full_ids = paddle.concat(
        [inputs["input_ids"][:, :input_len],
         paddle.to_tensor([generated_ids], dtype=inputs["input_ids"].dtype)],
        axis=1
    ) if generated_ids else inputs["input_ids"][:, :input_len]
    return processor.decode(full_ids[0][input_len:], skip_special_tokens=True)


# ==================== Main Evaluation ====================
def evaluate(args):
    print("Applying Paddle compatibility patches...")
    apply_paddle_patches()

    print("Loading PaddleOCR-VL libraries...")
    import paddle
    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    print(f"Setting Paddle device to: {device}")
    paddle.set_device(device)

    # Resolve model path
    model_path = args.model_name_or_path
    local_processor_path = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
    processor_path = local_processor_path if os.path.exists(local_processor_path) else model_path
    processor = AutoProcessor.from_pretrained(processor_path)

    print(f"Loading model from: {model_path}")
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path,
        convert_from_hf=True,
        load_checkpoint_format='naive',
        low_cpu_mem_usage=True,
        dtype="bfloat16"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    model.eval()

    # Load LoRA if specified
    if args.paddle_lora_dir:
        print(f"Loading PaddleOCR-VL LoRA from: {args.paddle_lora_dir}")
        import numpy as np
        LORA_SCALE = 2.0  # alpha/r = 32/16

        # Find LoRA file
        lora_path = Path(args.paddle_lora_dir)
        if lora_path.is_file():
            lora_file = str(lora_path)
        else:
            # Find best/final checkpoint
            candidates = [
                lora_path / "lora_best_v10_fixed_fp16.pdparams",
                lora_path / "lora_v10_fixed_final_fp16.pdparams",
                lora_path / "lora_best_v9_pure_fp16.pdparams",
                lora_path / "lora_v9_pure_final_fp16.pdparams",
                lora_path / "lora_final_fp16.pdparams",
                lora_path / "lora_weights_f32.pdparams",
            ]
            lora_file = None
            for c in candidates:
                if c.exists():
                    lora_file = str(c)
                    break
            if lora_file is None:
                # Try to find any .pdparams
                pdparams = list(lora_path.glob("*.pdparams"))
                if pdparams:
                    lora_file = str(pdparams[0])
                else:
                    raise FileNotFoundError(f"No .pdparams found in {args.paddle_lora_dir}")
        print(f"  Source: {lora_file}")

        lora_state = paddle.load(lora_file)
        lora_pairs = {}
        for k, v in lora_state.items():
            if k.endswith('.lora_A'):
                base_name = k[:-len('.lora_A')]
                clean_base = base_name[6:] if base_name.startswith('model.') else base_name
                lora_pairs.setdefault(clean_base, {})['A'] = v.numpy()
                lora_pairs[clean_base]['_orig_key'] = k
            elif k.endswith('.lora_B'):
                base_name = k[:-len('.lora_B')]
                clean_base = base_name[6:] if base_name.startswith('model.') else base_name
                lora_pairs.setdefault(clean_base, {})['B'] = v.numpy()

        print(f"  Found {len(lora_pairs)} LoRA adapter pairs")

        base_params = {}
        for n, p in model.named_parameters():
            base_params[n] = p

        merged = 0
        skipped_no_match = 0
        skipped_shape = 0
        for lora_base, adapters in lora_pairs.items():
            if 'A' not in adapters or 'B' not in adapters:
                skipped_no_match += 1
                continue
            lora_A = adapters['A']
            lora_B = adapters['B']
            weight_key = f"{lora_base}.weight"
            if weight_key not in base_params:
                skipped_no_match += 1
                continue
            p = base_params[weight_key]
            W = p.numpy()

            if lora_A.shape[-1] != lora_B.shape[0]:
                skipped_shape += 1
                continue

            delta = lora_A @ lora_B * LORA_SCALE
            if delta.shape == W.shape:
                W_new = W + delta.astype('float32')
            elif delta.shape[0] == W.shape[1] and delta.shape[1] == W.shape[0]:
                W_new = W + delta.T.astype('float32')
            elif delta.shape[0] == W.shape[0] and delta.shape[1] > W.shape[1]:
                W_new = W + delta[:, :W.shape[1]].astype('float32')
            elif delta.shape[0] < W.shape[0] and W.shape[0] % delta.shape[0] == 0:
                rep = W.shape[0] // delta.shape[0]
                W_new = W + np.tile(delta.astype('float32'), (rep, 1))
            elif delta.shape[0] == W.shape[0] and delta.shape[1] < W.shape[1] and W.shape[1] % delta.shape[1] == 0:
                rep = W.shape[1] // delta.shape[1]
                W_new = W + np.tile(delta.astype('float32'), (1, rep))
            else:
                skipped_shape += 1
                if skipped_shape <= 5:
                    print(f"  SKIP shape: {weight_key} delta={delta.shape} vs W={W.shape}")
                continue

            param_dtype = p.dtype
            p.set_value(paddle.to_tensor(W_new.astype('float16'), dtype=param_dtype, place=p.place))
            merged += 1

        model.eval()
        print(f"  Merged {merged}/{len(lora_pairs)} LoRA adapters (no_match={skipped_no_match}, shape={skipped_shape})")

    # Load test data
    data_dir = Path(args.data_path).parent
    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]

    # Auto-generate output path
    if args.output_path is None:
        model_tag = "base"
        if args.paddle_lora_dir:
            lora_name = Path(args.paddle_lora_dir).name
            model_tag = f"lora_{lora_name}"
        data_name = Path(args.data_path).stem
        args.output_path = f"results_v2_{model_tag}_{data_name}.jsonl"

    print(f"Loaded {len(samples)} test samples. Running inference with:")
    print(f"  repetition_penalty={args.repetition_penalty}")
    print(f"  max_new_tokens={args.max_length}")
    print(f"  method=manual_greedy_decode (UNIFIED)")

    results = []
    for i, sample in enumerate(samples):
        start = time.time()
        query = sample["messages"][0]["content"]
        image_path = sample["images"][0]

        # Resolve image path
        img_resolved_path = Path(image_path)
        if not img_resolved_path.exists():
            alt_path = data_dir / image_path
            if alt_path.exists():
                img_resolved_path = alt_path
            else:
                alt_path2 = data_dir / image_path.lstrip("./")
                if alt_path2.exists():
                    img_resolved_path = alt_path2
                else:
                    img_resolved_path = data_dir / img_resolved_path.name

        image = None
        try:
            image = Image.open(img_resolved_path).convert("RGB")
            w, h = image.size
            max_dim = 384
            if w > max_dim or h > max_dim:
                scale = max_dim / max(w, h)
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = BytesIO()
            image.save(buf, format='JPEG', quality=95)
            buf.seek(0)
            image = Image.open(buf)

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

            with paddle.no_grad():
                output_text = manual_greedy_decode_with_penalty(
                    model, inputs, processor,
                    max_new_tokens=args.max_length,
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=2
                )

            sample["prediction"] = output_text
            sample["label"] = sample["messages"][1]["content"]
            results.append(sample)

            # Incremental save
            with open(args.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            elapsed = time.time() - start
            pred_preview = output_text[:60].replace('\n', '\\n')
            print(f"[{i+1}/{len(samples)}] OK {img_resolved_path.name} {elapsed:.1f}s pred={pred_preview}...")

        except Exception as e:
            elapsed = time.time() - start
            print(f"[{i+1}/{len(samples)}] FAIL {img_resolved_path.name} {elapsed:.1f}s: {type(e).__name__}: {e}")
            sample["prediction"] = ""
            sample["label"] = sample["messages"][1]["content"]
            results.append(sample)
            with open(args.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        finally:
            if image is not None:
                image.close()
            import gc
            gc.collect()
            paddle.device.cuda.empty_cache()
            paddle.device.cuda.synchronize()
            sys.stdout.flush()

    return results


# ==================== Report ====================
def main():
    args = parse_args()
    start_time = time.time()

    # Clear output file
    output_file = Path(args.output_path) if args.output_path else None
    if args.output_path is None:
        model_tag = "base"
        if args.paddle_lora_dir:
            lora_name = Path(args.paddle_lora_dir).name
            model_tag = f"lora_{lora_name}"
        data_name = Path(args.data_path).stem
        args.output_path = f"results_v2_{model_tag}_{data_name}.jsonl"

    output_file = Path(args.output_path)
    if output_file.exists():
        output_file.unlink()

    results = evaluate(args)

    # Compute all metrics
    predictions = [r["prediction"] for r in results]
    references = [r["label"] for r in results]
    n = len(results)

    em_rate, em_count = compute_exact_match(predictions, references)
    comp_metrics = compute_component_f1(predictions, references)
    tok_recall = compute_token_recall(predictions, references)
    rep_rate, rep_count = compute_repetition_rate(predictions)
    avg_ned = compute_ned(predictions, references, unordered=args.unordered)
    avg_ned_ordered = compute_ned(predictions, references, unordered=False)

    # ── Print Report ──
    print("\n" + "=" * 60)
    print("           V2 Evaluation Report (Phase 1)")
    print("=" * 60)
    print(f"  Model:        {args.model_type}")
    if args.paddle_lora_dir:
        print(f"  LoRA:         {args.paddle_lora_dir}")
    print(f"  Test set:     {Path(args.data_path).name} ({n} samples)")
    print(f"  Rep. penalty: {args.repetition_penalty}")
    print(f"  Unordered:    {args.unordered}")
    print("-" * 60)
    print(f"  ★ exact_match:    {em_count}/{n} = {em_rate:.4f} ({em_rate*100:.1f}%)")
    print(f"  ★ component F1:   {comp_metrics['f1']:.4f} (P={comp_metrics['precision']:.4f}, R={comp_metrics['recall']:.4f})")
    print(f"    component TP/FP/FN: {comp_metrics['tp']}/{comp_metrics['fp']}/{comp_metrics['fn']}")
    print(f"  ★ token_recall:   {tok_recall:.4f} ({tok_recall*100:.1f}%)")
    print(f"  ★ repetition_rate: {rep_count}/{n} = {rep_rate:.4f} ({rep_rate*100:.1f}%)")
    print(f"    NED (ordered):   {avg_ned_ordered:.4f}")
    print(f"    NED (unordered): {avg_ned:.4f}")
    print("=" * 60)
    print(f"  Results: {args.output_path}")
    print(f"  Time:    {time.time() - start_time:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
