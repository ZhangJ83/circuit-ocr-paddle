#!/usr/bin/env python3
"""
Unified VLM Benchmarking Script
===============================
Supports evaluating:
1. PaddlePaddle-based PaddleOCR-VL-0.9B
2. PyTorch-based Qwen3-VL-8B-Instruct (base & LoRA adapter)
"""

import os
import sys

# Prepend matching CUDA/cuDNN DLL paths to system PATH for PaddlePaddle compatibility
# cuDNN 8.9.2.26 installed via conda (Paddle 2.6.2 needs cuDNN >= 8.6, system had 8.2)
dll_paths = [
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin",  # cuDNN 8.9.2.26 DLLs
    r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\lib\site-packages\torch\lib",  # zlibwapi.dll
    r"E:\080000software\080900_Miniconda\miniconda3\pkgs\cudatoolkit-11.3.1-h59b6b97_2\Library\bin"
]
os.environ["PATH"] = ";".join(dll_paths) + ";" + os.environ.get("PATH", "")

# Configure system proxy to access Hugging Face (disabled - proxy not available)
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
# os.environ["http_proxy"] = "http://127.0.0.1:7897"
# os.environ["https_proxy"] = "http://127.0.0.1:7897"
# os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

# Set environment variables first to avoid J drive missing error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "F:/hf_cache/hub"
os.environ["PADDLE_HOME"] = "F:/paddle_cache"
os.environ["HF_HUB_CACHE"] = "F:/hf_cache/hub"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Leave default for 4-bit loading
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Use default HF endpoint to avoid FileMetadataError
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["DOWNLOAD_SOURCE"] = "modelscope"

# Force huggingface_hub constants override if already imported
try:
    import huggingface_hub.constants
    huggingface_hub.constants.HF_HOME = "F:/hf_cache"
    huggingface_hub.constants.HF_HUB_CACHE = "F:/hf_cache/hub"
except Exception:
    pass

import argparse
import json
import time
from pathlib import Path
from PIL import Image
import Levenshtein  # Requires python-Levenshtein

def apply_paddle_patches():
    # Apply Paddle compatibility patches for older PaddlePaddle versions
    try:
        import sys
        from types import ModuleType
        import paddle
        # Handle missing flex_checkpoint in Paddle 2.6.x (paddleformers needs it)
        try:
            import paddle.distributed.flex_checkpoint.dcp.sharded_weight
        except (ImportError, ModuleNotFoundError, AttributeError):
            dummy = ModuleType('dummy')
            dummy.build_sharded_state_dict = lambda *a, **kw: None
            sys.modules.setdefault('paddle.distributed.flex_checkpoint', dummy)
            sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', dummy)
            sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', dummy)
        # NOTE: Do NOT alias float8 to float32 — PaddlePaddle 2.6.2 natively
        # supports float8_e4m3fn/float8_e5m2.  Aliasing them breaks paddleformers'
        # internal paddle_numpy_mapping (float32 tensors get routed through fp8
        # numpy conversion, producing NaN).
        paddle.LongTensor = paddle.Tensor
        paddle.linalg.fp8_fp8_half_gemm_fused = None
        paddle.Tensor.long = lambda self: self.astype("int64")
        paddle.Tensor.float = lambda self: self.astype("float32")
        paddle.Tensor.half = lambda self: self.astype("float16")
        
        # Patch paddle.Tensor.reshape and view to support PyTorch-style positional arguments
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
            
        # Patch paddle.Tensor.transpose to support PyTorch-style transpose(dim0, dim1)
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

        # Patch paddle.Tensor.masked_scatter
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

        # Patch get_flags and set_flags to bypass non-existent FLAGS_flash_attn_version
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
        # Patch gelu to handle string approximate arguments (like "none" or "tanh")
        old_gelu = paddle.nn.functional.gelu
        def patched_gelu(x, approximate=False, name=None):
            if isinstance(approximate, str):
                approximate = (approximate == 'tanh')
            return old_gelu(x, approximate, name)
        paddle.nn.functional.gelu = patched_gelu
        
        # Patch tensor creation functions to ignore PyTorch's 'device' argument
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
            # q, k, v are shape [b, l, h, d]
            q_tr = q.transpose([0, 2, 1, 3])
            k_tr = k.transpose([0, 2, 1, 3])
            v_tr = v.transpose([0, 2, 1, 3])

            b, h_q, l_q, d = q_tr.shape
            _, h_k, l_k, _ = k_tr.shape

            # Handle Grouped Query Attention (GQA) by repeating Key/Value heads
            if h_q != h_k:
                n_rep = h_q // h_k
                k_tr = k_tr.reshape([b, h_k, 1, l_k, d])
                k_tr = paddle.tile(k_tr, [1, 1, n_rep, 1, 1])
                k_tr = k_tr.reshape([b, h_q, l_k, d])

                v_tr = v_tr.reshape([b, h_k, 1, l_k, d])
                v_tr = paddle.tile(v_tr, [1, 1, n_rep, 1, 1])
                v_tr = v_tr.reshape([b, h_q, l_k, d])

            # Build explicit attention mask ONLY when needed:
            # - startend_row_indices present (flashmask)
            # - KV-cache decode where l_q != l_k (Paddle 2.6.2 is_causal bug)
            # In prefill (l_q == l_k), delegate to SDPA's built-in is_causal for speed.
            attn_mask = None
            use_causal = False

            if startend_row_indices is not None:
                if startend_row_indices.shape[-1] == 1:
                    startend_row_indices = startend_row_indices.squeeze(-1)

                if startend_row_indices.ndim == 3:
                    # Build flashmask: -inf everywhere except [start, end) ranges
                    se = startend_row_indices  # [b, l_q, 2|4]
                    mask = paddle.full([b, 1, l_q, l_k], -1e9, dtype=q.dtype)
                    if se.shape[-1] == 2:
                        starts, ends = se[..., 0], se[..., 1]  # [b, l_q]
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
                # KV-cache decode: Paddle 2.6.2 is_causal mishandles l_q != l_k.
                # Build correct causal mask: q_pos i attends to k_pos j iff
                # j <= l_k - l_q + i (i.e., all cached keys before the query).
                # Vectorized via broadcasting — fast even for large l_k.
                row_idx = paddle.arange(l_q, dtype='int32').reshape([1, 1, l_q, 1])
                col_idx = paddle.arange(l_k, dtype='int32').reshape([1, 1, 1, l_k])
                causal_bool = col_idx <= (l_k - l_q + row_idx)
                attn_mask = paddle.where(
                    causal_bool,
                    paddle.zeros([1, 1, l_q, l_k], dtype=q.dtype),
                    paddle.full([1, 1, l_q, l_k], -1e9, dtype=q.dtype)
                )
                # Broadcast to batch size
                if b > 1:
                    attn_mask = paddle.tile(attn_mask, [b, 1, 1, 1])
            else:
                # Prefill: l_q == l_k, SDPA built-in causal works correctly
                use_causal = causal

            try:
                out_tr = paddle.nn.functional.scaled_dot_product_attention(
                    q_tr, k_tr, v_tr,
                    attn_mask=attn_mask,
                    is_causal=use_causal,
                    training=False
                )
            except Exception:
                # Manual scaled dot product attention fallback
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

        # Patch safetensors PySafeSlice to support .shape attribute for compatibility
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
        except Exception as patch_e:
            print(f"Warning: Failed to patch safetensors PySafeSlice: {patch_e}", file=sys.stderr)
    except Exception as e:
        import sys
        print(f"Warning: Failed to apply Paddle compatibility patches: {e}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(description="Circuit VLM Benchmark Evaluation Script")
    parser.add_argument("--model_type", type=str, required=True, choices=["paddleocr-vl", "paddleocr-vl-lora", "qwen3-vl", "qwen3-vl-lora"],
                        help="Model framework type: paddleocr-vl, paddleocr-vl-lora, qwen3-vl, qwen3-vl-lora")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model name or path")
    parser.add_argument("--lora_path", type=str, default=None, help="PEFT LoRA path (for qwen3-vl-lora)")
    parser.add_argument("--paddle_lora_dir", type=str, default=None, help="PaddleOCR-VL LoRA weights directory (for paddleocr-vl-lora)")
    parser.add_argument("--data_path", type=str, required=True, help="Test dataset path (.jsonl)")
    parser.add_argument("--output_path", type=str, default="benchmark_results.jsonl", help="Output file to save answers")
    parser.add_argument("--max_length", type=int, default=1024, help="Max generation length")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of processed samples (for dry run)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from existing output file; skip already-processed samples")
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


def _manual_greedy_decode(model, inputs, processor, max_new_tokens=512, eos_token_id=2):
    """Bypass model.generate() which segfaults on Windows Paddle 2.6.2.

    Does greedy decoding token-by-token using model.forward() directly.
    Each forward call is independent (use_cache=False), avoiding KV-cache
    memory allocation bugs in Paddle's C++ generate().
    """
    import paddle
    current_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    # Forward kwargs shared across all steps
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
        # PaddleFormers forward returns (logits,) tuple or CausalLMOutput
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        next_token_id = int(paddle.argmax(logits[0, -1, :]).item())

        if next_token_id == eos_token_id:
            break

        generated_ids.append(next_token_id)
        # Append new token
        current_ids = paddle.concat(
            [current_ids, paddle.to_tensor([[next_token_id]], dtype=current_ids.dtype)],
            axis=1
        )

    # Decode only generated tokens
    full_ids = paddle.concat(
        [inputs["input_ids"][:, :input_len],
         paddle.to_tensor([generated_ids], dtype=inputs["input_ids"].dtype)],
        axis=1
    ) if generated_ids else inputs["input_ids"][:, :input_len]
    return processor.decode(full_ids[0][input_len:], skip_special_tokens=True)


# ==================== PaddleOCR-VL Loader and Predictor ====================
def evaluate_paddleocr_vl(args):
    print("Applying Paddle compatibility patches...")
    apply_paddle_patches()
    
    print("Loading PaddleOCR-VL libraries...")
    import paddle
    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
    from paddleformers.generation import GenerationConfig

    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    print(f"Setting Paddle device to: {device}")
    paddle.set_device(device)
    
    # Use local cache path to avoid HF network requests (GFW blocks them)
    model_path = args.model_name_or_path
    local_processor_path = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
    processor = AutoProcessor.from_pretrained(local_processor_path)
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path,
        convert_from_hf=True,
        load_checkpoint_format='naive',
        low_cpu_mem_usage=True,
        dtype="float16"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    model.eval()

    # Load PaddleOCR-VL LoRA weights if specified
    # Strategy: apply LoRA deltas directly to base model params via numpy.
    # LoRAModel wrapper + generate() crashes on Windows Paddle 2.6.2.
    if args.paddle_lora_dir:
        print(f"Loading PaddleOCR-VL LoRA from: {args.paddle_lora_dir}")
        import numpy as np
        LORA_SCALE = 2.0  # alpha/r = 16/8

        # Load LoRA weights (float32 version)
        lora_file = f"{args.paddle_lora_dir}/lora_final_fp16.pdparams"
        if not Path(lora_file).exists():
            lora_file = f"{args.paddle_lora_dir}/lora_weights_f32.pdparams"
        if not Path(lora_file).exists():
            lora_file = f"{args.paddle_lora_dir}/final_model_light.pdparams"
        print(f"  Source: {lora_file}")

        # Load LoRA state on CPU, extract adapter pairs
        lora_state = paddle.load(lora_file)
        lora_pairs = {}
        for k, v in lora_state.items():
            if k.endswith('.lora_A'):
                base_name = k[:-len('.lora_A')]
                # Strip 'model.' prefix to match model's internal key naming
                clean_base = base_name[6:] if base_name.startswith('model.') else base_name
                lora_pairs.setdefault(clean_base, {})['A'] = v.numpy()
                lora_pairs[clean_base]['_orig_key'] = k
            elif k.endswith('.lora_B'):
                base_name = k[:-len('.lora_B')]
                clean_base = base_name[6:] if base_name.startswith('model.') else base_name
                lora_pairs.setdefault(clean_base, {})['B'] = v.numpy()
        print(f"  Found {len(lora_pairs)} LoRA adapter pairs")

        # Build param map from named_parameters (iterator, no 3.6GB copy)
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
            lora_B = adapters['B']  # shape: (r, H) where H = heads_total * hidden_dim
            # Reshape lora_B from (r, heads*hidden) to (r*heads_used, hidden_out) based on weight shape
            weight_key = f"{lora_base}.weight"
            if weight_key not in base_params:
                skipped_no_match += 1
                continue
            p = base_params[weight_key]
            W = p.numpy()
            hidden_in, hidden_out = W.shape  # e.g. (1152, 1152) or (1152, 2304)

            if lora_A.shape[-1] != lora_B.shape[0]:
                skipped_shape += 1
                continue
            # Paddle Linear: W = [in_features, out_features], A=[in,r], B=[r,out]
            # delta = A@B = [in, out] — matches W directly
            delta = lora_A @ lora_B * LORA_SCALE
            if delta.shape == W.shape:
                W_new = W + delta.astype('float32')
            elif delta.shape[0] == W.shape[1] and delta.shape[1] == W.shape[0]:
                # Transposed: (A@B).T would match
                W_new = W + delta.T.astype('float32')
            elif delta.shape[0] == W.shape[0] and delta.shape[1] > W.shape[1]:
                # GQA: delta covers all heads, W only KV heads — truncate
                W_new = W + delta[:, :W.shape[1]].astype('float32')
            elif delta.shape[0] < W.shape[0] and W.shape[0] % delta.shape[0] == 0:
                # Need to tile rows
                rep = W.shape[0] // delta.shape[0]
                W_new = W + np.tile(delta.astype('float32'), (rep, 1))
            elif delta.shape[0] == W.shape[0] and delta.shape[1] < W.shape[1] and W.shape[1] % delta.shape[1] == 0:
                # Need to tile columns
                rep = W.shape[1] // delta.shape[1]
                W_new = W + np.tile(delta.astype('float32'), (1, rep))
            else:
                skipped_shape += 1
                if skipped_shape <= 5:
                    print(f"  SKIP shape: {weight_key} delta={delta.shape} vs W={W.shape}")
                continue
            # Match param dtype to avoid set_value dtype conversion bug
            param_dtype = p.dtype  # float16 or bfloat16
            p.set_value(paddle.to_tensor(W_new.astype('float16'), dtype=param_dtype, place=p.place))
            merged += 1

        model.eval()
        print(f"  Merged {merged}/{len(lora_pairs)} LoRA adapters into base model params (no_match={skipped_no_match}, shape={skipped_shape})")
        total = sum(p.size for p in model.parameters())
        print(f"  Total params: {total:,}")

    generation_config = GenerationConfig(
        do_sample=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True  # Try True: old runs got 100+ samples with this
    )

    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]

    # Resume: skip samples already present in existing output file
    already_processed = set()
    if args.resume and Path(args.output_path).exists():
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if "images" in d:
                        already_processed.add(tuple(d["images"]))
        print(f"Resuming: found {len(already_processed)} already-processed samples in {args.output_path}")
    samples_to_run = [s for s in samples if tuple(s["images"]) not in already_processed]
    print(f"Loaded {len(samples)} test samples, {len(samples_to_run)} to process. Running inference...")
    results = []
    import gc

    for i, sample in enumerate(samples_to_run):
        orig_idx = i + len(already_processed)
        start = time.time()
        query = sample["messages"][0]["content"]
        image_path = sample["images"][0]
        # Resolve path
        img_resolved_path = Path(image_path)
        if not img_resolved_path.exists():
            img_resolved_path = Path(args.data_path).parent / img_resolved_path.name

        image = None
        try:
            image = Image.open(img_resolved_path).convert("RGB")
            # Resize large images to avoid Paddle GPU crash on 8GB VRAM
            w, h = image.size
            max_dim = 768
            if w > max_dim or h > max_dim:
                scale = max_dim / max(w, h)
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            # Re-encode as JPEG to bypass Paddle 2.6.2 PNG C++ stack overflow bug
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format='JPEG', quality=95)
            buf.seek(0)
            image = Image.open(buf)
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
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd"
            )

            # Try model.generate() first, fall back to manual decode
            try:
                with paddle.no_grad():
                    outputs = model.generate(**inputs, generation_config=generation_config, max_new_tokens=args.max_length)
                    # Handle tuple output (ids_tensor, ...) or direct tensor
                    if isinstance(outputs, (list, tuple)):
                        tok = outputs[0]
                    else:
                        tok = outputs
                    # tok shape: [batch_size, seq_len]
                    output_ids = tok[0].tolist() if hasattr(tok, 'tolist') else tok[0].numpy().tolist()
                    output_ids = [int(x) for x in output_ids if int(x) > 0]
                    output_text = processor.decode(output_ids, skip_special_tokens=True)
            except Exception as gen_err:
                print(f"  generate() failed: {gen_err}, falling back to manual decode", flush=True)
                with paddle.no_grad():
                    output_text = _manual_greedy_decode(
                        model, inputs, processor,
                        max_new_tokens=args.max_length,
                        eos_token_id=2
                    )

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
            # Cleanup
            if image is not None:
                image.close()
            # Aggressive cleanup to prevent Paddle 2.6.2 memory leak
            import gc
            gc.collect()
            paddle.device.cuda.empty_cache()
            paddle.device.cuda.synchronize()
            sys.stdout.flush()

    return results

# ==================== Qwen3-VL Loader and Predictor ====================
def evaluate_qwen3_vl(args):
    print("Loading PyTorch & Transformers...")
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"Loading Qwen3 model from: {args.model_name_or_path}")

    # Load model with 4-bit quantization to fit 8GB VRAM
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto",
    )

    if args.model_type == "qwen3-vl-lora" and args.lora_path:
        print(f"Applying LoRA adapter from: {args.lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model.eval()

    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]

    # Resume: skip samples already present in existing output file
    already_processed = set()
    if args.resume and Path(args.output_path).exists():
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if "images" in d:
                        already_processed.add(tuple(d["images"]))
        print(f"Resuming: found {len(already_processed)} already-processed samples in {args.output_path}")
    samples_to_run = [s for s in samples if tuple(s["images"]) not in already_processed]
    print(f"Loaded {len(samples)} test samples, {len(samples_to_run)} to process. Running inference...")
    results = []
    import gc

    for i, sample in enumerate(samples_to_run):
        orig_idx = i + len(already_processed)
        start = time.time()
        query = sample["messages"][0]["content"]
        image_path = sample["images"][0]
        # Resolve path
        img_resolved_path = Path(image_path)
        if not img_resolved_path.exists():
            img_resolved_path = Path(args.data_path).parent / img_resolved_path.name

        try:
            # Resize large images to avoid CUDA OOM (max 1024x1024)
            from PIL import Image
            pil_img = Image.open(img_resolved_path).convert("RGB")
            w, h = pil_img.size
            max_dim = 1024
            if w > max_dim or h > max_dim:
                scale = max_dim / max(w, h)
                pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                # Save temp resized image
                import tempfile
                tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
                os.close(tmp_fd)
                pil_img.save(tmp_path, 'PNG')
                img_path_to_use = tmp_path
            else:
                img_path_to_use = str(img_resolved_path.absolute())

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path_to_use},
                        {"type": "text", "text": "Transcribe all text labels exactly as they appear in this circuit schematic. Output only the text, nothing else."},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # Clean up temp file
            if 'tmp_path' in dir():
                try: os.unlink(tmp_path)
                except: pass

            with torch.no_grad():
                max_tokens = min(args.max_length, 150)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

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
            sys.stdout.flush()

    return results

# ==================== Main Benchmark Logic ====================
def main():
    args = parse_args()
    start_time = time.time()

    # Clear output file at start (unless resuming)
    output_file = Path(args.output_path)
    if output_file.exists() and not args.resume:
        output_file.unlink()

    if args.model_type in ("paddleocr-vl", "paddleocr-vl-lora"):
        results = evaluate_paddleocr_vl(args)
    else:
        results = evaluate_qwen3_vl(args)

    # Compute metrics from all results in the output file (includes resumed data)
    output_file = Path(args.output_path)
    all_results = []
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    predictions = [res["prediction"] for res in all_results]
    references = [res["label"] for res in all_results]
    avg_ned = compute_metrics(predictions, references)

    # Metrics report (output file already written incrementally)
    print("\n" + "="*40)
    print("        Evaluation Report")
    print("="*40)
    print(f"Model Type: {args.model_type}")
    print(f"Model:      {args.model_name_or_path}")
    if args.lora_path:
        print(f"LoRA Path:  {args.lora_path}")
    if args.paddle_lora_dir:
        print(f"Paddle LoRA: {args.paddle_lora_dir}")
    print(f"Samples:    {len(all_results)}")
    print(f"Avg. NED:   {avg_ned:.4f} (Lower is better)")
    print("="*40)
    print(f"Results saved to: {output_file.absolute()}")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
