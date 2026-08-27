#!/usr/bin/env python3
"""
PaddleOCR-VL LoRA Training Launcher
====================================
Applies Paddle 2.6.2 compatibility patches, then invokes paddleformers CLI.

All patches are identical to those in eval_benchmark.py, proven working
on RTX 4060 + Paddle 2.6.2 + PaddleFormers 1.1.1.
"""

import os
import sys
import json

# ============================================================
# Pre-import patches: must run BEFORE paddleformers is imported
# ============================================================

def apply_paddle_patches():
    """Apply Paddle compatibility patches for Paddle 2.6.2."""
    from types import ModuleType
    import paddle

    # ---- Import hook: auto-stub any missing paddle.distributed.* submodule ----
    import importlib.abc, importlib.machinery, importlib.util

    class _PaddleStubLoader(importlib.abc.Loader):
        """Loader that creates stub modules for missing paddle.distributed.* imports."""
        def create_module(self, spec):
            return ModuleType(spec.name)
        def exec_module(self, module):
            pass

    class _PaddleStubFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if not fullname.startswith('paddle.distributed.'):
                return None
            if fullname in sys.modules:
                return None
            # Try real import (temporarily remove self to avoid recursion)
            idx = sys.meta_path.index(self)
            sys.meta_path.pop(idx)
            try:
                spec = importlib.util.find_spec(fullname)
                sys.meta_path.insert(idx, self)
                if spec is not None:
                    return spec
            except Exception:
                sys.meta_path.insert(idx, self)
            # Doesn't exist — stub it
            sys.meta_path.insert(idx, self)
            return importlib.machinery.ModuleSpec(fullname, _PaddleStubLoader(), is_package=True)

    sys.meta_path.insert(0, _PaddleStubFinder())

    # Handle missing flex_checkpoint in Paddle 2.6.x
    try:
        import paddle.distributed.flex_checkpoint.dcp.sharded_weight
    except (ImportError, ModuleNotFoundError, AttributeError):
        dummy = ModuleType('dummy')
        dummy.build_sharded_state_dict = lambda *a, **kw: None
        dummy.create_sharded_weight_with_new_local = lambda *a, **kw: None
        dummy.reshape_sharded_weight = lambda *a, **kw: None
        dummy.sharded_weight_parallel_cpu = lambda *a, **kw: None
        dummy.save_state_dict = lambda *a, **kw: None
        dummy.load_state_dict = lambda *a, **kw: None
        sys.modules.setdefault('paddle.distributed.flex_checkpoint', dummy)
        sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', dummy)
        sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', dummy)

    # Missing dtypes in Paddle 2.6.2
    paddle.float8_e4m3fn = paddle.float32
    paddle.float8_e5m2 = paddle.float32
    paddle.LongTensor = paddle.Tensor
    paddle.linalg.fp8_fp8_half_gemm_fused = None

    # Tensor dtype cast methods
    paddle.Tensor.long = lambda self: self.astype("int64")
    paddle.Tensor.float = lambda self: self.astype("float32")
    paddle.Tensor.half = lambda self: self.astype("float16")

    # reshape/view patch
    old_reshape = paddle.Tensor.reshape
    def patched_view(self, *args, **kwargs):
        if args and isinstance(args[0], paddle.dtype):
            return old_reshape(self, *args, **kwargs)
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

    # transpose patch
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

    # masked_scatter patch
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

    # get_flags / set_flags patches
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
    paddle.set_flags = patched_set_flags

    # gelu patch
    old_gelu = paddle.nn.functional.gelu
    def patched_gelu(x, approximate=False, name=None):
        if isinstance(approximate, str):
            approximate = (approximate == 'tanh')
        return old_gelu(x, approximate, name)
    paddle.nn.functional.gelu = patched_gelu

    # tensor creation patches
    def patch_creation_func(func_name):
        old_func = getattr(paddle, func_name)
        def patched(*args, **kwargs):
            kwargs.pop('device', None)
            return old_func(*args, **kwargs)
        setattr(paddle, func_name, patched)
    for name in ['empty', 'zeros', 'ones', 'arange', 'full', 'randn', 'rand']:
        if hasattr(paddle, name):
            patch_creation_func(name)

    # swiglu stub
    import paddle.nn.functional as pnf
    pnf.swiglu = lambda *args, **kwargs: None

    # fused_rms_norm_ext fallback
    def fallback_fused_rms_norm_ext(x, weight, epsilon=1e-6):
        variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
        rsqrt = paddle.rsqrt(variance + epsilon)
        normalized = x * rsqrt * weight
        return (normalized, rsqrt)
    import paddle.incubate.nn.functional as pinf
    pinf.fused_rms_norm_ext = fallback_fused_rms_norm_ext

    # flashmask_attention fallback
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
                q_tr, k_tr, v_tr,
                attn_mask=attn_mask,
                is_causal=use_causal,
                training=False
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

    import paddle.nn.functional.flash_attention as fa
    fa.flashmask_attention = fallback_flashmask_attention

    # create_async_load stub
    import paddle.incubate.tensor.manipulation as m
    m.create_async_load = lambda *args, **kwargs: None

    # SharedLayerDesc alias
    import paddle.distributed.fleet.meta_parallel as mp
    if hasattr(mp, 'SharedLayerDesc'):
        mp.LocalSharedLayerDesc = mp.SharedLayerDesc

    # PySafeSlice.shape patch
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
    except Exception as e:
        print(f"Warning: Failed to patch safetensors PySafeSlice: {e}", file=sys.stderr)

    print("[Launcher] All Paddle compatibility patches applied successfully.")


def main():
    # 1. Apply patches
    print("[Launcher] Applying Paddle 2.6.2 compatibility patches...")
    apply_paddle_patches()

    # 2. Set environment
    os.environ.setdefault("HF_HOME", "F:/hf_cache/hub")
    os.environ.setdefault("PADDLE_HOME", "F:/paddle_cache")
    os.environ.setdefault("HF_HUB_CACHE", "F:/hf_cache/hub")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # 3. Change to project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)
    print(f"[Launcher] Working directory: {os.getcwd()}")

    # 4. Verify GPU
    import paddle
    paddle.set_device('gpu')
    print(f"[Launcher] GPU: {paddle.device.cuda.get_device_name(0)}")
    print(f"[Launcher] VRAM: {paddle.device.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"[Launcher] Paddle: {paddle.__version__}")

    # 5. Import paddleformers CLI and invoke train
    print("[Launcher] Importing paddleformers...")
    from paddleformers.cli.cli import main as cli_main

    # Build CLI args
    config_file = "configs/paddleocr-vl_lora_8gb.yaml"
    sys.argv = [
        "paddleformers-cli", "train",
        config_file,
        "model_name_or_path=PaddlePaddle/PaddleOCR-VL",
        "train_dataset_path=./ocr_vl_sft-train.jsonl",
        "eval_dataset_path=./ocr_vl_sft-test.jsonl",
        "pre_alloc_memory=6.0",
    ]

    print(f"[Launcher] Starting training with config: {config_file}")
    print(f"[Launcher] Time: {__import__('datetime').datetime.now()}")
    print("=" * 60)

    try:
        cli_main()
    except SystemExit as e:
        print(f"\n[Launcher] Training finished with exit code: {e.code}")
        return e.code
    except Exception as e:
        print(f"\n[Launcher] Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
