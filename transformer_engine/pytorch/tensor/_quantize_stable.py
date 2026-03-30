# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Stable ABI quantize implementation for TE quantizer classes.

Replaces tex.quantize(tensor, quantizer, output, noop) calls with
direct calls to stable ABI ops, eliminating the pybind11 dependency.
"""

import torch

from transformer_engine.pytorch.tensor._extract import extract_tensor_data

# Load stable ops
_ops = None


def _get_ops():
    global _ops
    if _ops is not None:
        return _ops
    import glob
    import importlib.util
    from pathlib import Path

    te_spec = importlib.util.find_spec("transformer_engine")
    if te_spec is not None and te_spec.origin is not None:
        te_dir = Path(te_spec.origin).parent.parent
        candidates = glob.glob(str(te_dir / "te_stable_abi*"))
        if candidates:
            torch.ops.load_library(candidates[0])
    _ops = torch.ops.transformer_engine_stable
    return _ops


def quantize_into(src, quantizer, dst, noop_flag=None):
    """Quantize src into pre-allocated dst using stable ABI ops.

    Replaces: tex.quantize(src, quantizer, dst, noop_flag)
    """
    ops = _get_ops()

    # Early return for empty tensors
    if src.numel() == 0:
        return

    # Ensure contiguous input
    if not src.is_contiguous():
        src = src.contiguous()

    # Helper: transpose src to match columnwise layout [K, *M_dims].
    # get_columnwise_shape puts the last dim first, rest in original order.
    def _transpose_for_colwise(t):
        if t.ndim == 2:
            return t.T.contiguous()
        # For ndim >= 3: put last dim first, keep remaining dims in order
        perm = [t.ndim - 1] + list(range(t.ndim - 1))
        return t.permute(*perm).contiguous()

    # Handle columnwise-only Float8BlockwiseQTensor destination.
    # When _rowwise_data=None but _columnwise_data exists, we quantize the
    # transposed input (to match the [K,M] columnwise layout) into _columnwise_data.
    _col_only = (
        hasattr(dst, "_rowwise_data")
        and getattr(dst, "_rowwise_data", None) is None
        and hasattr(dst, "_columnwise_data")
        and getattr(dst, "_columnwise_data", None) is not None
        and not hasattr(dst, "_data")  # exclude Float8Tensor (which uses _data/_transpose)
    )
    if _col_only:
        col_data = dst._columnwise_data
        col_si = getattr(dst, "_columnwise_scale_inv", None)
        fp8_dtype_attr = getattr(dst, "_fp8_dtype", None)
        from transformer_engine.pytorch.tensor._extract import _FP8_DTYPE_TO_TE

        out_dtype = _FP8_DTYPE_TO_TE.get(str(fp8_dtype_attr), 7) if fp8_dtype_attr else 7
        block_dim = getattr(quantizer, "block_scaling_dim", 2)
        out_sm = 3 if block_dim == 2 else 2  # BLOCK_SCALING_2D=3, BLOCK_1D=2
        force_pow_2 = getattr(quantizer, "force_pow_2_scales", False)
        amax_eps = getattr(quantizer, "amax_epsilon", 0.0)
        if block_dim == 2:
            # 2D block scaling: quantize src (original shape) → tmp rowwise buffer,
            # then FP8-transpose into col_data and transpose the scale.
            # Do NOT pass src_transposed to ops.quantize: the kernel computes scale
            # block count from the input tensor shape, and src_transposed has a
            # different shape (e.g. 512×128×1) than what col_si expects (e.g. 1×4).
            rowwise_scale_shape = quantizer.get_scale_shape(list(src.shape), columnwise=False)
            tmp_si = torch.empty(rowwise_scale_shape, dtype=torch.float32, device=src.device)
            tmp_rowwise = col_data.new_empty(list(src.shape))  # uint8, same shape as src
            ops.quantize(
                src,
                tmp_rowwise,
                out_dtype,
                None,
                None,
                tmp_si,
                out_sm,
                force_pow_2,
                amax_eps,
                noop_flag,
            )
            ops.fp8_transpose(tmp_rowwise, out_dtype, col_data)
            if col_si is not None:
                col_si.zero_()
                transposed_si = tmp_si.T.contiguous()
                h = min(col_si.shape[0], transposed_si.shape[0])
                w = min(col_si.shape[1], transposed_si.shape[1])
                col_si[0:h, 0:w].copy_(transposed_si[0:h, 0:w])
        else:
            # 1D block scaling: the kernel must see src_transposed as (K, M) 2D.
            # _transpose_for_colwise gives (K, *M_dims) which the kernel would
            # flatten to (K*M_rest, M_last) — wrong shape for col_si.
            # Reshape to (K=last dim of src, M=all other dims) so the kernel sees
            # (dim0=K, dim1=M) and produces the correct per-row scale shape.
            K = src.shape[-1]
            M = src.numel() // K
            src_transposed_2d = _transpose_for_colwise(src).view(K, M)
            ops.quantize(
                src_transposed_2d,
                col_data,
                out_dtype,
                None,
                None,
                col_si,
                out_sm,
                force_pow_2,
                amax_eps,
                noop_flag,
            )
        dst._fp8_dtype = quantizer.dtype if hasattr(quantizer, "dtype") else dst._fp8_dtype
        return

    # Extract raw output buffers from dst
    out_data, out_dtype, out_scale_inv, out_sm = extract_tensor_data(dst)

    # Override scaling mode from quantizer if available (more reliable than tensor attrs)
    q_type = type(quantizer).__name__
    if "Block" in q_type:
        block_dim = getattr(quantizer, "block_scaling_dim", 2)
        out_sm = 3 if block_dim == 2 else 2  # BLOCK_SCALING_2D=3 or 1D=2
    elif "MXFP8" in q_type:
        out_sm = 1  # MXFP8_1D_SCALING=1
    elif "NVFP4" in q_type:
        out_sm = 4  # NVFP4_1D_SCALING=4
    elif "CurrentScaling" in q_type:
        out_sm = 0  # DELAYED_TENSOR_SCALING (current scaling uses delayed mode internally)

    # Get scale/amax from quantizer
    scale = getattr(quantizer, "scale", None)
    amax = getattr(quantizer, "amax", None)
    if scale is not None and (not isinstance(scale, torch.Tensor) or scale.numel() == 0):
        scale = None
    if amax is not None and (not isinstance(amax, torch.Tensor) or amax.numel() == 0):
        amax = None

    # Also check output for amax
    if amax is None:
        for attr in ("_amax", "_amax_rowwise", "amax_rowwise"):
            a = getattr(dst, attr, None)
            if isinstance(a, torch.Tensor) and a.numel() > 0:
                amax = a
                break

    force_pow_2 = getattr(quantizer, "force_pow_2_scales", False)
    amax_eps = getattr(quantizer, "amax_epsilon", 0.0)
    use_existing_amax = getattr(quantizer, "use_existing_amax", False)
    q_type = type(quantizer).__name__

    # Only pass scale_inv for FP8/FP4 output dtypes. The C++ CheckOutputTensor
    # asserts that scale_inv must NOT be set for non-FP8 outputs.
    is_fp8_or_fp4 = out_dtype in (7, 8, 9, 10)  # kFloat8E4M3, kFloat8E5M2, kFloat8E8M0, kFloat4E2M1
    effective_scale_inv = out_scale_inv if is_fp8_or_fp4 else None

    if use_existing_amax and amax is not None:
        ops.quantize_from_amax(
            src,
            out_data,
            out_dtype,
            amax,
            scale or torch.ones(1, dtype=torch.float32, device=src.device),
            effective_scale_inv,
            out_sm,
            force_pow_2,
            amax_eps,
            noop_flag,
        )
    elif "CurrentScaling" in q_type:
        if amax is None:
            amax = torch.zeros(1, dtype=torch.float32, device=src.device)
        if scale is None:
            scale = torch.zeros(1, dtype=torch.float32, device=src.device)
        ops.quantize_with_amax(
            src,
            out_data,
            out_dtype,
            amax,
            scale,
            effective_scale_inv,
            out_sm,
            force_pow_2,
            amax_eps,
            noop_flag,
        )
    else:
        ops.quantize(
            src,
            out_data,
            out_dtype,
            amax,
            scale,
            effective_scale_inv,
            out_sm,
            force_pow_2,
            amax_eps,
            noop_flag,
        )

    # NVFP4 quantize kernel doesn't write per-tensor amax via the stable path.
    # The NVFP4 dequantize formula is: output = fp4_value * scale_e4m3 * amax / (6 * 448).
    # With amax = 6 * 448 = 2688, this simplifies to: output = fp4_value * scale_e4m3.
    if "NVFP4" in q_type and amax is not None and amax.item() == 0.0:
        amax.fill_(6.0 * 448.0)

    # For Float8Tensor (delayed scaling), _transpose may be pre-allocated by make_empty
    # when columnwise_usage=True, but it is not filled by ops.quantize above (only _data
    # gets filled). Mark _transpose_invalid=True so update_usage(columnwise_usage=True)
    # will call _create_transpose() to fill it from _data on demand.
    if hasattr(dst, "_data") and dst._data is not None and hasattr(dst, "_transpose_invalid"):
        dst._transpose_invalid = True

    # For block-scaling tensors with both rowwise AND columnwise pre-allocated,
    # also fill the columnwise buffer. The pybind path filled both in one fused
    # nvte_quantize_v2 kernel. The stable path fills rowwise above, then derives
    # columnwise by FP8-transposing the quantized bytes and transposing the scales.
    # This matches _create_columnwise() in float8_blockwise_tensor_storage.py.
    _has_colwise = (
        hasattr(dst, "_rowwise_data")
        and getattr(dst, "_rowwise_data", None) is not None
        and hasattr(dst, "_columnwise_data")
        and getattr(dst, "_columnwise_data", None) is not None
        and not hasattr(dst, "_data")  # exclude Float8Tensor (uses _transpose/_create_transpose)
    )
    if _has_colwise and ("Block" in q_type or "NVFP4" in q_type):
        col_data = dst._columnwise_data
        col_si = getattr(dst, "_columnwise_scale_inv", None)
        fp8_dtype_attr = getattr(dst, "_fp8_dtype", None)
        from transformer_engine.pytorch.tensor._extract import _FP8_DTYPE_TO_TE

        col_dtype = (
            _FP8_DTYPE_TO_TE.get(str(fp8_dtype_attr), out_dtype) if fp8_dtype_attr else out_dtype
        )
        if "NVFP4" in q_type:
            # NVFP4 columnwise: derive from rowwise data by transposing the
            # already-quantized FP4 bytes and scales. This matches the pybind
            # path (_create_columnwise in nvfp4_tensor_storage.py) which uses
            # nvfp4_data_transpose + nvfp4_2d_scale_transpose.
            # nvfp4_data_transpose expects 2D [M, K_bytes]; flatten leading dims
            rd = out_data.reshape(-1, out_data.shape[-1]) if out_data.ndim > 2 else out_data
            ops.nvfp4_data_transpose(rd, col_data)
            if col_si is not None and out_scale_inv is not None:
                logical_shape = list(src.shape)
                M_val = 1
                for d in logical_shape[:-1]:
                    M_val *= d
                K_val = logical_shape[-1]
                TILE_SIZE = 16
                M_tiles = (M_val + TILE_SIZE - 1) // TILE_SIZE
                K_tiles = (K_val + TILE_SIZE - 1) // TILE_SIZE
                ops.nvfp4_2d_scale_transpose(out_scale_inv, col_si, M_tiles, K_tiles)
            # Copy rowwise amax to columnwise amax (matches _create_columnwise
            # in nvfp4_tensor_storage.py:445-447). cuBLAS NVFP4 GEMM uses amax
            # in the formula: out = fp4 * scale * amax / (6*448).
            amax_rw = getattr(dst, "_amax_rowwise", None)
            amax_cw = getattr(dst, "_amax_columnwise", None)
            if amax_rw is not None and amax_cw is not None:
                amax_cw.copy_(amax_rw)
            elif amax_rw is not None and amax_cw is None:
                dst._amax_columnwise = amax_rw.clone()
        else:
            block_dim = getattr(quantizer, "block_scaling_dim", 2)
            if block_dim == 2:
                # 2D block scaling: columnwise scale = transposed rowwise scale.
                # FP8-transpose the quantized bytes (identical to _create_columnwise)
                ops.fp8_transpose(out_data, col_dtype, col_data)
                # Transpose the rowwise scale_inv into the columnwise scale_inv buffer
                if col_si is not None and out_scale_inv is not None:
                    col_si.zero_()
                    transposed_si = out_scale_inv.T.contiguous()
                    h = min(col_si.shape[0], transposed_si.shape[0])
                    w = min(col_si.shape[1], transposed_si.shape[1])
                    col_si[0:h, 0:w].copy_(transposed_si[0:h, 0:w])
            else:
                # 1D block scaling: columnwise scale ≠ transposed rowwise scale (they cover
                # different block directions). Quantize src in "columnwise mode" by reshaping
                # the transposed src to (K, M) and calling ops.quantize in ROWWISE mode.
                # This gives per-K-block scales == the per-M-block columnwise scales we need.
                if col_si is not None and src.ndim >= 1:
                    K = src.shape[-1]
                    M = src.numel() // K
                    src_transposed_2d = _transpose_for_colwise(src).view(K, M)
                    ops.quantize(
                        src_transposed_2d,
                        col_data,
                        col_dtype,
                        None,
                        None,
                        col_si,
                        out_sm,
                        force_pow_2,
                        amax_eps,
                        noop_flag,
                    )


def quantize_new(tensor, quantizer):
    """Allocate output and quantize tensor using stable ABI ops.

    Replaces: return tex.quantize(tensor, quantizer)
    """
    # Ensure contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # MXFP8 requires dimensions divisible by block size (32). The pybind fused
    # C++ kernel handles non-aligned sizes internally by padding. In the stable
    # path we pad the input, quantize, then slice back to the original shape.
    _MXFP8_BLOCK = 32
    padded = False
    orig_shape = list(tensor.shape)
    q_type = type(quantizer).__name__
    if "MXFP8" in q_type and len(orig_shape) >= 2:
        last_dim = orig_shape[-1]
        first_dims_prod = 1
        for d in orig_shape[:-1]:
            first_dims_prod *= d
        need_pad_last = last_dim % _MXFP8_BLOCK != 0
        need_pad_first = first_dims_prod % _MXFP8_BLOCK != 0
        if need_pad_last or need_pad_first:
            pad_last = (_MXFP8_BLOCK - last_dim % _MXFP8_BLOCK) % _MXFP8_BLOCK
            # Flatten to 2D for padding, then reshape back
            flat = tensor.reshape(first_dims_prod, last_dim)
            pad_first = (_MXFP8_BLOCK - first_dims_prod % _MXFP8_BLOCK) % _MXFP8_BLOCK
            if pad_last > 0 or pad_first > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_last, 0, pad_first))
                tensor = flat  # keep as 2D for quantize
                padded = True

    # Allocate output via quantizer's make_empty (pure Python)
    dst = quantizer.make_empty(list(tensor.shape), dtype=tensor.dtype, device=tensor.device)

    # Quantize into the new output
    quantize_into(tensor, quantizer, dst)

    # If we padded, slice the quantized output back to the original shape
    if padded:
        first_dims_prod = 1
        for d in orig_shape[:-1]:
            first_dims_prod *= d
        # Slice back to original 2D shape, then restore original dims
        if hasattr(dst, "_rowwise_data") and dst._rowwise_data is not None:
            dst._rowwise_data = dst._rowwise_data[:first_dims_prod, : orig_shape[-1]]
        if hasattr(dst, "_rowwise_scale_inv") and dst._rowwise_scale_inv is not None:
            si = dst._rowwise_scale_inv
            # Scale has ceil(M/32) rows and ceil(K/32) cols
            orig_si_rows = (first_dims_prod + _MXFP8_BLOCK - 1) // _MXFP8_BLOCK
            orig_si_cols = (orig_shape[-1] + _MXFP8_BLOCK - 1) // _MXFP8_BLOCK
            if si.ndim == 2:
                dst._rowwise_scale_inv = si[:orig_si_rows, :orig_si_cols]

    return dst
