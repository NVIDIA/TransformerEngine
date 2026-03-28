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

    # Extract raw output buffers from dst
    out_data, out_dtype, out_scale_inv, out_sm = extract_tensor_data(dst)

    # Override scaling mode from quantizer if available (more reliable than tensor attrs)
    q_type = type(quantizer).__name__
    if 'Block' in q_type:
        block_dim = getattr(quantizer, 'block_scaling_dim', 2)
        out_sm = 3 if block_dim == 2 else 2  # BLOCK_SCALING_2D=3 or 1D=2
    elif 'MXFP8' in q_type:
        out_sm = 1  # MXFP8_1D_SCALING=1
    elif 'NVFP4' in q_type:
        out_sm = 4  # NVFP4_1D_SCALING=4
    elif 'CurrentScaling' in q_type:
        out_sm = 0  # DELAYED_TENSOR_SCALING (current scaling uses delayed mode internally)

    # Get scale/amax from quantizer
    scale = getattr(quantizer, 'scale', None)
    amax = getattr(quantizer, 'amax', None)
    if scale is not None and (not isinstance(scale, torch.Tensor) or scale.numel() == 0):
        scale = None
    if amax is not None and (not isinstance(amax, torch.Tensor) or amax.numel() == 0):
        amax = None

    # Also check output for amax
    if amax is None:
        for attr in ('_amax', '_amax_rowwise', 'amax_rowwise'):
            a = getattr(dst, attr, None)
            if isinstance(a, torch.Tensor) and a.numel() > 0:
                amax = a
                break

    force_pow_2 = getattr(quantizer, 'force_pow_2_scales', False)
    amax_eps = getattr(quantizer, 'amax_epsilon', 0.0)
    use_existing_amax = getattr(quantizer, 'use_existing_amax', False)
    q_type = type(quantizer).__name__

    if use_existing_amax and amax is not None:
        ops.quantize_from_amax(
            src, out_data, out_dtype, amax,
            scale or torch.ones(1, dtype=torch.float32, device=src.device),
            out_scale_inv, out_sm, force_pow_2, amax_eps, noop_flag)
    elif 'CurrentScaling' in q_type:
        if amax is None:
            amax = torch.zeros(1, dtype=torch.float32, device=src.device)
        if scale is None:
            scale = torch.zeros(1, dtype=torch.float32, device=src.device)
        ops.quantize_with_amax(
            src, out_data, out_dtype, amax, scale,
            out_scale_inv, out_sm, force_pow_2, amax_eps, noop_flag)
    else:
        ops.quantize(
            src, out_data, out_dtype, amax, scale,
            out_scale_inv, out_sm, force_pow_2, amax_eps, noop_flag)


def quantize_new(tensor, quantizer):
    """Allocate output and quantize tensor using stable ABI ops.

    Replaces: return tex.quantize(tensor, quantizer)
    """
    # Ensure contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Allocate output via quantizer's make_empty (pure Python)
    dst = quantizer.make_empty(
        list(tensor.shape), dtype=tensor.dtype, device=tensor.device)

    # Quantize into the new output
    quantize_into(tensor, quantizer, dst)

    return dst
