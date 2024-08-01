# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for transpose extensions"""
from typing import Dict, Optional, Tuple, Union

import torch

import transformer_engine_torch as tex
from ..constants import TE_DType


__all__ = [
    "fp8_cast_transpose_fused",
    "fp8_cast_transpose_bgrad_fused",
    "fp8_cast_transpose_bgrad_dgelu_fused",
    "fp8_transpose_bgrad_fused",
]


def _canonicalize_fp8_scales(
    *,
    scale: Optional[torch.Tensor],
    amax: Optional[torch.Tensor],
    scale_inv: Optional[torch.Tensor],
    fp8_meta: Optional[tex.FP8TensorMeta],
    fp8_meta_index: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Canonicalize FP8 scaling factors (scale, amax, scale-inverse)

    If a scaling factor is not provided, try to access it within the
    FP8 meta tensors. Returns dict with tensors and dict with tensor
    offsets.

    """

    # Default: use provided scales with no offsets
    scale_offset = 0
    amax_offset = 0
    scale_inv_offset = 0

    # Get scales from FP8 meta tensors if needed
    if (fp8_meta is not None) and any(arg is None for arg in (scale, amax, scale_inv)):
        if fp8_meta_index is None:
            raise ValueError("Provided `fp8_meta` without corresponding `fp8_meta_index`")
        fp8_meta_index = int(fp8_meta_index)
        if scale is None:
            scale = fp8_meta.scale
            scale_offset = fp8_meta_index
        if amax is None:
            amax = fp8_meta.amax_history
            amax_offset = fp8_meta_index
        if scale_inv is None:
            scale_inv = fp8_meta.scale_inv
            scale_inv_offset = fp8_meta_index

    # Construct empty tensors if needed
    if scale is None:
        scale = torch.Tensor()
        scale_offset = 0
    if amax is None:
        amax = torch.Tensor()
        amax_offset = 0
    if scale_inv is None:
        scale_inv = torch.Tensor()
        scale_inv_offset = 0

    # Pack tensors and offsets into dicts
    tensors = dict(scale=scale, amax=amax, scale_inv=scale_inv)
    offsets = dict(
        scale_offset=scale_offset,
        amax_offset=amax_offset,
        scale_inv_offset=scale_inv_offset,
    )
    return tensors, offsets


def fp8_cast_transpose_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    cast_out: Optional[torch.Tensor] = None,
    transpose_out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    noop_flag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast + Transpose with FP8 output"""

    # Allocate outputs if needed
    if transpose_out is None:
        transpose_out = torch.empty(inp.shape[1], inp.shape[0], device="cuda", dtype=torch.uint8)
    if cast_out is None:
        cast_out = torch.empty_like(inp, dtype=torch.uint8)

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = _canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Construct no-op flag if needed
    if noop_flag is None:
        noop_flag = torch.Tensor()

    # Launch kernel if needed
    if inp.nelement() > 0:
        tex.fused_cast_transpose_noop(
            inp,
            noop_flag,
            fp8_scales["scale"],
            fp8_scales["amax"],
            fp8_scales["scale_inv"],
            cast_out,
            transpose_out,
            otype,
            **fp8_scales_offsets,
        )

    return cast_out, transpose_out


def fp8_cast_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = _canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    return tex.fused_cast_transpose_bgrad(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        **fp8_scales_offsets,
    )


def fp8_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    grad_bias_type: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose + BGRAD with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = _canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    return tex.fused_fp8_transpose_bgrad(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        TE_DType[grad_bias_type],
        **fp8_scales_offsets,
    )


def fp8_cast_transpose_bgrad_dgelu_fused(
    grad_output: torch.Tensor,
    gelu_input: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD + DGELU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = _canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    return tex.fused_cast_transpose_bgrad_dgelu(
        grad_output,
        gelu_input,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        **fp8_scales_offsets,
    )
