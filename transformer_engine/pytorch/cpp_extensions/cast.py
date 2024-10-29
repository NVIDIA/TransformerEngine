# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for cast extensions"""
from typing import Optional, Union

import torch

import transformer_engine_torch as tex
from ._common import canonicalize_fp8_scales

__all__ = ["cast_to_fp8", "cast_from_fp8"]


def cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cast input to FP8"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch FP8 cast kernel
    if inp.nelement() == 0:
        if out is None:
            out = torch.empty_like(inp, dtype=torch.uint8)
    elif out is None:
        out = torch.ops.tex_ts.cast_to_fp8_ts(
            inp,
            fp8_scales["scale"],
            fp8_scales["amax"],
            fp8_scales["scale_inv"],
            fp8_scales_offsets["scale_offset"],
            otype,
        )
    else:
        torch.ops.tex_ts.cast_to_fp8_noalloc_ts(
            inp,
            fp8_scales["scale"],
            out,
            fp8_scales["amax"],
            fp8_scales["scale_inv"],
            fp8_scales_offsets["scale_offset"],
            otype,
        )
    return out


def cast_from_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    itype: tex.DType,
    otype: tex.DType,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cast input from FP8"""

    # Get scaling factors from FP8 meta tensors if needed
    scale_inv_offset = 0
    if (fp8_meta_tensor is not None) and (scale_inv is None):
        if fp8_tensor is None:
            raise ValueError("Provided `fp8_meta_tensor` without corresponding `fp8_tensor`")
        scale_inv = fp8_meta_tensor.scale_inv
        scale_inv_offset = int(fp8_tensor)

    # Construct empty tensors if needed
    if scale_inv is None:
        raise ValueError("Did not provide either `scale_inv` or `fp8_meta_tensor`")

    # Launch FP8 cast kernel
    return torch.ops.tex_ts.cast_from_fp8_ts(
        inp,
        scale_inv,
        scale_inv_offset,
        itype,
        otype,
    )
