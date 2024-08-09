# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for normalization extensions"""
from typing import Optional, Tuple, Union

import torch

import transformer_engine_torch as tex
from ._common import canonicalize_fp8_scales


__all__ = [
    "layernorm_fwd_fp8",
    "layernorm_fwd_fp8_inf",
    "layernorm_fwd_inf",
    "rmsnorm_fwd_fp8",
    "rmsnorm_fwd_fp8_inf",
    "rmsnorm_fwd_inf",
]


def layernorm_fwd_fp8(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool,
    ln_out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    if ln_out is not None:
        return tex.layernorm_fwd_fp8_noalloc(
            inp,
            weight,
            bias,
            eps,
            fp8_scales["scale"],
            ln_out,
            fp8_scales["amax"],
            fp8_scales["scale_inv"],
            otype,
            sm_margin,
            zero_centered_gamma,
            **fp8_scales_offsets,
        )
    return tex.layernorm_fwd_fp8(
        inp,
        weight,
        bias,
        eps,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        sm_margin,
        zero_centered_gamma,
        **fp8_scales_offsets,
    )


def layernorm_fwd_fp8_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """LayerNorm with FP8 output.

    This version of layernorm_fwd_fp8 is specialized for inference, and returns
    only the normalized output.
    """

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    ret = torch.ops.tex_ts.layernorm_fwd_fp8_inf_ts(
        inp,
        weight,
        bias,
        eps,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        fp8_scales_offsets["scale_offset"],
        otype,
        sm_margin,
        zero_centered_gamma,
    )
    return ret


def layernorm_fwd_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    """LayerNorm with FP8 output"""
    return torch.ops.tex_ts.layernorm_fwd_inf_ts(
        inp,
        weight,
        bias,
        eps,
        sm_margin,
        zero_centered_gamma,
    )


def rmsnorm_fwd_fp8(
    inp: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool,
    rmsnorm_out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    if rmsnorm_out is not None:
        return tex.rmsnorm_fwd_fp8_noalloc(
            inp,
            weight,
            eps,
            fp8_scales["scale"],
            rmsnorm_out,
            fp8_scales["amax"],
            fp8_scales["scale_inv"],
            otype,
            sm_margin,
            zero_centered_gamma,
            **fp8_scales_offsets,
        )
    return tex.rmsnorm_fwd_fp8(
        inp,
        weight,
        eps,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        sm_margin,
        zero_centered_gamma,
        **fp8_scales_offsets,
    )


def rmsnorm_fwd_fp8_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMSNorm with FP8 output.

    This version of rmsnorm_fwd_fp8 is specialized for inference, and returns
    only the normalized output.
    """

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    ret = torch.ops.tex_ts.rmsnorm_fwd_fp8_inf_ts(
        inp,
        weight,
        eps,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        fp8_scales_offsets["scale_offset"],
        otype,
        sm_margin,
        zero_centered_gamma,
    )
    return ret


def rmsnorm_fwd_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    """RMSNorm with FP8 output"""
    return torch.ops.tex_ts.rmsnorm_fwd_inf_ts(
        inp,
        weight,
        eps,
        sm_margin,
        zero_centered_gamma,
    )
