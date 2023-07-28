# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for normalization extensions"""
from typing import Optional, Tuple, Union
import torch
import transformer_engine_extensions as tex


__all__ = ['layernorm_fwd_fp8',
                       'layernorm_fwd_fp8_inf',
                       'layernorm_fwd_inf',
                       'layernorm_bwd']


def layernorm_fwd_fp8(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool,
    ln_out: Optional[torch.Tensor] = None,
    mu_out: Optional[torch.Tensor] = None,
    rsigma_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm with FP8 output"""
    if None not in [ln_out, mu_out, rsigma_out]:
        tex.layernorm_fwd_fp8_noalloc_ex(
            inp,
            weight,
            bias,
            eps,
            fp8_meta_tensor.scale[fp8_tensor],
            ln_out,
            fp8_meta_tensor.amax_history[0][fp8_tensor],
            fp8_meta_tensor.scale_inv[fp8_tensor],
            otype,
            sm_margin,
            zero_centered_gamma,
            mu_out,
            rsigma_out
        )
        return (ln_out, mu_out, rsigma_out)

    if ln_out is not None:
        return tex.layernorm_fwd_fp8_noalloc(
            inp,
            weight,
            bias,
            eps,
            fp8_meta_tensor.scale[fp8_tensor],
            ln_out,
            fp8_meta_tensor.amax_history[0][fp8_tensor],
            fp8_meta_tensor.scale_inv[fp8_tensor],
            otype,
            sm_margin,
            zero_centered_gamma
        )

    return tex.layernorm_fwd_fp8(
        inp,
        weight,
        bias,
        eps,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
        sm_margin,
        zero_centered_gamma
    )


def layernorm_fwd_fp8_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    zero_centered_gamma,
) -> torch.Tensor:
    """LayerNorm with FP8 output.

    This version of layernorm_fwd_fp8 is specialized for inference, and returns
    only the normalized output.
    """
    ret = torch.ops.tex_ts.layernorm_fwd_fp8_inf_ts(
        inp,
        weight,
        bias,
        eps,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
        zero_centered_gamma)
    return ret


def layernorm_fwd_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    """LayerNorm with FP8 output"""
    return torch.ops.tex_ts.layernorm_fwd_inf_ts(
        inp,
        weight,
        bias,
        eps,
        zero_centered_gamma,
    )


def layernorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
    sm_margin: int,
    zero_centered_gamma: bool,
    dgrad_out: Optional[torch.Tensor] = None,
    wgrad_out: Optional[torch.Tensor] = None,
    bgrad_out: Optional[torch.Tensor] = None,
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm Backward"""
    if None not in [dgrad_out, wgrad_out, bgrad_out]:
        tex.layernorm_bwd_noalloc_ex(
            dz,
            x,
            mu,
            rsigma,
            weight,
            sm_margin,
            zero_centered_gamma,
            dgrad_out,
            wgrad_out,
            bgrad_out
        )
        return (dgrad_out, wgrad_out, bgrad_out)

    return tex.layernorm_bwd(
        dz,
        x,
        mu,
        rsigma,
        weight,
        sm_margin,
        zero_centered_gamma,
    )
