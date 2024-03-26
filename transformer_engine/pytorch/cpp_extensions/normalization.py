# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for normalization extensions"""
from typing import Optional, Tuple, Union
import torch
import transformer_engine_extensions as tex


__all__ = ['layernorm_fwd_fp8',
           'layernorm_fwd_fp8_inf',
           'layernorm_fwd_inf',
           'rmsnorm_fwd_fp8',
           'rmsnorm_fwd_fp8_inf',
           'rmsnorm_fwd_inf',
           'get_norm_workspace_and_barrier',
           'set_norm_workspace_and_barrier']


_norm_scratch_spaces = {}


def get_norm_key(inp: torch.Tensor, weight: torch.Tensor, fp8: bool) -> str:
    """Get unique key for workspace/barrier config."""
    return f"{inp.shape}_{inp.dtype}_{weight.shape}_{weight.dtype}_{fp8}"


def get_norm_workspace_and_barrier(
    inp: torch.Tensor,
    weight: torch.Tensor,
    fp8: bool,
) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    """Get workspace and barrier for config."""
    key = get_norm_key(inp, weight, fp8)
    return key, _norm_scratch_spaces.get(key, (None, None))


def set_norm_workspace_and_barrier(
    key: str,
    workspace: torch.Tensor,
    barrier: torch.Tensor,
) -> None:
    """Set workspace and barrier for config."""
    if key not in _norm_scratch_spaces:
        _norm_scratch_spaces[key] = (workspace, barrier)


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm with FP8 output"""
    conf, (workspace, barrier) = get_norm_workspace_and_barrier(inp, weight, True)
    if ln_out is not None:
        out, mu, rsigma, workspace, barrier = tex.layernorm_fwd_fp8_noalloc(
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
            workspace,
            barrier,
        )
    else:
        out, mu, rsigma, workspace, barrier = tex.layernorm_fwd_fp8(
            inp,
            weight,
            bias,
            eps,
            fp8_meta_tensor.scale[fp8_tensor],
            fp8_meta_tensor.amax_history[0][fp8_tensor],
            fp8_meta_tensor.scale_inv[fp8_tensor],
            otype,
            sm_margin,
            zero_centered_gamma,
            workspace,
            barrier,
        )
    set_norm_workspace_and_barrier(conf, workspace, barrier)
    return out, mu, rsigma


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


def rmsnorm_fwd_fp8(
    inp: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool,
    rmsnorm_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm with FP8 output"""
    conf, (workspace, barrier) = get_norm_workspace_and_barrier(inp, weight, True)
    if rmsnorm_out is not None:
        out, rsigma, workspace, barrier = tex.rmsnorm_fwd_fp8_noalloc(
            inp,
            weight,
            eps,
            fp8_meta_tensor.scale[fp8_tensor],
            rmsnorm_out,
            fp8_meta_tensor.amax_history[0][fp8_tensor],
            fp8_meta_tensor.scale_inv[fp8_tensor],
            otype,
            sm_margin,
            zero_centered_gamma
        )
    else:
        out, rsigma, workspace, barrier = tex.rmsnorm_fwd_fp8(
            inp,
            weight,
            eps,
            fp8_meta_tensor.scale[fp8_tensor],
            fp8_meta_tensor.amax_history[0][fp8_tensor],
            fp8_meta_tensor.scale_inv[fp8_tensor],
            otype,
            sm_margin,
            zero_centered_gamma
        )
    set_norm_workspace_and_barrier(conf, workspace, barrier)
    return out, rsigma


def rmsnorm_fwd_fp8_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    zero_centered_gamma,
) -> torch.Tensor:
    """RMSNorm with FP8 output.

    This version of rmsnorm_fwd_fp8 is specialized for inference, and returns
    only the normalized output.
    """
    ret = torch.ops.tex_ts.rmsnorm_fwd_fp8_inf_ts(
        inp,
        weight,
        eps,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
        zero_centered_gamma)
    return ret


def rmsnorm_fwd_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    """RMSNorm with FP8 output"""
    return torch.ops.tex_ts.rmsnorm_fwd_inf_ts(
        inp,
        weight,
        eps,
        zero_centered_gamma,
    )
