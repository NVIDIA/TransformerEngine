# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

from typing import Union, Dict, Any

import torch

from .. import cpp_extensions as tex
from ..fp8 import get_fp8_te_dtype

def _get_normalization_func(normalization: str,
                            fp8_output: bool,
                            is_grad_enabled: bool,
                            forward: bool):
    fwd_normalization_funcs = {
            ('LayerNorm', True, True):   tex.layernorm_fwd_fp8,
            ('LayerNorm', True, False):  tex.layernorm_fwd_fp8_inf,
            ('LayerNorm', False, True):  tex.layernorm_fwd_noalloc,
            ('LayerNorm', False, False): tex.layernorm_fwd_inf,
            ('RMSNorm', True, True):     tex.rmsnorm_fwd_fp8,
            ('RMSNorm', True, False):    tex.rmsnorm_fwd_fp8_inf,
            ('RMSNorm', False, True):    tex.rmsnorm_fwd_noalloc,
            ('RMSNorm', False, False):   tex.rmsnorm_fwd_inf,
    }
    bwd_normalization_funcs = {
            ('LayerNorm', True):  tex.layernorm_bwd_fp8,
            ('LayerNorm', False):  tex.layernorm_bwd,
            ('RMSNorm', True): tex.rmsnorm_bwd_fp8,
            ('RMSNorm', False): tex.rmsnorm_bwd,
    }

    if forward:
        return fwd_normalization_funcs[(normalization, fp8_output, is_grad_enabled)]
    assert not fp8_output, "FP8 output is not supported in backward normalization!"
    assert is_grad_enabled, "Gradient has to be enabled to call backward normalization!"
    return bwd_normalization_funcs[(normalization, fp8_output)]

def _apply_normalization(inputmat:torch.Tensor,
                         ln_out: torch.Tensor,
                         ln_weight: torch.Tensor,
                         ln_bias: Union[torch.Tensor, None],
                         eps: float,
                         fp8_out: bool,
                         fp8_meta: Dict[str, Any],
                         normalization: str,
                         fwd_ln_sm_margin: int,
                         zero_centered_gamma: bool,
                         is_grad_enabled: bool):
    normalization_func = _get_normalization_func(normalization,
                                                 fp8_out,
                                                 is_grad_enabled,
                                                 True)

    inputs = (inputmat, ln_weight) if ln_bias is None else (inputmat, ln_weight, ln_bias)
    if fp8_out:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        if is_grad_enabled:
            output_key = "ln_out" if normalization == "LayerNorm" else "rmsnorm_out"
            output_kwarg = {output_key: ln_out}
            output = normalization_func(
                *inputs,
                eps,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                fwd_ln_sm_margin,
                zero_centered_gamma,
                **output_kwarg,
            )
        else:
            return normalization_func(
                *inputs,
                eps,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                zero_centered_gamma,
            ), None
    else:
        if is_grad_enabled:
            output = normalization_func(
                *inputs, ln_out, eps,
                fwd_ln_sm_margin, zero_centered_gamma
            )
        else:
            return normalization_func(
                    *inputs, eps, zero_centered_gamma
            ), None
    return (ln_out, output[1])


def _get_normalization_grad(d_ln_out: torch.Tensor,
                            fake_input: torch.Tensor,
                            ln_out: torch.Tensor,
                            ln_weight: torch.Tensor,
                            ln_bias: Union[torch.Tensor, None],
                            rsigma: torch.Tensor,
                            eps: float,
                            fwd_scale_inverse: torch.Tensor,
                            fp8_out: bool,
                            normalization: str,
                            bwd_ln_sm_margin: int,
                            zero_centered_gamma: bool):
    normalization_func = _get_normalization_func(normalization,
                                                 fp8_out,
                                                 True,
                                                 False)

    inputs = [d_ln_out, ln_out, fake_input, rsigma, ln_weight]
    if ln_bias is not None:
        inputs.append(ln_bias)

    if fp8_out:
        output = normalization_func(
            *inputs,
            eps,
            fwd_scale_inverse,
            bwd_ln_sm_margin,
            zero_centered_gamma,
        )
    else:
        output = normalization_func(
            *inputs,
            eps,
            bwd_ln_sm_margin,
            zero_centered_gamma,
        )

    return output[0], output[1], None if ln_bias is None else output[2]
