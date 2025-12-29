# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch

__all__ = [
    "rmsnorm_fwd_torch",
    "rmsnorm_bwd_torch",
]


def rmsnorm_fwd_torch(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    odtype,
    sm_margin,
    zero_centered_gamma,
):
    if weight.device != input.device:
        weight = weight.to(input.device)

    variance = input.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    y = input * inv_rms
    if zero_centered_gamma:
        y = y * (1 + weight)
    else:
        y = y * weight

    rstdevs = inv_rms.squeeze(-1)

    return y, None, rstdevs


def rmsnorm_bwd_torch(
    dy,
    x,
    rsigma,
    gamma,
    sm_margin,
    zero_centered_gamma,
    eps,
):
    inv_rms = rsigma.unsqueeze(-1)

    x_norm = x * inv_rms

    if zero_centered_gamma:
        weight = 1 + gamma
    else:
        weight = gamma

    dw = (dy * x_norm).sum(dim=tuple(range(dy.ndim - 1)))

    dy_weighted = dy * weight

    mean_term = (dy_weighted * x_norm).mean(-1, keepdim=True)
    dx = inv_rms * (dy_weighted - x_norm * mean_term)
    return dx, dw
