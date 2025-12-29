# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    "layernorm_fwd_torch",
    "layernorm_bwd_torch",
]


def layernorm_fwd_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    ln_out: Optional[torch.Tensor],
    quantizer: Any,
    odtype: torch.dtype,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    rsigma = torch.rsqrt(var + eps)

    normalized = (input - mean) * rsigma

    if zero_centered_gamma:
        output = normalized * (1.0 + weight)
    else:
        output = normalized * weight

    if bias is not None:
        output = output + bias

    if output.dtype != odtype:
        output = output.to(odtype)

    mean = mean.squeeze(-1)
    rsigma = rsigma.squeeze(-1)

    return output, mean, rsigma


def layernorm_bwd_torch(
    dy: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mu.ndim < x.ndim:
        mu = mu.unsqueeze(-1)
    if rsigma.ndim < x.ndim:
        rsigma = rsigma.unsqueeze(-1)

    x_normalized = (x - mu) * rsigma

    N = x.shape[-1]

    if zero_centered_gamma:
        gamma_adj = 1.0 + gamma
    else:
        gamma_adj = gamma

    dy_gamma = dy * gamma_adj

    mean_dy_gamma = dy_gamma.mean(dim=-1, keepdim=True)

    mean_dy_gamma_x = (dy_gamma * x_normalized).mean(dim=-1, keepdim=True)

    dx = rsigma * (dy_gamma - mean_dy_gamma - x_normalized * mean_dy_gamma_x)

    dgamma = (dy * x_normalized).sum(dim=tuple(range(dy.ndim - 1)))

    dbeta = dy.sum(dim=tuple(range(dy.ndim - 1)))

    return dx, dgamma, dbeta
