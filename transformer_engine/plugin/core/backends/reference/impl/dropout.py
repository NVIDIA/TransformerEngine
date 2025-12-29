# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    "dropout_fwd_torch",
    "dropout_bwd_torch",
]


def dropout_fwd_torch(
    input: torch.Tensor,
    dropout_probability: float,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dropout_probability == 0.0:
        output = input.clone() if out is None else input.clone().to(out)
        mask = torch.ones_like(input, dtype=torch.uint8)
        return output, mask

    mask = torch.bernoulli(
        torch.full_like(input, 1.0 - dropout_probability)
    ).to(torch.uint8)

    scale = 1.0 / (1.0 - dropout_probability)
    output = input * mask.to(input.dtype) * scale

    if out is not None:
        out.copy_(output)
        output = out

    return output, mask


def dropout_bwd_torch(
    grad_output: torch.Tensor,
    mask: torch.Tensor,
    dropout_probability: float,
    grad_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dropout_probability == 0.0:
        return grad_output.clone() if grad_input is None else grad_output.clone().to(grad_input)

    scale = 1.0 / (1.0 - dropout_probability)
    grad = grad_output * mask.to(grad_output.dtype) * scale

    if grad_input is not None:
        grad_input.copy_(grad)
        grad = grad_input

    return grad
