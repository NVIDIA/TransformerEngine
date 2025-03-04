# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import typing
from transformer_engine.pytorch.triton import linear_cross_entropy as linear_cross_entropy_kernels

class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                hidden: torch.Tensor,
                weight: torch.Tensor,
                labels: torch.Tensor,
                reduction: typing.Optional[str] = "mean") -> typing.List[torch.Tensor]:
        with torch.cuda.nvtx.range("EfficientEntropy-forward"):
            REDUCTION = linear_cross_entropy_kernels.get_entropy_reduction_enum_number(reduction.lower())

            logprobs, entropy, _maximum, _maximum_indices, _acc =\
                linear_cross_entropy_kernels.efficient_entropy_foward(hidden, weight, labels, REDUCTION)

            ctx.save_for_backward(hidden, weight, labels, _maximum, _maximum_indices, _acc)
            ctx.REDUCTION = REDUCTION

        return logprobs, entropy

    @staticmethod
    def backward(ctx,
                 dlogprobs: torch.Tensor,
                 dentropy: torch.Tensor) -> typing.List[torch.Tensor]:
        with torch.cuda.nvtx.range("EfficientEntropy-backward"):
            (hidden, weight, labels, _maximum, _maximum_indices, _acc) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION

            d_hidden, d_weight = linear_cross_entropy_kernels.efficient_entropy_backward(
                dlogprobs, dentropy,
                hidden, weight, labels,
                _maximum, _maximum_indices, _acc,
                REDUCTION)

        return (d_hidden, d_weight, None, None)


linear_cross_entropy = LinearCrossEntropy.apply

__all__ = ["linear_cross_entropy", "LinearCrossEntropy"]