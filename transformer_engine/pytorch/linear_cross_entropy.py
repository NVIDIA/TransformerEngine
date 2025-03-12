# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Linear Cross Entropy API
Fuse cross entropy with linear layer.
"""

import typing
import torch
from transformer_engine.pytorch.triton import linear_cross_entropy_with_token_entropy as linear_cross_entropy_with_token_entropy_kernels
from transformer_engine.pytorch.triton import linear_cross_entropy as linear_cross_entropy_kernels


class LinearCrossEntropyWithTokenEntropy(torch.autograd.Function):
    """
    This class implements a custom autograd function for linear (matmul) and cross entropy, as well as entropy per token,
    whose equivalent logic in PyTorch is:
        ```python
        def torch_entropy(hidden, weight, labels):
            logits = torch.matmul(hidden, weight)
            pd = torch.nn.functional.softmax(logits, dim=-1)
            entropy_a = torch.logsumexp(logits, dim=-1)
            entropy_b = torch.sum(pd * logits, dim=-1)
            entropy = entropy_a - entropy_b
            logprobs = torch.nn.functional.cross_entropy(logits, labels)
            return logprobs, entropy
        ```
    """
    @staticmethod
    def forward(ctx,
                hidden: torch.Tensor,
                weight: torch.Tensor,
                labels: torch.Tensor,
                reduction: typing.Optional[str] = "mean") -> typing.List[torch.Tensor]:
        """
        The forward pass of the Linear Cross Entropy with Token Entropy.
        Args:
            hidden (torch.Tensor): The input tensor of shape (num_tokens, hidden_size).
            weight (torch.Tensor): The weight tensor of shape (hidden_size, vocab_size).
            labels (torch.Tensor): The labels tensor of shape (num_tokens,).
            reduction (str, optional): The reduction method. Defaults to "mean", and can be
                one of "none", "sum", "mean".
        Returns:
            logprobs (torch.Tensor): The cross entropy.
            entropy (torch.Tensor): The entropy of shape (num_tokens,).
        """
        with torch.cuda.nvtx.range("EfficientEntropy-forward"):
            REDUCTION = linear_cross_entropy_with_token_entropy_kernels.get_entropy_reduction_enum_number(reduction.lower())

            logprobs, entropy, _maximum, _maximum_indices, _acc =\
                linear_cross_entropy_with_token_entropy_kernels.efficient_entropy_foward(hidden, weight, labels, REDUCTION)

            ctx.save_for_backward(hidden, weight, labels, _maximum, _maximum_indices, _acc)
            ctx.REDUCTION = REDUCTION

        return logprobs, entropy

    @staticmethod
    def backward(ctx,
                 dlogprobs: torch.Tensor,
                 dentropy: torch.Tensor) -> typing.List[torch.Tensor]:
        """
        The backward pass of the Linear Cross Entropy.
        Args:
            dlogprobs (torch.Tensor): The gradient of the cross entropy.
            dentropy (torch.Tensor): The gradient of the entropy.
        Returns:
            dhidden (torch.Tensor): The gradient of the hidden.
            dweight (torch.Tensor): The gradient of the weight.
        """
        with torch.cuda.nvtx.range("EfficientEntropy-backward"):
            (hidden, weight, labels, _maximum, _maximum_indices, _acc) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION

            d_hidden, d_weight = linear_cross_entropy_with_token_entropy_kernels.efficient_entropy_backward(
                dlogprobs, dentropy,
                hidden, weight, labels,
                _maximum, _maximum_indices, _acc,
                REDUCTION)

        return d_hidden, d_weight, None, None


linear_cross_entropy_with_token_entropy = LinearCrossEntropyWithTokenEntropy.apply


class LinearCrossEntropy(torch.autograd.Function):
    """
    This class implements a custom autograd function for linear and cross entropy, whose equivalent logic in PyTorch is:
        ```python
        def torch_entropy(hidden, weight, labels):
            logits = torch.matmul(hidden, weight)
            logprobs = torch.nn.functional.cross_entropy(logits, labels)
            return logprobs
        ```
    """
    @staticmethod
    def forward(ctx,
                hidden: torch.Tensor,
                weight: torch.Tensor,
                labels: torch.Tensor,
                reduction: typing.Optional[str] = "mean",
                dist_process_group: typing.Optional[torch.distributed.ProcessGroup] = None) -> torch.Tensor:
        """
        The forward pass of the Linear Cross Entropy.
        If dist_process_group is passed for distributed loss calculation,
        the weight tensor to each distributed rank should be (*, vocab_size / world_size).
        Note that each of the ranks should get equal shards along the vocab_size dimension.

        Args:
            hidden (torch.Tensor): The input tensor of shape (num_tokens, hidden_size).
            weight (torch.Tensor): The weight tensor of shape (hidden_size, vocab_size).
            labels (torch.Tensor): The labels tensor of shape (num_tokens,).
            reduction (str, optional): The reduction method. Defaults to "mean", and can be
                one of "none", "sum", "mean".
        Returns:
            logprobs (torch.Tensor): The cross entropy.
        """
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            REDUCTION = linear_cross_entropy_with_token_entropy_kernels.get_entropy_reduction_enum_number(reduction.lower())

            logprobs, _maximum, _maximum_indices, _acc =\
                linear_cross_entropy_kernels.efficient_entropy_forward(
                    hidden, weight, labels, REDUCTION,
                    dist_process_group)

            ctx.save_for_backward(hidden, weight, labels, _maximum, _maximum_indices, _acc)
            ctx.REDUCTION = REDUCTION

        return logprobs

    @staticmethod
    def backward(ctx,
                 dlogprobs: torch.Tensor) -> typing.List[torch.Tensor]:
        """
        The backward pass of the Linear Cross Entropy.
        Args:
            dlogprobs (torch.Tensor): The gradient of the cross entropy.
        Returns:
            dhidden (torch.Tensor): The gradient of the hidden.
            dweight (torch.Tensor): The gradient of the weight.
        """
        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (hidden, weight, labels, _maximum, _maximum_indices, _acc) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION

            d_hidden, d_weight = linear_cross_entropy_kernels.efficient_entropy_backward(
                dlogprobs, 
                hidden, weight, labels,
                _maximum, _maximum_indices, _acc,
                REDUCTION
            )
        return d_hidden, d_weight, None, None, None
    

linear_cross_entropy = LinearCrossEntropy.apply

__all__ = ["linear_cross_entropy_with_token_entropy", 
           "LinearCrossEntropyWithTokenEntropy",
           "linear_cross_entropy",
           "LinearCrossEntropy"]
