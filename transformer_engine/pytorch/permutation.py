# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import os
import torch
import warnings
from typing import Tuple

import transformer_engine_torch as tex


__all__ = [
    "Permute",
    "Unpermute",
]


class _Permute(torch.autograd.Function):
    """functional Permute"""

    workspace = None
    dtype = None
    max_expanded_token_num = 0

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int,
        max_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Empty input check
        if not inp.numel():
            return inp, None

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert indices.is_cuda, "TransformerEngine needs CUDA."
        # Shape check
        assert inp.size(0) == indices.size(0), "Permute not possible"

        # Data type check
        if indices.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `indices` of Permute is {indices.dtype}! "
                "The recommended type is torch.int32."
            )
            indices = indices.to(torch.int32)

        num_topK = indices.size(1)

        input_max_expanded_token_num = max(max_token_num, inp.size(0)) * num_topK
        if _Permute.max_expanded_token_num < input_max_expanded_token_num:
            _Permute.max_expanded_token_num = input_max_expanded_token_num
            _Permute.workspace = []

        if _Permute.dtype != inp.dtype:
            _Permute.dtype = inp.dtype
            _Permute.workspace = []

        permuted_act, row_id_map, _Permute.workspace = tex.moe_permute(
            inp, indices, num_out_tokens, _Permute.workspace, _Permute.max_expanded_token_num
        )

        ctx.row_id_map = row_id_map
        ctx.num_tokens = indices.size(0)
        ctx.num_topK = indices.size(1)
        return permuted_act, row_id_map

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
    ) -> Tuple[torch.Tensor, ...]:
        # Empty input check
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, None

        if not permuted_act_grad.is_contiguous():
            permuted_act_grad = permuted_act_grad.contiguous()

        row_id_map = ctx.row_id_map
        num_tokens = ctx.num_tokens
        num_topK = ctx.num_topK

        unpermuted_act_grad = tex.moe_unpermute_fwd(
            permuted_act_grad, row_id_map, torch.empty(0), num_tokens, num_topK
        )
        return unpermuted_act_grad, None, None, None


class _Unpermute(torch.autograd.Function):
    """functional Unpermute"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        probs: torch.Tensor = torch.empty(0),
    ) -> torch.Tensor:
        # Empty input check
        if not inp.numel():
            ctx.probs = probs
            return inp

        # None probs check
        if probs.numel():
            assert probs.is_cuda, "TransformerEngine needs CUDA."

            if probs.dtype != torch.float32:
                warnings.warn(
                    f"The data type of the input `probs` of Unpermute is {probs.dtype}! "
                    "The recommended type is torch.float32."
                )
                probs = probs.to(torch.float32)

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."

        # Data type check
        if row_id_map.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `row_id_map` of Unpermute is {row_id_map.dtype}! "
                "The recommended type is torch.int32."
            )
            row_id_map = row_id_map.to(torch.int32)

        num_topK = probs.size(1) if probs.numel() else 1
        num_tokens = probs.size(0) if probs.numel() else row_id_map.size(0)

        unpermuted_output = tex.moe_unpermute_fwd(inp, row_id_map, probs, num_tokens, num_topK)

        ctx.save_for_backward(inp, row_id_map, probs)
        return unpermuted_output

    @staticmethod
    def backward(
        ctx,
        unpermuted_act_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        # Empty input check
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.probs

        if not unpermuted_act_grad.is_contiguous():
            unpermuted_act_grad = unpermuted_act_grad.contiguous()

        inp, row_id_map, probs = ctx.saved_tensors

        act_grad = None
        if ctx.needs_input_grad[0]:
            act_grad, prob_grad = tex.moe_unpermute_bwd(unpermuted_act_grad, inp, row_id_map, probs)

        if not ctx.needs_input_grad[2]:
            prob_grad = None
        return act_grad, None, prob_grad


def Permute(
    inp: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int = -1,
    max_token_num: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens based on the indices. Token with the same index will be grouped together.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    indices: torch.Tensor
        The token to expert indices tensor of shape [num_tokens, topK] and dtype 'int32'.
    num_out_tokens: int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    max_token_num: int, default = -1
        The maximum number of tokens, used for workspace allocation.
        By default, set to '-1', meaning the calculation of the size of workspace is
        automatically taken over by the operator.
    """
    return _Permute.apply(inp, indices, num_out_tokens, max_token_num)


def Unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor = torch.empty(0),
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    row_id_map: torch.Tensor
        The tensor of a mapping table for sorted indices used to unpermute the tokens,
        which is the second output tensor of `Permute`.
    probs: torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    """
    return _Unpermute.apply(inp, row_id_map, probs)
