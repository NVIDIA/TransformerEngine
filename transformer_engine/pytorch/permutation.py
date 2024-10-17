# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import warnings
from typing import Tuple
import torch

import transformer_engine_torch as tex
from .constants import TE_DType
from .float8_tensor import Float8Tensor


__all__ = [
    "moe_permute",
    "moe_unpermute",
]


class _moe_permute(torch.autograd.Function):
    """functional Permute"""

    workspace = None
    max_expanded_token_num = 0

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int,
        max_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        # Empty input check
        if not inp.numel():
            return inp, torch.tensor([], device=inp.device)

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert indices.is_cuda, "TransformerEngine needs CUDA."
        # Shape check
        assert inp.size(0) == indices.size(0), "Permute not possible"

        # Data type check
        fp8 = isinstance(inp, Float8Tensor)
        if fp8:
            dtype = inp._fp8_dtype
            fp8_scale_inv = inp._scale_inv
            inp = inp._data
        else:
            dtype = TE_DType[inp.dtype]
        if indices.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `indices` of Permute is {indices.dtype}! "
                "The recommended type is torch.int32."
            )
            indices = indices.to(torch.int32)

        topK = indices.size(1)

        input_max_expanded_token_num = max(max_token_num, inp.size(0)) * topK
        if _moe_permute.max_expanded_token_num < input_max_expanded_token_num:
            _moe_permute.max_expanded_token_num = input_max_expanded_token_num
            _moe_permute.workspace = []

        permuted_act, row_id_map, _moe_permute.workspace = tex.moe_permute_fwd(
            inp,
            dtype,
            indices,
            num_out_tokens,
            _moe_permute.workspace,
            _moe_permute.max_expanded_token_num,
        )

        if fp8:
            permuted_act = Float8Tensor(
                data=permuted_act, fp8_dtype=dtype, fp8_scale_inv=fp8_scale_inv
            )

        ctx.row_id_map = row_id_map
        ctx.num_tokens = indices.size(0)
        ctx.topK = indices.size(1)
        ctx.fp8 = fp8
        return permuted_act, row_id_map

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        # Empty input check
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, None

        if not permuted_act_grad.is_contiguous():
            permuted_act_grad = permuted_act_grad.contiguous()

        if ctx.fp8:
            assert isinstance(
                permuted_act_grad, Float8Tensor
            ), "Grad of the output must be in Float8Tensor type for FP8 moe_permute."
            dtype = permuted_act_grad._fp8_dtype
            fp8_scale_inv = permuted_act_grad._scale_inv
            permuted_act_grad = permuted_act_grad._data
        else:
            dtype = TE_DType[permuted_act_grad.dtype]

        act_grad = None
        if ctx.needs_input_grad[0]:
            act_grad = tex.moe_permute_bwd(
                permuted_act_grad, dtype, ctx.row_id_map, torch.empty(0), ctx.num_tokens, ctx.topK
            )
            if ctx.fp8:
                act_grad = Float8Tensor(
                    data=act_grad, fp8_dtype=dtype, fp8_scale_inv=fp8_scale_inv * ctx.topK
                )

        return act_grad, None, None, None


class _moe_unpermute(torch.autograd.Function):
    """functional Unpermute"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        # Empty input check
        if not inp.numel():
            ctx.probs = probs
            return inp

        # None probs check
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

            if probs.dtype != torch.float32:
                warnings.warn(
                    f"The data type of the input `probs` of Unpermute is {probs.dtype}! "
                    "The recommended type is torch.float32."
                )
                probs = probs.to(torch.float32)

            num_tokens = probs.size(0)
            topK = probs.size(1)
        else:
            num_tokens = row_id_map.size(0)
            topK = 1
            probs = torch.empty(0)

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."

        # Data type check
        fp8 = isinstance(inp, Float8Tensor)
        if fp8:
            dtype = inp._fp8_dtype
            fp8_scale_inv = inp._scale_inv
            inp = inp._data
        else:
            dtype = TE_DType[inp.dtype]
        if row_id_map.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `row_id_map` of Unpermute is {row_id_map.dtype}! "
                "The recommended type is torch.int32."
            )
            row_id_map = row_id_map.to(torch.int32)

        unpermuted_output = tex.moe_unpermute_fwd(inp, dtype, row_id_map, probs, num_tokens, topK)

        if fp8:
            unpermuted_output = Float8Tensor(
                data=unpermuted_output, fp8_dtype=dtype, fp8_scale_inv=fp8_scale_inv
            )

        ctx.save_for_backward(inp, row_id_map, probs)
        ctx.fp8 = fp8
        return unpermuted_output

    @staticmethod
    def backward(
        ctx,
        unpermuted_act_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        # Empty input check
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.probs

        if not unpermuted_act_grad.is_contiguous():
            unpermuted_act_grad = unpermuted_act_grad.contiguous()

        if ctx.fp8:
            assert isinstance(
                unpermuted_act_grad, Float8Tensor
            ), "Grad of the output must be in Float8Tensor type for FP8 moe_unpermute."
            dtype = unpermuted_act_grad._fp8_dtype
            fp8_scale_inv = unpermuted_act_grad._scale_inv
            unpermuted_act_grad = unpermuted_act_grad._data
        else:
            dtype = TE_DType[unpermuted_act_grad.dtype]

        inp, row_id_map, probs = ctx.saved_tensors

        act_grad = None
        prob_grad = None
        if ctx.needs_input_grad[0]:
            act_grad, prob_grad = tex.moe_unpermute_bwd(
                unpermuted_act_grad, inp, dtype, row_id_map, probs
            )
            if ctx.fp8:
                act_grad = Float8Tensor(data=act_grad, fp8_dtype=dtype, fp8_scale_inv=fp8_scale_inv)
        if not ctx.needs_input_grad[2]:
            prob_grad = None

        return act_grad, None, prob_grad


def moe_permute(
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
    return _moe_permute.apply(inp, indices, num_out_tokens, max_token_num)


def moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor = None,
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
    return _moe_unpermute.apply(inp, row_id_map, probs)
