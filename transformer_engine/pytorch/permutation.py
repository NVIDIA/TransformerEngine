# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE Permutaion API"""
import warnings
from typing import Optional, Tuple
import torch

import transformer_engine_torch as tex
import transformer_engine.pytorch.triton.permutation as triton_permutation
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

__all__ = [
    "moe_permute",
    "moe_unpermute",
    "moe_sort_chunks_by_index",
]


class _moe_permute_index_map(torch.autograd.Function):
    """functional Permute with index router map"""

    workspace = None
    max_expanded_token_num = 0

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        index: torch.Tensor,
        num_out_tokens: int,
        max_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        # Empty input check
        if not inp.numel():
            return inp, torch.tensor([], device=inp.device)

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert index.is_cuda, "TransformerEngine needs CUDA."
        # Shape check
        assert inp.size(0) == index.size(0), "Permute not possible"

        # Data type check
        dtype = TE_DType[inp.dtype]
        if index.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `index` of Permute is {index.dtype}! "
                "The recommended type is torch.int32."
            )
            index = index.to(torch.int32)

        topK = index.size(1)

        input_max_expanded_token_num = max(max_token_num, inp.size(0)) * topK
        if _moe_permute_index_map.max_expanded_token_num < input_max_expanded_token_num:
            _moe_permute_index_map.max_expanded_token_num = input_max_expanded_token_num
            _moe_permute_index_map.workspace = []

        permuted_act, row_id_map, _moe_permute_index_map.workspace = tex.moe_permute_fwd(
            inp,
            dtype,
            index,
            num_out_tokens,
            _moe_permute_index_map.workspace,
            _moe_permute_index_map.max_expanded_token_num,
        )

        ctx.row_id_map = row_id_map
        ctx.num_tokens = index.size(0)
        ctx.topK = index.size(1)
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

        dtype = TE_DType[permuted_act_grad.dtype]
        act_grad = None
        if ctx.needs_input_grad[0]:
            act_grad = tex.moe_permute_bwd(
                permuted_act_grad, dtype, ctx.row_id_map, torch.empty(0), ctx.num_tokens, ctx.topK
            )

        return act_grad, None, None, None


class _moe_unpermute_index_map(torch.autograd.Function):
    """functional Unpermute with index router map"""

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
        dtype = TE_DType[inp.dtype]
        if row_id_map.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `row_id_map` of Unpermute is {row_id_map.dtype}! "
                "The recommended type is torch.int32."
            )
            row_id_map = row_id_map.to(torch.int32)

        unpermuted_output = tex.moe_unpermute_fwd(inp, dtype, row_id_map, probs, num_tokens, topK)

        ctx.save_for_backward(inp, row_id_map, probs)
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

        dtype = TE_DType[unpermuted_act_grad.dtype]
        inp, row_id_map, probs = ctx.saved_tensors

        act_grad = None
        prob_grad = None
        if ctx.needs_input_grad[0]:
            act_grad, prob_grad = tex.moe_unpermute_bwd(
                unpermuted_act_grad, inp, dtype, row_id_map, probs
            )
        if not ctx.needs_input_grad[2]:
            prob_grad = None

        return act_grad, None, prob_grad


class _moe_permute_mask_map(torch.autograd.Function):
    """functional Permute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.probs = probs
            return inp, torch.tensor([], device=inp.device), torch.tensor([], device=inp.device)

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert routing_map.is_cuda, "TransformerEngine needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

        assert inp.size(0) == routing_map.size(0), "Permute not possible"
        num_tokens, hidden_size = inp.size()
        num_experts = routing_map.size(1)
        assert (
            num_out_tokens is not None
        ), "num_out_tokens must be provided to the fused permute function."

        row_id_map = triton_permutation.make_row_id_map(routing_map, num_tokens, num_experts)

        fp8 = isinstance(inp, QuantizedTensor)
        per_tensor_recipe = isinstance(inp, Float8Tensor)
        blockwise_recipe = isinstance(inp, Float8BlockwiseQTensor)
        mxfp8_recipe = isinstance(inp, MXFP8Tensor)

        if fp8:
            fp8_dtype = inp._fp8_dtype
            fake_dtype = inp.dtype
            # blockwise scaling
            if blockwise_recipe:
                fp8_scale = inp._rowwise_scale_inv.T.contiguous()
                scale_hidden_dim = fp8_scale.shape[1]
                assert num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                inp = inp._rowwise_data
            # mxfp8 scaling
            elif mxfp8_recipe:
                fp8_scale = inp._rowwise_scale_inv.contiguous()
                scale_hidden_dim = fp8_scale.shape[1]
                assert num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                inp = inp._rowwise_data
            # per-tensor scaling
            elif per_tensor_recipe:
                # Kernel does not need scale in per-tensor scaling
                fp8_scale = None
                scale_hidden_dim = None
                fp8_scale_inv = inp._scale_inv
                inp = inp._data
            else:
                raise ValueError("Unsupported FP8 recipe")
        else:
            fp8_scale = None
            fp8_dtype = None
            scale_hidden_dim = None

        output, permuted_scale, permuted_probs = triton_permutation.permute_with_mask_map(
            inp,
            row_id_map,
            probs,
            fp8_scale,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
            scale_hidden_dim,
        )

        if fp8:
            if per_tensor_recipe:
                output = Float8Tensor(
                    data=output,
                    fp8_dtype=fp8_dtype,
                    fp8_scale_inv=fp8_scale_inv,
                    shape=output.shape,
                    dtype=fake_dtype,
                )
            elif blockwise_recipe:
                output = Float8BlockwiseQTensor(
                    shape=output.shape,
                    dtype=fake_dtype,
                    rowwise_data=output,
                    rowwise_scale_inv=permuted_scale.T.contiguous(),
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    fp8_dtype=fp8_dtype,
                    quantizer=None,
                    is_2D_scaled=False,
                    requires_grad=output.requires_grad,
                )
            elif mxfp8_recipe:
                output = MXFP8Tensor(
                    shape=output.shape,
                    dtype=fake_dtype,
                    fp8_dtype=fp8_dtype,
                    rowwise_data=output,
                    rowwise_scale_inv=permuted_scale.contiguous(),
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    quantizer=None,
                    requires_grad=output.requires_grad,
                )

        ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, row_id_map, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, ctx.probs

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            assert not isinstance(
                permuted_act_grad, QuantizedTensor
            ), "The backward of moe_permute does not support FP8."
            act_grad, probs_grad = triton_permutation.unpermute_with_mask_map(
                permuted_act_grad,
                row_id_map,
                None,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.num_experts,
                ctx.hidden_size,
            )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad


class _moe_unpermute_mask_map(torch.autograd.Function):
    """functional Unpermute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        merging_probs: Optional[torch.Tensor],
        restore_shape: Optional[torch.Size],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.merging_probs = merging_probs
            return inp

        if restore_shape is None:
            restore_shape = inp.shape
        num_tokens, hidden_size = restore_shape
        num_experts = (row_id_map.size(1) - 1) // 2

        with_probs = merging_probs is not None
        if with_probs:
            assert merging_probs.is_cuda, "TransformerEngine needs CUDA."

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."

        assert not isinstance(
            inp, QuantizedTensor
        ), "The forward of moe_unpermute does not support FP8."
        unpermuted_output, _ = triton_permutation.unpermute_with_mask_map(
            inp,
            row_id_map,
            merging_probs,
            None,
            num_tokens,
            num_experts,
            hidden_size,
        )

        if with_probs:
            ctx.save_for_backward(inp, row_id_map, merging_probs)
        else:
            ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.num_permuted_tokens = inp.size(0)
        ctx.hidden_size = hidden_size
        ctx.with_probs = with_probs
        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        # pylint: disable=missing-function-docstring
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.merging_probs, None

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            if ctx.with_probs:
                fwd_input, row_id_map, merging_probs = ctx.saved_tensors
            else:
                (row_id_map,) = ctx.saved_tensors

            fp8 = isinstance(unpermuted_act_grad, QuantizedTensor)
            per_tensor_recipe = isinstance(unpermuted_act_grad, Float8Tensor)
            blockwise_recipe = isinstance(unpermuted_act_grad, Float8BlockwiseQTensor)
            mxfp8_recipe = isinstance(unpermuted_act_grad, MXFP8Tensor)

            if fp8:
                fp8_dtype = unpermuted_act_grad._fp8_dtype
                fake_dtype = unpermuted_act_grad.dtype
                # per-tensor scaling
                if per_tensor_recipe:
                    # Kernel does not need scale in per-tensor scaling
                    fp8_scale = None
                    scale_hidden_dim = None
                    fp8_scale_inv = unpermuted_act_grad._scale_inv
                    unpermuted_act_grad = unpermuted_act_grad._data
                # blockwise scaling
                elif blockwise_recipe:
                    fp8_scale = unpermuted_act_grad._rowwise_scale_inv.T.contiguous()
                    unpermuted_act_grad = unpermuted_act_grad._rowwise_data
                    scale_hidden_dim = fp8_scale.shape[1]
                    assert ctx.num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                # mxfp8 scaling
                elif mxfp8_recipe:
                    fp8_scale = unpermuted_act_grad._rowwise_scale_inv.contiguous()
                    unpermuted_act_grad = unpermuted_act_grad._rowwise_data
                    scale_hidden_dim = fp8_scale.shape[1]
                    assert ctx.num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                else:
                    raise ValueError("Unsupported FP8 recipe")
            else:
                scale_hidden_dim = None
                fp8_dtype = None
                fp8_scale = None

            if ctx.with_probs:
                assert (
                    not fp8
                ), "The backward of moe_unpermute with merging probs does not support FP8."
                act_grad, probs_grad = (
                    triton_permutation.unpermute_with_mask_map_bwd_with_merging_probs(
                        unpermuted_act_grad,
                        row_id_map,
                        fwd_input,
                        merging_probs,
                        ctx.num_tokens,
                        ctx.num_experts,
                        ctx.num_permuted_tokens,
                        ctx.hidden_size,
                    )
                )
            else:
                act_grad, permuted_scale, _ = triton_permutation.permute_with_mask_map(
                    unpermuted_act_grad,
                    row_id_map,
                    None,
                    fp8_scale,
                    ctx.num_tokens,
                    ctx.num_experts,
                    ctx.num_permuted_tokens,
                    ctx.hidden_size,
                    scale_hidden_dim,
                )

            if fp8:
                if per_tensor_recipe:
                    act_grad = Float8Tensor(
                        data=act_grad,
                        fp8_dtype=fp8_dtype,
                        fp8_scale_inv=fp8_scale_inv,
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                    )
                elif blockwise_recipe:
                    act_grad = Float8BlockwiseQTensor(
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                        rowwise_data=act_grad,
                        rowwise_scale_inv=permuted_scale.T.contiguous(),
                        columnwise_data=None,
                        columnwise_scale_inv=None,
                        fp8_dtype=fp8_dtype,
                        quantizer=None,
                        is_2D_scaled=False,
                        requires_grad=act_grad.requires_grad,
                    )
                elif mxfp8_recipe:
                    act_grad = MXFP8Tensor(
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                        fp8_dtype=fp8_dtype,
                        rowwise_data=act_grad,
                        rowwise_scale_inv=permuted_scale.contiguous(),
                        columnwise_data=None,
                        columnwise_scale_inv=None,
                        quantizer=None,
                        requires_grad=act_grad.requires_grad,
                    )

        if not ctx.needs_input_grad[2]:
            probs_grad = None
        return act_grad, None, probs_grad, None


def moe_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
    max_token_num: int = -1,
    map_type: str = "mask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens based on the routing_map. Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    routing_map: torch.Tensor
        The token to expert mapping tensor.
        If map_type is 'mask', routing_map is of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
        If map_type is 'index', routing_map is of shape [num_tokens, topK] and dtype 'int32'.
        The values in it are the routed expert indices.
    num_out_tokens: int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    max_token_num: int, default = -1
        The maximum number of tokens, used for workspace allocation.
        By default, set to '-1', meaning the calculation of the size of workspace is
        automatically taken over by the operator.
    map_type: str, default = 'mask'
        Type of the routing map tensor.
        Options are: 'mask', 'index'.
        Refer to `routing_map` for more details.
    """
    if map_type == "index":
        return _moe_permute_index_map.apply(inp, routing_map, num_out_tokens, max_token_num)
    if map_type == "mask":
        output, row_id_map, _ = _moe_permute_mask_map.apply(inp, routing_map, num_out_tokens, None)
        return output, row_id_map
    raise ValueError("map_type should be one of 'mask' or 'index'")


def moe_permute_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens and probs based on the routing_map.
    Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs: torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens, num_experts]. It will be permuted with the tokens
        according to the routing_map.
    routing_map: torch.Tensor
        The token to expert mapping tensor of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
    num_out_tokens: int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    """
    output, row_id_map, permuted_probs = _moe_permute_mask_map.apply(
        inp, routing_map, num_out_tokens, probs
    )
    return output, permuted_probs, row_id_map


def moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor] = None,
    restore_shape: Optional[torch.Size] = None,
    map_type: str = "mask",
    probs: Optional[torch.Tensor] = None,
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
    merging_probs: torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    restore_shape: torch.Size, default = None
        The output shape after the unpermute operation.
    map_type: str, default = 'mask'
        Type of the routing map tensor. Should be the same as the value passed to moe_permute.
        Options are: 'mask', 'index'.
    probs: torch.Tensor, default = None
        Renamed to merging_probs. Keep for backward compatibility.
    """
    if probs is not None:
        if merging_probs is not None:
            raise ValueError(
                "Both merging_probs and probs kwarg are provided. probs is deprecated."
            )
        warnings.warn("probs kwarg is deprecated. Use merging_probs kwarg instead.")
        merging_probs = probs
    if map_type == "index":
        return _moe_unpermute_index_map.apply(inp, row_id_map, merging_probs)
    if map_type == "mask":
        return _moe_unpermute_mask_map.apply(inp, row_id_map, merging_probs, restore_shape)
    raise ValueError("map_type should be one of 'mask' or 'index'")


class _moe_chunk_sort(torch.autograd.Function):
    """functional MoE chunk permute"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            return inp, probs

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert split_sizes.is_cuda, "TransformerEngine needs CUDA."
        assert sorted_idxs.is_cuda, "TransformerEngine needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

        num_tokens, hidden_size = inp.shape
        num_splits = split_sizes.size(0)
        assert num_splits == sorted_idxs.size(0)

        fp8 = isinstance(inp, Float8Tensor)
        if fp8:
            fp8_dtype = inp._fp8_dtype
            fp8_scale_inv = inp._scale_inv
            fake_dtype = inp.dtype
            inp = inp._data

        row_id_map = triton_permutation.make_chunk_sort_map(
            split_sizes,
            sorted_idxs,
            num_tokens,
            num_splits,
        )
        output, permuted_probs = triton_permutation.sort_chunks_by_map(
            inp,
            row_id_map,
            probs,
            num_tokens,
            hidden_size,
            is_forward=True,
        )
        if fp8:
            output = Float8Tensor(
                data=output,
                fp8_dtype=fp8_dtype,
                fp8_scale_inv=fp8_scale_inv,
                shape=output.shape,
                dtype=fake_dtype,
            )

        ctx.save_for_backward(row_id_map)
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, permuted_probs_grad

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            fp8 = isinstance(permuted_act_grad, Float8Tensor)
            if fp8:
                fp8_dtype = permuted_act_grad._fp8_dtype
                fp8_scale_inv = permuted_act_grad._scale_inv
                fake_dtype = permuted_act_grad.dtype
                permuted_act_grad = permuted_act_grad._data
            act_grad, probs_grad = triton_permutation.sort_chunks_by_map(
                permuted_act_grad,
                row_id_map,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.hidden_size,
                is_forward=False,
            )
            if fp8:
                act_grad = Float8Tensor(
                    data=act_grad,
                    fp8_dtype=fp8_dtype,
                    fp8_scale_inv=fp8_scale_inv,
                    shape=act_grad.shape,
                    dtype=fake_dtype,
                )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad


def moe_sort_chunks_by_index(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    split_sizes: torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices: torch.Tensor
        Chunk indices used to permute the chunks.
    """
    output, _ = _moe_chunk_sort.apply(inp, split_sizes, sorted_index, None)
    return output


def moe_sort_chunks_by_index_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor and probs based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs: torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens]. It will be permuted with the tokens according to
        the split_sizes and sorted_indices.
    split_sizes: torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices: torch.Tensor
        Chunk indices used to permute the chunks.
    """
    output, permuted_probs = _moe_chunk_sort.apply(inp, split_sizes, sorted_index, probs)
    return output, permuted_probs
