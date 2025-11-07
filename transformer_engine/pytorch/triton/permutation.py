# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for Permutation Triton kernels."""

from typing import Union

import torch
import triton

from transformer_engine.common.triton.permutation import (
    _row_id_map_pass_1_kernel,
    _row_id_map_pass_2_kernel,
    _row_id_map_pass_3_kernel,
    _permute_kernel,
    _unpermute_kernel,
    _unpermute_bwd_with_merging_probs_kernel,
    _make_chunk_sort_map_kernel,
    _sort_chunks_by_map_kernel,
)


def make_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
):
    """
    Prepare the row_id_map for the permutation.

    Parameters
    ----------
    routing_map: torch.Tensor
        Input tensor of shape `[num_tokens, num_experts]`. It is a mask tensor that indicates
        which experts are routed to which tokens. The values in it: 1 means the token is routed to
        this expert and 0 means not.
    num_tokens: int
        Number of tokens in the input tensor.
    num_experts: int
        Number of experts in the input tensor.

    Returns
    -------
    row_id_map: torch.Tensor
        The row_id_map for the permutation of shape `[num_tokens, num_experts * 2 + 1]`.
        For each token, the last item is the number of experts that are routed (n_routed).
        The first n_routed items are the destination row indices in the permuted tokens.
        The [num_experts, num_experts + n_routed) items are the indices of the experts corresponding
        to the first n_routed row indices above.
    """
    row_id_map = torch.empty((num_tokens, num_experts * 2 + 1), dtype=torch.int32, device="cuda")
    block_size = 1024
    grid = (num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor = torch.empty(grid, dtype=torch.int32, device="cuda")

    # supposing num_tokens == 5, num_experts == 3, block_size == 3
    # and we have a routing_map like this:
    # [[1, 1, 0],
    #  [1, 0, 1],
    #  [0, 0, 1],
    #  [1, 1, 0],
    #  [0, 0, 0]]

    # pass 1: block cumsum
    # for each expert, compute the cumsum of every block_size tokens
    # the row_id_map will be like this after pass 1 (r means useless values):
    # [[1, 1, 0, r, r, r, r],
    #  [2, 0, 1, r, r, r, r],
    #  [0, 0, 2, r, r, r, r],
    #  [1, 1, 0, r, r, r, r],
    #  [0, 0, 0, r, r, r, r]]
    _row_id_map_pass_1_kernel[grid](
        routing_map,
        row_id_map,
        workspace_tensor,
        num_tokens,
        routing_map.stride(0),
        routing_map.stride(1),
        row_id_map.stride(0),
        row_id_map.stride(1),
        block_size,
    )

    # pass 2: cumsum all and process the mask
    # process the block cumsum into the global cumsum and then into the dst row indices
    # the row_id_map will be like this after pass 2 (r means useless value):
    # [[ 0,  3, -1, r, r, r, r],
    #  [ 1, -1,  5, r, r, r, r],
    #  [-1, -1,  6, r, r, r, r],
    #  [ 2,  4, -1, r, r, r, r],
    #  [-1, -1, -1, r, r, r, r]]
    _row_id_map_pass_2_kernel[grid](
        row_id_map,
        workspace_tensor,
        num_tokens,
        row_id_map.stride(0),
        row_id_map.stride(1),
        triton.next_power_of_2(num_experts * triton.cdiv(num_tokens, block_size)),
        block_size,
    )

    # pass 3: make the row_id_map from the sparse structure to the dense structure
    # the row_id_map will be like this after pass 3 (r means useless value):
    # [[3, 0, r, 1, 0, r, 2],
    #  [5, 1, r, 2, 0, r, 2],
    #  [6, r, r, 2, r, r, 1],
    #  [4, 2, r, 1, 0, r, 2],
    #  [r, r, r, r, r, r, 0]]
    grid = (num_tokens,)
    _row_id_map_pass_3_kernel[grid](
        row_id_map,
        num_experts,
        row_id_map.stride(0),
        row_id_map.stride(1),
        triton.next_power_of_2(num_experts),
    )
    return row_id_map


def permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    scale: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    scale_hidden_dim: int,
):
    """
    Permute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    probs: torch.Tensor
        The probabilities of the input tensor. If it is not None, it will be permuted.
    scale: torch.Tensor
        The scale of the input tensor. If it is not None, it will be permuted.
    num_tokens: int
        Number of tokens in the input tensor.
    num_experts: int
        Number of experts in the input tensor.
    num_out_tokens: int
        Number of tokens in the permuted tensor.
    hidden_size: int
        Hidden size of the input tensor.
    scale_hidden_dim: int
        Hidden size of the scale tensor.
    """
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_out_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None

    if scale is not None:
        permuted_scale = torch.empty(
            (num_out_tokens, scale_hidden_dim), dtype=scale.dtype, device="cuda"
        )
    else:
        permuted_scale = None
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda META: (num_tokens, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))
    _permute_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        scale,
        permuted_probs,
        permuted_scale,
        num_experts,
        hidden_size,
        scale_hidden_dim,
        row_id_map.stride(0),
        row_id_map.stride(1),
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        probs.stride(1) if probs is not None else None,
        scale.stride(0) if scale is not None else None,
        scale.stride(1) if scale is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        permuted_scale.stride(0) if permuted_scale is not None else None,
        permuted_scale.stride(1) if permuted_scale is not None else None,
        PERMUTE_PROBS=probs is not None,
        PERMUTE_SCALE=scale is not None,
    )
    return output, permuted_scale, permuted_probs


def unpermute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Union[torch.Tensor, None],
    permuted_probs: Union[torch.Tensor, None],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
):
    """
    Unpermute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_out_tokens, hidden_size]`.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    merging_probs: torch.Tensor
        The merging probabilities of the input tensor. If it is not None, it will be used as weights
        to reduce the unpermuted tokens.
    permuted_probs: torch.Tensor
        The permuted probabilities of the input tensor. If it is not None, it will be unpermuted.
    num_tokens: int
        Number of tokens in the permuted tensor.
    num_experts: int
        Number of experts in the permuted tensor.
    hidden_size: int
        Hidden size of the permuted tensor.
    """
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if permuted_probs is not None:
        unpermuted_probs = torch.empty(
            (num_tokens, num_experts), dtype=permuted_probs.dtype, device="cuda"
        )
    else:
        unpermuted_probs = None
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda META: (num_tokens, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))
    _unpermute_kernel[grid](
        inp,
        output,
        row_id_map,
        merging_probs,
        permuted_probs,
        unpermuted_probs,
        num_experts,
        hidden_size,
        row_id_map.stride(0),
        row_id_map.stride(1),
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        merging_probs.stride(0) if merging_probs is not None else None,
        merging_probs.stride(1) if merging_probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        unpermuted_probs.stride(0) if unpermuted_probs is not None else None,
        unpermuted_probs.stride(1) if unpermuted_probs is not None else None,
        PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
        WITH_MERGING_PROBS=merging_probs is not None,
        PERMUTE_PROBS=permuted_probs is not None,
    )
    return output, unpermuted_probs


def unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    """
    Unpermute backward pass kernel with merging probs.

    Parameters
    ----------
    fwd_output_grad: torch.Tensor
        The gradient of the output tensor of shape `[num_tokens, hidden_size]`.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    fwd_input: torch.Tensor
        The input tensor of the forward pass of shape `[num_out_tokens, hidden_size]`.
    merging_probs: torch.Tensor
        The merging probabilities of the input tensor of shape `[num_tokens, num_experts]`.
    num_tokens: int
        Number of tokens in the permuted tensor.
    num_experts: int
        Number of experts in the permuted tensor.
    num_out_tokens: int
        Number of tokens in the output tensor.
    hidden_size: int
        Hidden size of the output tensor.
    """
    act_grad = torch.empty(
        (num_out_tokens, hidden_size), dtype=fwd_output_grad.dtype, device="cuda"
    )
    merging_probs_grad = torch.empty(
        (num_tokens, num_experts), dtype=merging_probs.dtype, device="cuda"
    )
    grid = (num_tokens,)
    _unpermute_bwd_with_merging_probs_kernel[grid](
        fwd_output_grad,
        act_grad,
        fwd_input,
        merging_probs,
        merging_probs_grad,
        row_id_map,
        num_experts,
        hidden_size,
        row_id_map.stride(0),
        row_id_map.stride(1),
        fwd_output_grad.stride(0),
        fwd_output_grad.stride(1),
        act_grad.stride(0),
        act_grad.stride(1),
        fwd_input.stride(0),
        fwd_input.stride(1),
        merging_probs.stride(0),
        merging_probs.stride(1),
        merging_probs_grad.stride(0),
        merging_probs_grad.stride(1),
        PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
    )
    return act_grad, merging_probs_grad


def make_chunk_sort_map(
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    num_splits: int,
):
    """
    Make a row_id_map for chunk sort.

    Parameters
    ----------
    split_sizes: torch.Tensor
        The sizes of the chunks of shape `[num_splits,]`.
    sorted_indices: torch.Tensor
        The indices of the sorted chunks of shape `[num_splits,]`.
    num_tokens: int
        Number of tokens in the input tensor.
    num_splits: int
        Number of splits of split_sizes and sorted_indices.
    """
    row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device="cuda")
    grid = (num_tokens,)
    _make_chunk_sort_map_kernel[grid](
        split_sizes,
        sorted_indices,
        row_id_map,
        num_splits,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
    )
    return row_id_map


def sort_chunks_by_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
):
    """
    Sort chunks with row_id_map.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens,]`.
    probs: torch.Tensor
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens: int
        Number of tokens in the input tensor.
    hidden_size: int
        Hidden size of the input tensor.
    is_forward: bool
        Whether the sort is for forward or backward.
    """
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None
    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda META: (num_tokens, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))
    _sort_chunks_by_map_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
        FORWARD=is_forward,
    )
    return output, permuted_probs
