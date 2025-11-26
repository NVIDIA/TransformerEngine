# Copyright (c) 2025-2028, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX wrapper functions for Permutation Triton kernels."""

from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
import triton
import jax_triton as jt

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
    routing_map: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
) -> jnp.ndarray:
    """
    Prepare the row_id_map for the permutation using JAX-Triton.

    Parameters
    ----------
    routing_map : jnp.ndarray
        Input tensor of shape `[num_tokens, num_experts]`. It is a mask tensor that indicates
        which experts are routed to which tokens. The values in it: 1 means the token is routed to
        this expert and 0 means not.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.

    Returns
    -------
    row_id_map : jnp.ndarray
        The row_id_map for the permutation of shape `[num_tokens, num_experts * 2 + 1]`.
        For each token, the last item is the number of experts that are routed (n_routed).
        The first n_routed items are the destination row indices in the permuted tokens.
        The [num_experts, num_experts + n_routed) items are the indices of the experts corresponding
        to the first n_routed row indices above.
    """
    row_id_map_shape = (num_tokens, num_experts * 2 + 1)
    block_size = 1024
    grid = (num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor_shape = grid

    # supposing num_tokens == 5, num_experts == 3, block_size == 3
    # and we have a routing_map like this:
    # [[1, 1, 0],
    #  [1, 0, 1],
    #  [0, 0, 1],
    #  [1, 1, 0],
    #  [0, 0, 0]]

    # Pass 1: block cumsum
    # for each expert, compute the cumsum of every block_size tokens
    # the row_id_map will be like this after pass 1 (r means useless values):
    # [[1, 1, 0, r, r, r, r],
    #  [2, 0, 1, r, r, r, r],
    #  [0, 0, 2, r, r, r, r],
    #  [1, 1, 0, r, r, r, r],
    #  [0, 0, 0, r, r, r, r]]
    # Note: "r" = -1 in the triton common kernel implementation

    # Compute strides manually (JAX arrays don't have .strides attribute)
    # For routing_map of shape [num_tokens, num_experts]
    routing_stride_token = num_experts
    routing_stride_expert = 1
    # For row_id_map of shape [num_tokens, num_experts * 2 + 1]
    row_id_stride_token = num_experts * 2 + 1
    row_id_stride_expert = 1  # Move to next column (contiguous)

    # Pass 1: Block cumsum
    row_id_map_pass1, workspace_tensor = jt.triton_call(
        routing_map,
        num_tokens,
        routing_stride_token,
        routing_stride_expert,
        row_id_stride_token,
        row_id_stride_expert,
        kernel=_row_id_map_pass_1_kernel,
        out_shape=[
            ShapeDtypeStruct(row_id_map_shape, jnp.int32),
            ShapeDtypeStruct(workspace_tensor_shape, jnp.int32),
        ],
        grid=grid,
        BLOCK_SIZE=block_size,
    )

    # Pass 2: cumsum all and process the mask
    row_id_map_pass2, workspace_tensor = jt.triton_call(
        row_id_map_pass1,
        workspace_tensor,
        num_tokens,
        row_id_stride_token,
        row_id_stride_expert,
        kernel=_row_id_map_pass_2_kernel,
        out_shape=[
            ShapeDtypeStruct(row_id_map_shape, jnp.int32),
            ShapeDtypeStruct(workspace_tensor_shape, jnp.int32),
        ],
        input_output_aliases={0: 0, 1: 1},
        grid=grid,
        WORKSPACE_LOAD_WIDTH=triton.next_power_of_2(
            num_experts * triton.cdiv(num_tokens, block_size)
        ),
        BLOCK_SIZE=block_size,
    )

    # Initialize columns [num_experts:] to -1 since Pass 1/2 only wrote to [0:num_experts]
    # Reference implementation expects -1 for invalid entries, not garbage
    row_id_map = row_id_map_pass2.at[:, num_experts:].set(-1)

    # Pass 3: make the row_id_map from sparse to dense structure
    grid = (num_tokens,)
    load_size = triton.next_power_of_2(num_experts)
    row_id_map = jt.triton_call(
        row_id_map,
        row_id_stride_token,
        row_id_stride_expert,
        kernel=_row_id_map_pass_3_kernel,
        out_shape=ShapeDtypeStruct(row_id_map_shape, jnp.int32),
        input_output_aliases={0: 0},
        num_experts=num_experts,
        grid=grid,
        LOAD_SIZE=load_size,
    )

    return row_id_map


def permute_with_mask_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Permute the input tensor based on the row_id_map using JAX-Triton.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    probs : Optional[jnp.ndarray]
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.
    num_out_tokens : int
        Number of tokens in the permuted tensor.
    hidden_size : int
        Hidden size of the input tensor.

    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape `[num_out_tokens, hidden_size]`.
    permuted_probs : Optional[jnp.ndarray]
        Permuted probabilities if probs was provided, None otherwise.
    """
    # one block per token, multiple blocks for hidden dimension
    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))

    with_probs = probs is not None
    # Compute strides manually (JAX arrays don't have .strides attribute)
    # [num_tokens, hidden_size]
    inp_stride_token = hidden_size
    inp_stride_hidden = 1
    # [num_out_tokens, hidden_size]
    output_stride_token = hidden_size
    output_stride_hidden = 1
    # [num_tokens, num_experts * 2 + 1]
    row_id_stride_token = num_experts * 2 + 1
    row_id_stride_expert = 1

    # [num_out_tokens]
    permuted_probs_stride_token = 1

    if with_probs:
        if probs.ndim > 1:
            # [num_tokens, num_experts]
            probs_stride_token = num_experts
            probs_stride_expert = 1
        else:
            # [num_tokens]
            probs_stride_token = 1
            probs_stride_expert = 1
        out_shape = [
            ShapeDtypeStruct((num_out_tokens, hidden_size), inp.dtype),
            ShapeDtypeStruct((num_out_tokens,), probs.dtype),
        ]
    else:
        probs_stride_token = 0
        probs_stride_expert = 0
        probs = jnp.zeros((0,), dtype=inp.dtype)
        out_shape = [
            ShapeDtypeStruct((num_out_tokens, hidden_size), inp.dtype),
            ShapeDtypeStruct((0,), inp.dtype),
        ]

    dummy_scale = jnp.zeros((num_tokens, hidden_size), dtype=inp.dtype)
    dummy_permuted_scale = jnp.zeros((num_out_tokens, hidden_size), dtype=inp.dtype)

    outputs = jt.triton_call(
        inp,
        row_id_map,
        probs,
        dummy_scale, # scale
        dummy_permuted_scale, # permuted_scale
        0,
        row_id_stride_token,
        row_id_stride_expert,
        inp_stride_token,
        inp_stride_hidden,
        output_stride_token,
        output_stride_hidden,
        probs_stride_token,
        probs_stride_expert,
        hidden_size,
        1,
        permuted_probs_stride_token,
        hidden_size,
        1,
        kernel=_permute_kernel,
        out_shape=out_shape,
        grid=grid_fn,
        num_experts=num_experts,
        hidden_size=hidden_size,
        PERMUTE_PROBS=with_probs,
        PERMUTE_SCALE=False,
        # BLOCK_SIZE is keyword constexpr from autotune
    )

    output = outputs[0]
    if with_probs:
        permuted_probs = outputs[1]
    else:
        permuted_probs = None

    return output, permuted_probs


def unpermute_with_mask_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
    permuted_probs: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Unpermute the input tensor based on the row_id_map using JAX-Triton.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_out_tokens, hidden_size]`.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    merging_probs : Optional[jnp.ndarray]
        The merging probabilities of the input tensor. If it is not None, it will be used as weights
        to reduce the unpermuted tokens.
    permuted_probs : Optional[jnp.ndarray]
        The permuted probabilities of the input tensor. If it is not None, it will be unpermuted.
    num_tokens : int
        Number of tokens in the permuted tensor.
    num_experts : int
        Number of experts in the permuted tensor.
    hidden_size : int
        Hidden size of the permuted tensor.

    Returns
    -------
    output : jnp.ndarray
        Unpermuted output tensor of shape `[num_tokens, hidden_size]`.
    unpermuted_probs : Optional[jnp.ndarray]
        Unpermuted probabilities if permuted_probs was provided, None otherwise.
    """
    with_merging_probs = merging_probs is not None
    with_probs = permuted_probs is not None

    # Compute strides manually (JAX arrays don't have .strides attribute)
    # [num_out_tokens, hidden_size],
    inp_stride_token = hidden_size
    inp_stride_hidden = 1
    # [num_tokens, hidden_size],
    output_stride_token = hidden_size
    output_stride_hidden = 1
    # [num_tokens, num_experts * 2 + 1],
    row_id_stride_token = num_experts * 2 + 1
    row_id_stride_expert = 1
    # [num_tokens, num_experts] if present:
    if with_merging_probs:
        merging_probs_stride_token = num_experts
        merging_probs_stride_expert = 1
    else:
        merging_probs_stride_token = 0
        merging_probs_stride_expert = 0
    # [num_out_tokens] if present:
    permuted_probs_stride_token = 1
    # [num_tokens, num_experts] (output):
    unpermuted_probs_stride_token = num_experts
    unpermuted_probs_stride_expert = 1

    # One block per token, multiple blocks for hidden dimension
    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))

    merging_probs = merging_probs if with_merging_probs else jnp.zeros((0,), dtype=inp.dtype)
    permuted_probs = permuted_probs if with_probs else jnp.zeros((0,), dtype=inp.dtype)

    if with_probs:
        out_shape = [
            ShapeDtypeStruct((num_tokens, hidden_size), inp.dtype),
            ShapeDtypeStruct((num_tokens, num_experts), permuted_probs.dtype),
        ]
    else:
        out_shape = [
            ShapeDtypeStruct((num_tokens, hidden_size), inp.dtype),
            ShapeDtypeStruct((0,), inp.dtype),
        ]

    outputs = jt.triton_call(
        inp,
        row_id_map,
        merging_probs,
        permuted_probs,
        row_id_stride_token,
        row_id_stride_expert,
        inp_stride_token,
        inp_stride_hidden,
        output_stride_token,
        output_stride_hidden,
        merging_probs_stride_token,
        merging_probs_stride_expert,
        1,
        0,
        0,
        kernel=_unpermute_kernel,
        out_shape=out_shape,
        grid=grid_fn,
        num_experts=num_experts,
        hidden_size=hidden_size,
        PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
        WITH_MERGING_PROBS=with_merging_probs,
        PERMUTE_PROBS=with_probs,
        # BLOCK_SIZE is keyword constexpr from autotune
    )
    output = outputs[0]
    if with_probs:
        unpermuted_probs = outputs[1]
    else:
        unpermuted_probs = None

    return output, unpermuted_probs


def make_chunk_sort_map(
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
    num_tokens: int,
    num_splits: int,
) -> jnp.ndarray:
    """
    Make a row_id_map for chunk sort using JAX-Triton.

    Parameters
    ----------
    split_sizes : jnp.ndarray
        The sizes of the chunks of shape `[num_splits,]`.
    sorted_indices : jnp.ndarray
        The indices of the sorted chunks of shape `[num_splits,]`.
    num_tokens : int
        Number of tokens in the input tensor.
    num_splits : int
        Number of splits of split_sizes and sorted_indices.

    Returns
    -------
    row_id_map : jnp.ndarray
        Row ID map for chunk sorting of shape `[num_tokens,]`.
    """
    grid = (num_tokens,)

    row_id_map = jt.triton_call(
        split_sizes,
        sorted_indices,
        kernel=_make_chunk_sort_map_kernel,
        out_shape=[ShapeDtypeStruct((num_tokens,), jnp.int32)],
        grid=grid,
        num_splits=num_splits,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
    )[0]

    return row_id_map


def sort_chunks_by_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Sort chunks with row_id_map using JAX-Triton.

    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_tokens, hidden_size]`.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens,]`.
    probs : Optional[jnp.ndarray]
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens : int
        Number of tokens in the input tensor.
    hidden_size : int
        Hidden size of the input tensor.
    is_forward : bool
        Whether the sort is for forward or backward.

    Returns
    -------
    output : jnp.ndarray
        Sorted output tensor of shape `[num_tokens, hidden_size]`.
    permuted_probs : Optional[jnp.ndarray]
        Sorted probabilities if probs was provided, None otherwise.
    """
    # Compute strides manually (JAX arrays don't have .strides attribute)
    # [num_tokens, hidden_size]
    inp_stride_token = hidden_size
    inp_stride_hidden = 1
    output_stride_token = hidden_size
    output_stride_hidden = 1
    # [num_tokens]
    probs_stride_token = 1
    permuted_probs_stride_token = 1

    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))

    with_probs = probs is not None
    if with_probs:
        out_shape = [
            ShapeDtypeStruct((num_tokens, hidden_size), inp.dtype),
            ShapeDtypeStruct((num_tokens,), probs.dtype),
        ]
    else:
        out_shape = [
            ShapeDtypeStruct((num_tokens, hidden_size), inp.dtype),
            ShapeDtypeStruct((0,), inp.dtype),
        ]
        probs = jnp.zeros((0,), dtype=inp.dtype)

    outputs = jt.triton_call(
        inp,
        row_id_map,
        probs,
        inp_stride_token,
        inp_stride_hidden,
        output_stride_token,
        output_stride_hidden,
        probs_stride_token,
        permuted_probs_stride_token,
        kernel=_sort_chunks_by_map_kernel,
        out_shape=out_shape,
        grid=grid_fn,
        hidden_size=hidden_size,
        PERMUTE_PROBS=with_probs,
        # BLOCK_SIZE is provided by autotune
        FORWARD=is_forward,
    )
    output = outputs[0]
    if with_probs:
        permuted_probs = outputs[1]
    else:
        permuted_probs = None

    return output, permuted_probs
