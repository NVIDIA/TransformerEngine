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
    row_id_map = jnp.full((num_tokens, num_experts * 2 + 1), -1, dtype=jnp.int32)
    block_size = 1024
    grid = (num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor = jnp.zeros(grid, dtype=jnp.int32)
    
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
    # For routing_map of shape [num_tokens, num_experts], C-contiguous:
    routing_stride_token = num_experts  # Move to next token
    routing_stride_expert = 1           # Move to next expert (contiguous)
    # For row_id_map of shape [num_tokens, num_experts * 2 + 1], C-contiguous:
    row_id_stride_token = num_experts * 2 + 1  # Move to next token
    row_id_stride_expert = 1                    # Move to next column (contiguous)
    
    # Pass 1: Block cumsum
    row_id_map_pass1, workspace_tensor = jt.triton_call(
        routing_map,              # Input 0 (ptr): routing_map_ptr
        num_tokens,               # Scalar: num_tokens
        routing_stride_token,     # Scalar: stride_routing_map_token
        routing_stride_expert,    # Scalar: stride_routing_map_expert
        row_id_stride_token,      # Scalar: stride_row_id_map_token
        row_id_stride_expert,     # Scalar: stride_row_id_map_expert
        kernel=_row_id_map_pass_1_kernel,
        out_shape=[
            ShapeDtypeStruct(row_id_map.shape, row_id_map.dtype),
            ShapeDtypeStruct(workspace_tensor.shape, workspace_tensor.dtype),
        ],
        grid=grid,
        BLOCK_SIZE=block_size,  # Constexpr - pass as keyword
    )

    # Pass 2: cumsum all and process the mask
    # Strides remain the same as Pass 1
    # Note: Pass 2 takes the outputs from Pass 1 as inputs
    row_id_map_pass2, workspace_tensor = jt.triton_call(
        row_id_map_pass1,               # Input 0 (ptr): row_id_map_ptr (from Pass 1)
        workspace_tensor,         # Input 1 (ptr): workspace_ptr (from Pass 1)
        num_tokens,               # Scalar: num_tokens
        row_id_stride_token,      # Scalar: stride_row_id_map_token
        row_id_stride_expert,     # Scalar: stride_row_id_map_expert
        kernel=_row_id_map_pass_2_kernel,
        out_shape=[
            ShapeDtypeStruct(row_id_map.shape, row_id_map.dtype),
            ShapeDtypeStruct(workspace_tensor.shape, workspace_tensor.dtype),
        ],
        input_output_aliases={0: 0, 1: 1},  # row_id_map input→output, workspace input→output
        grid=grid,
        WORKSPACE_LOAD_WIDTH=triton.next_power_of_2(num_experts * triton.cdiv(num_tokens, block_size)),  # Constexpr
        BLOCK_SIZE=block_size,  # Constexpr
    )
    # Initialize columns [num_experts:] to -1 since Pass 1/2 only wrote to [0:num_experts]
    # Reference implementation expects -1 for invalid entries, not garbage
    row_id_map = row_id_map_pass2.at[:, num_experts:].set(-1)

    # Pass 3: make the row_id_map from sparse to dense structure
    grid = (num_tokens,)
    load_size = triton.next_power_of_2(num_experts)
    row_id_map = jt.triton_call(
        row_id_map,               # Input 0 (ptr): row_id_map_ptr (from Pass 2, with -1 initialized)
        row_id_stride_token,      # Scalar 1: stride_row_id_map_token
        row_id_stride_expert,     # Scalar 2: stride_row_id_map_expert
        kernel=_row_id_map_pass_3_kernel,
        out_shape=[ShapeDtypeStruct(row_id_map.shape, row_id_map.dtype)],
        input_output_aliases={0: 0},  # row_id_map input→output
        num_experts=num_experts,
        grid=grid,
        LOAD_SIZE=load_size,  # Constexpr
    )[0]
    
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
    # Compute strides manually (JAX arrays don't have .strides attribute)
    # For inp of shape [num_tokens, hidden_size], C-contiguous:
    inp_stride_token = hidden_size
    inp_stride_hidden = 1
    # For output of shape [num_out_tokens, hidden_size], C-contiguous:
    output_stride_token = hidden_size
    output_stride_hidden = 1
    # For row_id_map of shape [num_tokens, num_experts * 2 + 1], C-contiguous:
    row_id_stride_token = num_experts * 2 + 1
    row_id_stride_expert = 1
    # For probs: depends on dimensionality
    if probs is not None:
        if probs.ndim > 1:
            # Shape [num_tokens, num_experts]
            probs_stride_token = num_experts
            probs_stride_expert = 1
        else:
            # Shape [num_tokens]
            probs_stride_token = 1
            probs_stride_expert = 1
    else:
        probs_stride_token = 0
        probs_stride_expert = 0
    # For permuted_probs of shape [num_out_tokens], C-contiguous:
    permuted_probs_stride_token = 1
    
    # Grid: one block per token, multiple blocks for hidden dimension
    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta['BLOCK_SIZE']))
    
    if probs is not None:
        # jax-triton doesn't handle None pointers correctly, create dummy tensors
        # Make dummy tensors large enough to not cause out-of-bounds access
        dummy_scale = jnp.zeros((num_tokens, hidden_size), dtype=inp.dtype)
        dummy_permuted_scale = jnp.zeros((num_out_tokens, hidden_size), dtype=inp.dtype)
        
        output, permuted_probs = jt.triton_call(
            inp,                  # Input 0 (ptr): input_ptr
            row_id_map,           # Input 1 (ptr): row_id_map_ptr
            probs,                # Input 2 (ptr): probs_ptr
            dummy_scale,          # Input 3 (ptr): scale_ptr (dummy, not used)
            dummy_permuted_scale, # Input 4 (ptr): permuted_scale_ptr (dummy, not used)
            0,                    # Scalar 5: scale_hidden_dim (not used)
            row_id_stride_token,  # Scalar 6: stride_row_id_map_token
            row_id_stride_expert, # Scalar 7: stride_row_id_map_expert
            inp_stride_token,     # Scalar 8: stride_input_token
            inp_stride_hidden,    # Scalar 9: stride_input_hidden
            output_stride_token,  # Scalar 10: stride_output_token
            output_stride_hidden, # Scalar 11: stride_output_hidden
            probs_stride_token,   # Scalar 12: stride_probs_token
            probs_stride_expert,  # Scalar 13: stride_probs_expert
            hidden_size,          # Scalar 14: stride_scale_token (use actual stride)
            1,                    # Scalar 15: stride_scale_hidden
            permuted_probs_stride_token,  # Scalar 16: stride_permuted_probs_token
            hidden_size,          # Scalar 17: stride_permuted_scale_token (use actual stride)
            1,                    # Scalar 18: stride_permuted_scale_hidden
            kernel=_permute_kernel,
            out_shape=[
                ShapeDtypeStruct((num_out_tokens, hidden_size), inp.dtype),  # Positional: output_ptr
                ShapeDtypeStruct((num_out_tokens,), probs.dtype),  # Positional: permuted_probs_ptr
            ],
            grid=grid_fn,
            num_experts=num_experts,  # Keyword constexpr
            hidden_size=hidden_size,  # Keyword constexpr
            PERMUTE_PROBS=True,   # Keyword constexpr
            PERMUTE_SCALE=False,  # Keyword constexpr
            # BLOCK_SIZE is keyword constexpr from autotune
        )
    else:
        # jax-triton doesn't handle None pointers correctly, create dummy tensors
        # Make dummy tensors large enough to not cause out-of-bounds access
        dummy_probs = jnp.zeros((num_tokens, num_experts), dtype=inp.dtype)
        dummy_scale = jnp.zeros((num_tokens, hidden_size), dtype=inp.dtype)
        dummy_permuted_scale = jnp.zeros((num_out_tokens, hidden_size), dtype=inp.dtype)
        
        result = jt.triton_call(
            inp,                  # Input 0 (ptr): input_ptr
            row_id_map,           # Input 1 (ptr): row_id_map_ptr
            dummy_probs,          # Input 2 (ptr): probs_ptr (dummy, not used)
            dummy_scale,          # Input 3 (ptr): scale_ptr (dummy, not used)
            dummy_permuted_scale, # Input 4 (ptr): permuted_scale_ptr (dummy, not used)
            0,                    # Scalar 5: scale_hidden_dim (not used)
            row_id_stride_token,  # Scalar 6: stride_row_id_map_token
            row_id_stride_expert, # Scalar 7: stride_row_id_map_expert
            inp_stride_token,     # Scalar 8: stride_input_token
            inp_stride_hidden,    # Scalar 9: stride_input_hidden
            output_stride_token,  # Scalar 10: stride_output_token
            output_stride_hidden, # Scalar 11: stride_output_hidden
            probs_stride_token,   # Scalar 12: stride_probs_token (use actual)
            probs_stride_expert,  # Scalar 13: stride_probs_expert (use actual)
            hidden_size,          # Scalar 14: stride_scale_token (use actual stride)
            1,                    # Scalar 15: stride_scale_hidden
            permuted_probs_stride_token,  # Scalar 16: stride_permuted_probs_token (use actual)
            hidden_size,          # Scalar 17: stride_permuted_scale_token (use actual stride)
            1,                    # Scalar 18: stride_permuted_scale_hidden
            kernel=_permute_kernel,
            out_shape=[
                ShapeDtypeStruct((num_out_tokens, hidden_size), inp.dtype),  # Positional: output_ptr
                ShapeDtypeStruct((num_out_tokens,), inp.dtype),  # Positional: permuted_probs_ptr (dummy)
            ],
            grid=grid_fn,
            num_experts=num_experts,  # Keyword constexpr
            hidden_size=hidden_size,  # Keyword constexpr
            PERMUTE_PROBS=False,  # Keyword constexpr
            PERMUTE_SCALE=False,  # Keyword constexpr
            # BLOCK_SIZE is keyword constexpr from autotune
        )
        output = result[0]
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
    # Compute strides manually (JAX arrays don't have .strides attribute)
    # For inp of shape [num_out_tokens, hidden_size], C-contiguous:
    inp_stride_token = hidden_size
    inp_stride_hidden = 1
    # For output of shape [num_tokens, hidden_size], C-contiguous:
    output_stride_token = hidden_size
    output_stride_hidden = 1
    # For row_id_map of shape [num_tokens, num_experts * 2 + 1], C-contiguous:
    row_id_stride_token = num_experts * 2 + 1
    row_id_stride_expert = 1
    # For merging_probs of shape [num_tokens, num_experts] if present:
    if merging_probs is not None:
        merging_probs_stride_token = num_experts
        merging_probs_stride_expert = 1
    else:
        merging_probs_stride_token = 0
        merging_probs_stride_expert = 0
    # For permuted_probs of shape [num_out_tokens] if present:
    permuted_probs_stride_token = 1
    # For unpermuted_probs of shape [num_tokens, num_experts] (output):
    unpermuted_probs_stride_token = num_experts
    unpermuted_probs_stride_expert = 1
    
    # Grid: one block per token, multiple blocks for hidden dimension
    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta['BLOCK_SIZE']))
    
    if permuted_probs is not None:
        # Ensure merging_probs is not None (use dummy if needed)
        merging_probs_arg = merging_probs if merging_probs is not None else jnp.zeros((num_tokens, num_experts), dtype=inp.dtype)
        
        output, unpermuted_probs = jt.triton_call(
            inp,                  # Input 0 (ptr): input_ptr
            row_id_map,           # Input 1 (ptr): row_id_map_ptr
            merging_probs_arg,    # Input 2 (ptr): merging_probs_ptr (real or dummy)
            permuted_probs,       # Input 3 (ptr): permuted_probs_ptr
            row_id_stride_token,         # Scalar 4: stride_row_id_map_token
            row_id_stride_expert,        # Scalar 5: stride_row_id_map_expert
            inp_stride_token,            # Scalar 6: stride_input_token
            inp_stride_hidden,           # Scalar 7: stride_input_hidden
            output_stride_token,         # Scalar 8: stride_output_token
            output_stride_hidden,        # Scalar 9: stride_output_hidden
            merging_probs_stride_token,  # Scalar 10: stride_merging_probs_token
            merging_probs_stride_expert, # Scalar 11: stride_merging_probs_expert
            permuted_probs_stride_token, # Scalar 12: stride_permuted_probs_token
            unpermuted_probs_stride_token,   # Scalar 13: stride_unpermuted_probs_token
            unpermuted_probs_stride_expert,  # Scalar 14: stride_unpermuted_probs_expert
            kernel=_unpermute_kernel,
            out_shape=[
                ShapeDtypeStruct((num_tokens, hidden_size), inp.dtype),  # Positional: output_ptr
                ShapeDtypeStruct((num_tokens, num_experts), permuted_probs.dtype),  # Positional: unpermuted_probs_ptr
            ],
            grid=grid_fn,
            num_experts=num_experts,             # Keyword constexpr
            hidden_size=hidden_size,             # Keyword constexpr
            PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),  # Keyword constexpr
            WITH_MERGING_PROBS=merging_probs is not None,  # Keyword constexpr
            PERMUTE_PROBS=True,   # Keyword constexpr
            # BLOCK_SIZE is keyword constexpr from autotune
        )
    else:
        # jax-triton doesn't handle None pointers correctly, create dummy tensors if needed
        dummy_permuted_probs = jnp.zeros((num_tokens,), dtype=inp.dtype)  # Proper size dummy
        merging_probs_arg = merging_probs if merging_probs is not None else jnp.zeros((num_tokens, num_experts), dtype=inp.dtype)
        
        result = jt.triton_call(
            inp,                  # Input 0 (ptr): input_ptr
            row_id_map,           # Input 1 (ptr): row_id_map_ptr
            merging_probs_arg,    # Input 2 (ptr): merging_probs_ptr (real or dummy)
            dummy_permuted_probs, # Input 3 (ptr): permuted_probs_ptr (dummy, not used)
            row_id_stride_token,         # Scalar 4: stride_row_id_map_token
            row_id_stride_expert,        # Scalar 5: stride_row_id_map_expert
            inp_stride_token,            # Scalar 6: stride_input_token
            inp_stride_hidden,           # Scalar 7: stride_input_hidden
            output_stride_token,         # Scalar 8: stride_output_token
            output_stride_hidden,        # Scalar 9: stride_output_hidden
            merging_probs_stride_token,  # Scalar 10: stride_merging_probs_token
            merging_probs_stride_expert, # Scalar 11: stride_merging_probs_expert
            1,                    # Scalar 12: stride_permuted_probs_token (dummy stride)
            0,                    # Scalar 13: stride_unpermuted_probs_token
            0,                    # Scalar 14: stride_unpermuted_probs_expert
            kernel=_unpermute_kernel,
            out_shape=[
                ShapeDtypeStruct((num_tokens, hidden_size), inp.dtype),  # Positional: output_ptr
                ShapeDtypeStruct((num_tokens, num_experts), inp.dtype),  # Positional: unpermuted_probs_ptr (dummy)
            ],
            grid=grid_fn,
            num_experts=num_experts,             # Keyword constexpr
            hidden_size=hidden_size,             # Keyword constexpr
            PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),  # Keyword constexpr
            WITH_MERGING_PROBS=merging_probs is not None,  # Keyword constexpr
            PERMUTE_PROBS=False,  # Keyword constexpr
            # BLOCK_SIZE is keyword constexpr from autotune
        )
        output = result[0]
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
        split_sizes,          # Input 0 (ptr): split_sizes_ptr
        sorted_indices,       # Input 1 (ptr): sorted_indices_ptr
        kernel=_make_chunk_sort_map_kernel,
        out_shape=[ShapeDtypeStruct((num_tokens,), jnp.int32)],
        grid=grid,
        num_splits=num_splits,  # Constexpr
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),  # Constexpr
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
    # For inp and output of shape [num_tokens, hidden_size], C-contiguous:
    inp_stride_token = hidden_size
    inp_stride_hidden = 1
    output_stride_token = hidden_size
    output_stride_hidden = 1
    # For probs and permuted_probs of shape [num_tokens], C-contiguous:
    probs_stride_token = 1
    permuted_probs_stride_token = 1
    
    # Grid: one block per token, multiple blocks for hidden dimension
    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta['BLOCK_SIZE']))
    
    if probs is not None:
        output, permuted_probs = jt.triton_call(
            inp,                  # Input 0 (ptr): input_ptr
            row_id_map,           # Input 1 (ptr): row_id_map_ptr
            probs,                # Input 2 (ptr): probs_ptr
            inp_stride_token,     # Scalar 3: stride_input_token
            inp_stride_hidden,    # Scalar 4: stride_input_hidden
            output_stride_token,  # Scalar 5: stride_output_token
            output_stride_hidden, # Scalar 6: stride_output_hidden
            probs_stride_token,   # Scalar 7: stride_probs_token
            permuted_probs_stride_token,  # Scalar 8: stride_permuted_probs_token
            kernel=_sort_chunks_by_map_kernel,
            out_shape=[
                ShapeDtypeStruct((num_tokens, hidden_size), inp.dtype),  # Added at end: output_ptr
                ShapeDtypeStruct((num_tokens,), probs.dtype),  # Added at end: permuted_probs_ptr
            ],
            grid=grid_fn,
            hidden_size=hidden_size,  # Constexpr 9: hidden_size
            PERMUTE_PROBS=True,   # Constexpr 10: PERMUTE_PROBS
            # BLOCK_SIZE is Constexpr 11, provided by autotune
            FORWARD=is_forward,   # Constexpr 12: FORWARD
            # output_ptr and permuted_probs_ptr (13-14) are added automatically by jax-triton from out_shape
        )
    else:
        # Note: jax-triton might not handle None correctly, so create a dummy probs tensor
        dummy_probs = jnp.zeros((num_tokens,), dtype=inp.dtype)
        
        result = jt.triton_call(
            inp,                  # Input 0 (ptr): input_ptr
            row_id_map,           # Input 1 (ptr): row_id_map_ptr
            dummy_probs,          # Input 2 (ptr): probs_ptr (dummy, not used by kernel)
            inp_stride_token,     # Scalar 3: stride_input_token
            inp_stride_hidden,    # Scalar 4: stride_input_hidden
            output_stride_token,  # Scalar 5: stride_output_token
            output_stride_hidden, # Scalar 6: stride_output_hidden
            probs_stride_token,   # Scalar 7: stride_probs_token (use actual stride)
            permuted_probs_stride_token,  # Scalar 8: stride_permuted_probs_token
            kernel=_sort_chunks_by_map_kernel,
            out_shape=[
                ShapeDtypeStruct((num_tokens, hidden_size), inp.dtype),  # Positional after strides: output_ptr
                ShapeDtypeStruct((num_tokens,), inp.dtype),  # Positional after strides: permuted_probs_ptr (dummy)
            ],
            grid=grid_fn,
            hidden_size=hidden_size,  # Keyword constexpr
            PERMUTE_PROBS=False,  # Keyword constexpr
            FORWARD=is_forward,   # Keyword constexpr
            # BLOCK_SIZE is added by autotune as keyword constexpr
        )
        output = result[0]
        permuted_probs = None
    
    return output, permuted_probs