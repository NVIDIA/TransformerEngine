# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused chunk sorting, padding and restoration for dispatched MoE tokens."""

from typing import Optional, Tuple

import torch
import triton

from transformer_engine.common.triton.chunk_padding import (
    copy_padded_chunks,
    make_padded_chunk_map,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


@torch.library.custom_op("te_moe::padded_chunk_map", mutates_args=[])
def _padded_chunk_map(
    splits: torch.Tensor, order: torch.Tensor, output_splits: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """Build a destination-to-source map without reading GPU metadata on the host."""
    result = torch.empty(num_rows, device=splits.device, dtype=torch.int32)
    if num_rows:
        make_padded_chunk_map[(triton.cdiv(num_rows, 128),)](
            splits,
            order,
            output_splits,
            result,
            splits.numel(),
            num_rows,
            triton.next_power_of_2(splits.numel()),
            128,
        )
    return result


@_padded_chunk_map.register_fake
def _padded_chunk_map_fake(
    splits: torch.Tensor, order: torch.Tensor, output_splits: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """Infer the map shape from the explicit output size."""
    del order, output_splits
    return torch.empty(num_rows, device=splits.device, dtype=torch.int32)


@torch.library.custom_op("te_moe::padded_chunk_copy", mutates_args=[])
def _padded_chunk_copy(
    inp: torch.Tensor,
    probs: Optional[torch.Tensor],
    row_map: torch.Tensor,
    num_rows: int,
    reverse: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a padded bijection or its inverse."""
    output = inp.new_empty((num_rows, inp.shape[1]))
    output_probs = (
        probs.new_empty(num_rows) if probs is not None else torch.empty(0, device=inp.device)
    )
    if row_map.numel():
        copy_padded_chunks[(triton.cdiv(row_map.numel() * inp.shape[1], 1024),)](
            inp,
            probs,
            row_map,
            output,
            output_probs,
            inp.stride(0),
            inp.stride(1),
            probs.stride(0) if probs is not None else 0,
            row_map.numel(),
            inp.shape[1],
            reverse,
            probs is not None,
            1024,
        )
    return output, output_probs


@_padded_chunk_copy.register_fake
def _padded_chunk_copy_fake(
    inp: torch.Tensor,
    probs: Optional[torch.Tensor],
    row_map: torch.Tensor,
    num_rows: int,
    reverse: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer token and probability output shapes."""
    del row_map, reverse
    return inp.new_empty((num_rows, inp.shape[1])), (
        probs.new_empty(num_rows) if probs is not None else torch.empty(0, device=inp.device)
    )


def _copy_setup_context(ctx, inputs, output):
    """Retain only the row map and scalar shape metadata for backward."""
    inp, probs, row_map, _, reverse = inputs
    ctx.save_for_backward(row_map)
    ctx.num_rows = inp.shape[0]
    ctx.reverse = reverse
    ctx.with_probs = probs is not None
    if probs is None:
        ctx.mark_non_differentiable(output[1])


def _copy_backward(ctx, grad, grad_probs):
    """The adjoint of a zero-padded bijection discards padded rows."""
    (row_map,) = ctx.saved_tensors
    dx, dp = _padded_chunk_copy(
        grad, grad_probs if ctx.with_probs else None, row_map, ctx.num_rows, not ctx.reverse
    )
    return dx, dp if ctx.with_probs else None, None, None, None


_padded_chunk_copy.register_autograd(_copy_backward, setup_context=_copy_setup_context)


def moe_sort_chunks_and_pad(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    output_split_sizes: torch.Tensor,
    num_out_tokens: int,
    probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Sort chunks and append zero padding in a single token copy.

    Parameters
    ----------
    inp : torch.Tensor
        CUDA FP32, FP16 or BF16 tokens of shape ``[num_tokens, hidden_size]``.
        Quantized tensors are not supported.
    split_sizes : torch.Tensor
        Contiguous CUDA integer chunk sizes in input order, summing to num_tokens.
    sorted_indices : torch.Tensor
        Contiguous CUDA integer permutation of the chunk indices.
    output_split_sizes : torch.Tensor
        Contiguous CUDA integer sizes in output order. Each entry must be at least
        ``split_sizes[sorted_indices[i]]``; extra rows are filled with zeros.
        To align an expert comprising several rank chunks, append its padding to
        the last chunk of that expert. For example, chunks of sizes [2, 3] can use
        output sizes [2, 6] to align their combined size to eight rows.
    num_out_tokens : int
        Sum of output_split_sizes. Supplied explicitly to avoid GPU-to-CPU
        synchronization and permit CUDA graph capture with a fixed output size.
    probs : torch.Tensor, optional
        CUDA FP32, FP16 or BF16 probabilities of shape ``[num_tokens]``.

    Returns
    -------
    output : torch.Tensor
        Sorted, zero-padded tokens of shape ``[num_out_tokens, hidden_size]``.
    output_probs : torch.Tensor or None
        Sorted, zero-padded probabilities, if provided.
    row_map : torch.Tensor
        Int32 destination-to-source indices, with -1 at padding positions.
        Pass this map and the original token count to moe_unpad_and_restore_chunks.

    Notes
    -----
    Metadata values must satisfy the contracts above. Values are not copied to
    the host for validation. Metadata may change during graph replay if the
    input and output buffer sizes remain fixed. Padding has zero gradient.
    """
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    if (
        isinstance(inp, QuantizedTensor)
        or not inp.is_cuda
        or inp.ndim != 2
        or inp.dtype not in dtypes
        or inp.shape[1] == 0
    ):
        raise ValueError("Expected a CUDA FP32, FP16 or BF16 token matrix")
    if split_sizes.numel() == 0 or any(
        tensor.device != inp.device
        or tensor.ndim != 1
        or not tensor.is_contiguous()
        or tensor.dtype not in (torch.int32, torch.int64)
        or tensor.numel() != split_sizes.numel()
        for tensor in (split_sizes, sorted_indices, output_split_sizes)
    ):
        raise ValueError("Expected matching nonempty CUDA integer metadata vectors")
    if num_out_tokens < inp.shape[0] or num_out_tokens >= 2**31:
        raise ValueError("Output size must cover all input rows and fit in int32")
    if probs is not None and (
        isinstance(probs, QuantizedTensor)
        or probs.device != inp.device
        or probs.shape != (inp.shape[0],)
        or probs.dtype not in dtypes
    ):
        raise ValueError("Expected one floating-point probability per input row")
    row_map = _padded_chunk_map(split_sizes, sorted_indices, output_split_sizes, num_out_tokens)
    output, output_probs = _padded_chunk_copy(inp, probs, row_map, num_out_tokens, False)
    return output, output_probs if probs is not None else None, row_map


def moe_unpad_and_restore_chunks(
    inp: torch.Tensor, row_map: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """Discard padding and restore the input order of moe_sort_chunks_and_pad.

    Parameters
    ----------
    inp : torch.Tensor
        Expert outputs of shape ``[num_padded_tokens, hidden_size]``. The hidden
        size may differ from the input passed to moe_sort_chunks_and_pad.
    row_map : torch.Tensor
        The unmodified map returned by moe_sort_chunks_and_pad.
    num_tokens : int
        Original number of unpadded tokens.

    Returns
    -------
    torch.Tensor
        Restored tokens. Backward fills padded rows with zero gradients.
    """
    if (
        isinstance(inp, QuantizedTensor)
        or not inp.is_cuda
        or inp.ndim != 2
        or inp.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or inp.shape[1] == 0
    ):
        raise ValueError("Expected a CUDA floating-point token matrix")
    if (
        row_map.device != inp.device
        or row_map.dtype != torch.int32
        or row_map.shape != (inp.shape[0],)
        or not row_map.is_contiguous()
        or not 0 <= num_tokens <= inp.shape[0]
    ):
        raise ValueError("Expected the padded row map and original token count")
    return _padded_chunk_copy(inp, None, row_map, num_tokens, True)[0]
