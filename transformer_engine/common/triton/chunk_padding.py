# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Chunk permutation with zero padding between output chunks."""

import triton
import triton.language as tl


@triton.jit
def make_padded_chunk_map(
    split_sizes,
    sorted_indices,
    output_split_sizes,
    row_map,
    num_chunks: tl.constexpr,
    num_output_rows: tl.constexpr,
    CHUNKS: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Map each output row to an input row, or -1 for padding."""
    chunks = tl.arange(0, CHUNKS)
    sizes = tl.load(split_sizes + chunks, chunks < num_chunks, other=0).to(tl.int32)
    starts = tl.cumsum(sizes) - sizes
    order = tl.load(sorted_indices + chunks, chunks < num_chunks, other=0).to(tl.int32)
    output_sizes = tl.load(output_split_sizes + chunks, chunks < num_chunks, other=0)
    ends = tl.cumsum(output_sizes.to(tl.int32))
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    output_chunk = tl.sum(
        ((rows[:, None] >= ends[None, :]) & (chunks[None, :] < num_chunks)).to(tl.int32),
        axis=1,
    )
    safe_chunk = tl.minimum(output_chunk, num_chunks - 1)
    source_chunk = tl.gather(order, safe_chunk, axis=0)
    offset = rows - tl.gather(ends - output_sizes, safe_chunk, axis=0)
    source = tl.gather(starts, source_chunk, axis=0) + offset
    valid = (output_chunk < num_chunks) & (offset < tl.gather(sizes, source_chunk, axis=0))
    tl.store(row_map + rows, tl.where(valid, source, -1), rows < num_output_rows)


@triton.jit
def copy_padded_chunks(
    inp,
    probs,
    row_map,
    output,
    output_probs,
    stride_row,
    stride_col,
    stride_prob,
    num_padded_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    REVERSE: tl.constexpr,
    WITH_PROBS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy tokens and probabilities together; reverse copies discard padding."""
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = index // hidden_size
    col = index % hidden_size
    active = row < num_padded_rows
    original = tl.load(row_map + row, active, other=-1).to(tl.int64)
    if REVERSE:
        source_row, target_row = row, original
    else:
        source_row, target_row = original, row
    value = tl.load(
        inp + source_row * stride_row + col * stride_col, active & (original >= 0), other=0
    )
    store = active & (original >= 0) if REVERSE else active
    tl.store(output + target_row * hidden_size + col, value, store)
    if WITH_PROBS:
        probability = tl.load(
            probs + source_row * stride_prob, active & (original >= 0) & (col == 0), other=0
        )
        tl.store(output_probs + target_row, probability, store & (col == 0))
