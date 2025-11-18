# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Efficient Permutation kernels written with OpenAI Triton."""

import triton
import triton.language as tl

from triton.language import core
from triton.language.standard import _log2
from packaging import version


# The following three argsort related kernels are adapted from
# the issue https://github.com/triton-lang/triton/issues/3698

get_int_dtype = core.get_int_dtype
if version.parse(triton.__version__) >= version.parse("3.5.0"):
    get_int_dtype = triton.constexpr_function(get_int_dtype)


@triton.jit
def _compare_and_swap(x, indices, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * (2**i), 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    z = tl.reshape(indices, shape)

    mask = tl.arange(0, 2)[None, :, None]

    l_value = tl.reshape(tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape), x.shape).to(
        x.dtype
    )
    r_value = tl.reshape(tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape), x.shape).to(
        x.dtype
    )

    l_indice = tl.reshape(tl.broadcast_to(tl.sum(z * (1 - mask), 1)[:, None, :], shape), x.shape)
    r_indice = tl.reshape(tl.broadcast_to(tl.sum(z * mask, 1)[:, None, :], shape), x.shape)

    idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    il_value = l_value.to(idtype, bitcast=True)
    ir_value = r_value.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    flag1 = tl.where(((l_value > r_value) ^ flip) != 0, il_value ^ ir_value, tl.zeros_like(ix))
    ret = ix ^ flag1
    flag2 = tl.where(((l_value > r_value) ^ flip) != 0, l_indice ^ r_indice, tl.zeros_like(ix))
    ind = indices ^ flag2

    return ret.to(x.dtype, bitcast=True), ind


@triton.jit
def _bitonic_merge(x, indices, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    if order == 2:
        shape: tl.constexpr = [n_outer * (2 ** (n_dims - 1 - stage)), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = tl.full(x.shape, value=order, dtype=tl.int32)
    for i in tl.static_range(stage):
        x, indices = _compare_and_swap(x, indices, flip, i + (n_dims - stage), n_dims)
    return x, indices


@triton.jit
def _argsort(x, indices, n_dims: tl.constexpr):
    for i in tl.static_range(1, n_dims + 1):
        x, indices = _bitonic_merge(x, indices, i, 2 if i < n_dims else 1, n_dims)
    return x, indices


@triton.jit
def _row_id_map_pass_1_kernel(
    # pointers
    routing_map_ptr,
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # strides
    stride_routing_map_token,
    stride_routing_map_expert,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_token_mask = tl.load(
        routing_map_ptr + pid_m * stride_routing_map_expert + offset * stride_routing_map_token,
        mask=(offset < num_tokens),
        other=0,
    ).to(tl.int32)
    row_id_within_token_block = tl.cumsum(expert_token_mask) * expert_token_mask
    tl.store(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        row_id_within_token_block,
        mask=offset < num_tokens,
    )
    n_tokens_per_block = tl.sum(expert_token_mask)
    tl.store(workspace_ptr + pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n, n_tokens_per_block)


@triton.jit
def _row_id_map_pass_2_kernel(
    # pointers
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    WORKSPACE_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    chunk_idx = pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_id_within_token_block = tl.load(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        mask=(offset < num_tokens),
        other=0,
    )

    workspace_off = tl.arange(0, WORKSPACE_LOAD_WIDTH)
    n_tokens_per_chunk = tl.load(workspace_ptr + workspace_off, mask=workspace_off < chunk_idx)
    row_id = tl.where(
        row_id_within_token_block == 0,
        -1,
        row_id_within_token_block + tl.sum(n_tokens_per_chunk) - 1,
    )
    tl.store(
        row_id_map_ptr + pid_m * stride_row_id_map_expert + offset * stride_row_id_map_token,
        row_id,
        mask=(offset < num_tokens),
    )


@triton.jit
def _row_id_map_pass_3_kernel(
    # pointers
    row_id_map_ptr,
    # sizes
    num_experts: tl.constexpr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    LOAD_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_dims: tl.constexpr = _log2(LOAD_SIZE)
    off = tl.arange(0, LOAD_SIZE)
    row_id_map = tl.load(
        row_id_map_ptr + pid * stride_row_id_map_token + stride_row_id_map_expert * off,
        mask=off < num_experts,
        other=-1,
    )
    n_routed = tl.sum(tl.where(row_id_map != -1, 1, 0))
    indices = off
    sorted_map, indices = _argsort(row_id_map, indices, n_dims=n_dims)
    tl.store(
        row_id_map_ptr + pid * stride_row_id_map_token + off * stride_row_id_map_expert,
        sorted_map,
        mask=off < n_routed,
    )
    tl.store(
        row_id_map_ptr
        + pid * stride_row_id_map_token
        + (num_experts + off) * stride_row_id_map_expert,
        indices,
        mask=off < n_routed,
    )
    tl.store(
        row_id_map_ptr + pid * stride_row_id_map_token + num_experts * 2 * stride_row_id_map_expert,
        n_routed,
    )


@triton.jit
def _permute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    scale_ptr,
    permuted_probs_ptr,
    permuted_scale_ptr,
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    scale_hidden_dim,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_probs_expert,
    stride_scale_token,
    stride_scale_hidden,
    stride_permuted_probs_token,
    stride_permuted_scale_token,
    stride_permuted_scale_hidden,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    PERMUTE_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    cur_off = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cur_off < hidden_size
    src_row = pid_t.to(tl.int64)
    input_off = src_row * stride_input_token + cur_off * stride_input_hidden
    inp = tl.load(input_ptr + input_off, mask=mask)
    if PERMUTE_SCALE:
        mask_scale = cur_off < scale_hidden_dim
        scale_off = pid_t * stride_scale_token + cur_off * stride_scale_hidden
        scale = tl.load(scale_ptr + scale_off, mask=mask_scale)
    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr + pid_t * stride_row_id_map_token + idx * stride_row_id_map_expert
        ).to(tl.int64)
        output_off = dst_row * stride_output_token + cur_off * stride_output_hidden
        if PERMUTE_SCALE:
            permuted_scale_off = (
                dst_row * stride_permuted_scale_token + cur_off * stride_permuted_scale_hidden
            )
            tl.store(permuted_scale_ptr + permuted_scale_off, scale, mask=mask_scale)
        if PERMUTE_PROBS:
            expert_idx = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + (num_experts + idx) * stride_row_id_map_expert
            )
            prob_off = pid_t * stride_probs_token + expert_idx * stride_probs_expert
            prob = tl.load(probs_ptr + prob_off)
            if pid_h == 0:
                permuted_prob_off = dst_row * stride_permuted_probs_token
                tl.store(permuted_probs_ptr + permuted_prob_off, prob)
            if prob == 0.0:
                # for routing_map padding
                # dst_row != -1 and prob == 0.0 means that this slot is padded
                tl.store(output_ptr + output_off, 0.0, mask=mask)
            else:
                tl.store(output_ptr + output_off, inp, mask=mask)
        else:
            tl.store(output_ptr + output_off, inp, mask=mask)


try:
    _permute_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_permute_kernel)
except RuntimeError:
    pass


@triton.jit
def _unpermute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    merging_probs_ptr,
    permuted_probs_ptr,
    unpermuted_probs_ptr,
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_permuted_probs_token,
    stride_unpermuted_probs_token,
    stride_unpermuted_probs_expert,
    # metas
    PROBS_LOAD_WIDTH: tl.constexpr,
    WITH_MERGING_PROBS: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = input_ptr.dtype.element_ty
    compute_type = tl.float32

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    current_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = current_offset < hidden_size
    if PERMUTE_PROBS:
        # write 0.0 to probs_grad that are not routed
        if pid_h == 0:
            map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
            unpermuted_prob_off = (
                pid_t * stride_unpermuted_probs_token
                + stride_unpermuted_probs_expert * map_load_off
            )
            tl.store(
                unpermuted_probs_ptr + unpermuted_prob_off, 0.0, mask=map_load_off < num_experts
            )
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        src_row = tl.load(
            row_id_map_ptr + pid_t * stride_row_id_map_token + idx * stride_row_id_map_expert
        ).to(tl.int64)
        input_off = src_row * stride_input_token + current_offset * stride_input_hidden
        inp = tl.load(input_ptr + input_off, mask=mask)
        inp = inp.to(compute_type)
        if WITH_MERGING_PROBS:
            expert_idx = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + (num_experts + idx) * stride_row_id_map_expert
            )
            merging_prob_off = (
                pid_t * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
            )
            merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
            inp *= merging_prob
        accumulator += inp
        if PERMUTE_PROBS:
            if pid_h == 0:
                expert_idx = tl.load(
                    row_id_map_ptr
                    + pid_t * stride_row_id_map_token
                    + (num_experts + idx) * stride_row_id_map_expert
                )
                unpermuted_prob_off = (
                    pid_t * stride_unpermuted_probs_token
                    + expert_idx * stride_unpermuted_probs_expert
                )
                permuted_prob_off = src_row * stride_permuted_probs_token
                prob = tl.load(permuted_probs_ptr + permuted_prob_off)
                tl.store(unpermuted_probs_ptr + unpermuted_prob_off, prob)
    accumulator = accumulator.to(data_type)
    dst_row = pid_t.to(tl.int64)
    output_off = dst_row * stride_output_token + current_offset * stride_output_hidden
    tl.store(output_ptr + output_off, accumulator, mask=mask)


try:
    _unpermute_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_unpermute_kernel)
except RuntimeError:
    pass


@triton.jit
def _unpermute_bwd_with_merging_probs_kernel(
    # pointers
    fwd_output_grad_ptr,
    fwd_input_grad_ptr,
    fwd_input_ptr,
    merging_probs_ptr,
    merging_probs_grad_ptr,
    row_id_map_ptr,
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_fwd_input_token,
    stride_fwd_input_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_merging_probs_grad_token,
    stride_merging_probs_grad_expert,
    # metas
    PROBS_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)
    map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
    token_probs_grad_off = (
        pid * stride_merging_probs_grad_token + stride_merging_probs_grad_expert * map_load_off
    )
    tl.store(merging_probs_grad_ptr + token_probs_grad_off, 0.0, mask=map_load_off < num_experts)
    n_routed = tl.load(
        row_id_map_ptr + pid * stride_row_id_map_token + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr + pid * stride_row_id_map_token + idx * stride_row_id_map_expert
        ).to(tl.int64)
        expert_idx = tl.load(
            row_id_map_ptr
            + pid * stride_row_id_map_token
            + (num_experts + idx) * stride_row_id_map_expert
        )
        prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        current_start = 0
        while current_start < hidden_size:
            current_offset = current_start + tl.arange(0, BLOCK_SIZE)
            mask = current_offset < hidden_size
            src_row = pid.to(tl.int64)
            input_off = (
                src_row * stride_fwd_output_grad_token
                + current_offset * stride_fwd_output_grad_hidden
            )
            inp = tl.load(fwd_output_grad_ptr + input_off, mask=mask)
            inp = inp.to(compute_type)
            merging_prob_off = (
                pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
            )
            merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
            output = inp * merging_prob
            output = output.to(data_type)
            output_off = (
                dst_row * stride_fwd_input_grad_token
                + current_offset * stride_fwd_input_grad_hidden
            )
            tl.store(fwd_input_grad_ptr + output_off, output, mask=mask)

            fwd_input_off = (
                dst_row * stride_fwd_input_token + current_offset * stride_fwd_input_hidden
            )
            fwd_input = tl.load(fwd_input_ptr + fwd_input_off, mask=mask)
            prob_grad_accum += fwd_input.to(compute_type) * inp
            current_start += BLOCK_SIZE
        probs_grad = tl.sum(prob_grad_accum).to(merging_probs_grad_ptr.dtype.element_ty)
        probs_grad_off = (
            pid * stride_merging_probs_grad_token + expert_idx * stride_merging_probs_grad_expert
        )
        tl.store(merging_probs_grad_ptr + probs_grad_off, probs_grad)


try:
    _unpermute_bwd_with_merging_probs_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_unpermute_bwd_with_merging_probs_kernel)
except RuntimeError:
    pass


@triton.jit
def _make_chunk_sort_map_kernel(
    # pointers
    split_sizes_ptr,
    sorted_indices_ptr,
    dst_rows_ptr,
    # sizes
    num_splits: tl.constexpr,
    # metas
    IDX_LOAD_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)

    load_split_offset = tl.arange(0, IDX_LOAD_WIDTH)
    sorted_indices = tl.load(
        sorted_indices_ptr + load_split_offset, mask=load_split_offset < num_splits
    )

    # get chunk idx of the current token in the input tensor
    input_split_sizes = tl.load(
        split_sizes_ptr + load_split_offset, mask=load_split_offset < num_splits, other=0
    ).to(tl.int32)
    input_split_sizes_cumsum = tl.cumsum(input_split_sizes)
    input_split_sizes_mask = tl.where(input_split_sizes_cumsum <= pid, 1, 0)
    input_chunk_idx = tl.sum(input_split_sizes_mask)
    input_split_sizes_presum = tl.sum(input_split_sizes * input_split_sizes_mask)
    in_chunk_offset = pid - input_split_sizes_presum

    # get chunk idx of the current token in the output tensor
    output_chunk_mask = tl.where(sorted_indices == input_chunk_idx, 1, 0)
    output_chunk_idx = tl.argmax(output_chunk_mask, axis=-1)

    # make row_id_map
    output_split_sizes = tl.load(
        split_sizes_ptr + sorted_indices, mask=load_split_offset < num_splits
    ).to(tl.int32)
    output_pre_split_sizes = tl.where(load_split_offset < output_chunk_idx, output_split_sizes, 0)
    dst_row = tl.sum(output_pre_split_sizes) + in_chunk_offset
    tl.store(dst_rows_ptr + pid, dst_row)


@triton.jit
def _sort_chunks_by_map_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    hidden_size: tl.constexpr,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FORWARD: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if FORWARD:
        src_row = pid_t.to(tl.int64)
        dst_row = tl.load(row_id_map_ptr + pid_t).to(tl.int64)
    else:
        src_row = tl.load(row_id_map_ptr + pid_t).to(tl.int64)
        dst_row = pid_t.to(tl.int64)
    current_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = current_offset < hidden_size
    input_offsets = src_row * stride_input_token + current_offset * stride_input_hidden
    output_offsets = dst_row * stride_output_token + current_offset * stride_output_hidden
    inp = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + output_offsets, inp, mask=mask)
    if PERMUTE_PROBS:
        if pid_h == 0:
            prob_off = src_row * stride_probs_token
            prob = tl.load(probs_ptr + prob_off)
            permuted_prob_off = dst_row * stride_permuted_probs_token
            tl.store(permuted_probs_ptr + permuted_prob_off, prob)


try:
    _sort_chunks_by_map_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_sort_chunks_by_map_kernel)
except RuntimeError:
    pass
