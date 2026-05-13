# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal helpers for the NVTE_TP_INVARIANT_MODE code path.

Gated on ``NVTE_TP_INVARIANT_MODE=1`` and used by ``linear.py`` and
``layernorm_linear.py``. With the env var unset (default), these helpers are
unreachable and stock GEMM paths are taken unchanged.

The TP-invariant path performs the full-K (or full-out) GEMM after all-gathering
sharded operands across the TP group. Result: bit-identical numerics across
TP=1/2/4/... because the underlying GEMM K-dimension accumulation order is fixed
regardless of how the operands were sharded.

Limitations:
- FP8 not supported (callers should assert ``not fp8`` before calling).
- Trades compute for invariance (gathered operands + full GEMM). Off by default.
"""

from typing import Optional

import torch

from ..cpp_extensions import general_gemm
from ..utils import nvtx_range_pop, nvtx_range_push

__all__ = [
    "allgather_along_dim",
    "tp_invariant_row_parallel_gemm",
    "tp_invariant_column_parallel_dgrad",
]


def allgather_along_dim(
    tensor: torch.Tensor,
    group,
    world_size: int,
    dim: int,
) -> torch.Tensor:
    """All-gather ``tensor`` from every rank in ``group`` and concat along ``dim``."""
    chunks = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(chunks, tensor.contiguous(), group=group)
    return torch.cat(chunks, dim=dim)


def tp_invariant_row_parallel_gemm(
    weightmat: torch.Tensor,
    inputmat_total: torch.Tensor,
    bias: Optional[torch.Tensor],
    tp_group,
    tp_size: int,
    sequence_parallel: bool,
    activation_dtype: torch.dtype,
    nvtx_label: str = "tp_invariant_gemm",
) -> torch.Tensor:
    """Row-parallel forward GEMM with TP-invariant numerics.

    All-gathers input + weight along the contracted (K) dim, runs the full GEMM
    (matching TP=1 accumulation order), then scatters along the sequence dim
    when ``sequence_parallel=True``.

    Args:
        weightmat: Local weight shard of shape ``[out, hidden/TP]``.
        inputmat_total: Local input shard of shape ``[..., hidden/TP]``.
        bias: Optional bias of shape ``[out]``.
        tp_group: TP process group.
        tp_size: Size of the TP group.
        sequence_parallel: If True, scatter the full output along dim 0.
        activation_dtype: Output dtype (matches stock GEMM behavior).
        nvtx_label: NVTX range label.

    Returns:
        Output of shape ``[..., out]`` (full sequence) or ``[.../TP, out]``
        (sequence-scattered when ``sequence_parallel=True``).
    """
    nvtx_range_push(nvtx_label)

    inputmat_gathered = allgather_along_dim(inputmat_total, tp_group, tp_size, dim=-1)
    weight_gathered = allgather_along_dim(weightmat, tp_group, tp_size, dim=-1)

    input_2d = inputmat_gathered.reshape(-1, inputmat_gathered.shape[-1])
    out = general_gemm(
        weight_gathered,
        input_2d,
        out_dtype=activation_dtype,
        bias=bias,
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(inputmat_gathered.shape[:-1] + (weight_gathered.shape[0],))

    if sequence_parallel:
        rank = torch.distributed.get_rank(tp_group)
        out = out.chunk(tp_size, dim=0)[rank].contiguous()

    nvtx_range_pop(nvtx_label)
    return out


def tp_invariant_column_parallel_dgrad(
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    tp_group,
    tp_size: int,
    sequence_parallel: bool,
    activation_dtype: torch.dtype,
    partition_stride: int = 1,
    nvtx_label: str = "tp_invariant_dgrad",
) -> torch.Tensor:
    """Column-parallel backward dgrad with TP-invariant numerics.

    All-gathers grad_output (along out dim) and weight (along out dim), runs the
    full dgrad GEMM, then scatters along the sequence dim under SP.

    For gated MLPs (e.g. SwiGLU FC1 where each rank holds interleaved [gate|val]
    halves), ``partition_stride > 1`` triggers a deinterleave step to recover
    the TP=1 layout [gate_all | val_all] before the full GEMM:

      Per-rank layout:    [g_0 | v_0 | g_1 | v_1 | ... | g_{TP-1} | v_{TP-1}]
      TP=1 native layout: [g_0 | g_1 | ... | g_{TP-1} | v_0 | v_1 | ... | v_{TP-1}]

    For non-gated layers (QKV etc., partition_stride=1) the naive all-gather
    already matches the TP=1 ordering.

    Args:
        weight: Local weight shard of shape ``[out/TP, in]``.
        grad_output: Local grad shard of shape ``[..., out/TP]``.
        tp_group: TP process group.
        tp_size: Size of TP group.
        sequence_parallel: If True, scatter dgrad along dim 0.
        activation_dtype: Output dtype.
        partition_stride: >1 triggers the deinterleave for gated MLPs.
        nvtx_label: NVTX range label.

    Returns:
        dgrad of shape ``[..., in]`` or sequence-scattered.
    """
    nvtx_range_push(nvtx_label)

    grad_output_gathered = allgather_along_dim(grad_output, tp_group, tp_size, dim=-1)
    weight_gathered = allgather_along_dim(weight, tp_group, tp_size, dim=0)

    if partition_stride > 1:
        # Deinterleave gated [gate|val] halves to TP=1 [gate_all | val_all].
        # Currently only the 2-way gated split (SwiGLU FC1 layout) is handled.
        assert (
            partition_stride == 2
        ), f"deinterleave only supports partition_stride=2 (gated halve); got {partition_stride}"
        chunk_sz = weight.shape[0]  # out_features per rank
        half = chunk_sz // 2
        first_w = [weight_gathered[i * chunk_sz : i * chunk_sz + half] for i in range(tp_size)]
        second_w = [
            weight_gathered[i * chunk_sz + half : (i + 1) * chunk_sz] for i in range(tp_size)
        ]
        weight_gathered = torch.cat(first_w + second_w, dim=0)

        g_dim = grad_output_gathered.shape[-1] // tp_size
        g_half = g_dim // 2
        first_g = [
            grad_output_gathered[..., i * g_dim : i * g_dim + g_half] for i in range(tp_size)
        ]
        second_g = [
            grad_output_gathered[..., i * g_dim + g_half : (i + 1) * g_dim] for i in range(tp_size)
        ]
        grad_output_gathered = torch.cat(first_g + second_g, dim=-1)

    grad_output_2d = grad_output_gathered.reshape(
        -1,
        grad_output_gathered.shape[-1],
    )
    dgrad = general_gemm(
        weight_gathered,
        grad_output_2d,
        layout="NN",
        grad=True,
        out_dtype=activation_dtype,
    )
    if isinstance(dgrad, tuple):
        dgrad = dgrad[0]

    if sequence_parallel:
        rank = torch.distributed.get_rank(tp_group)
        dgrad = dgrad.chunk(tp_size, dim=0)[rank].contiguous()

    nvtx_range_pop(nvtx_label)
    return dgrad
