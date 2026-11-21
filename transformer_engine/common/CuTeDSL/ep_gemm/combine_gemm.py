# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped GEMM that publishes its output to the EP combine (BF16 or MXFP8, SM 10.0, 10.3 and
10.7).

    mode        a         b (weight)                   d
    forward     [rows,k]  [E,n,k]                      [rows,n] BF16, scaled per row by prob
    backward    [rows,k]  [E,k,n] (the forward weight) [rows,n] BF16, scaled per row by prob

BF16 operands, or E4M3 with GEMM-swizzled E8M0 scales ``sfa`` and ``sfb`` for MXFP8 (the backward
reads the columnwise forward weight). Common operands: ``expert_row_ends`` [E] int32, ``alpha``
[E] FP32, ``prob`` [rows] FP32, ``workspace`` uint8, stream. ``signal[u] = u`` once columns
``[u * n_chunk, (u + 1) * n_chunk)`` of ``d`` are final; the caller sets ``signal[]`` to -1 before
each launch. With ``n_chunk`` 0 the GEMM publishes nothing and runs in the stock tile order; its
workspace ends in one flag word the caller sets.
"""

import functools
from dataclasses import dataclass, field

import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import OperandMajorMode

from ._loader import (
    device_compute_capability,
    CLUSTER_MN,
    MMA_TILER_MN,
    ROWS_PER_TILE,
    blockscaled_kwargs,
    compile_launch,
    fake_tensor,
    fp8_fake,
    import_bf16_kernel,
    import_gemm_modules,
    is_sm107,
    mxfp8_tail_tile_kwargs,
    prob_view,
    register_function,
    rows_view,
    scale_fake,
    tail_fakes,
    tail_tile_kwargs,
    weight_view,
)


@dataclass(frozen=True)
class CombineGemmConfig:
    """Static configuration of one compiled EP grouped GEMM combine."""

    rows: int  # receive capacity in rows
    k: int  # reduction size (ffn)
    n: int  # output columns (hidden)
    num_experts: int  # local experts
    num_clusters: int  # persistent clusters beside the combine's SMs
    n_chunk: int  # columns per readiness flag, 0: no flags
    dependents_at_entry: bool  # the combine may start at the GEMM's entry (signaled combine only)
    backward: bool = False
    mxfp8: bool = False
    rows_alignment: int = ROWS_PER_TILE  # rows per expert slab are a multiple of it
    grid_clusters: int = 0  # clusters launched (0: num_clusters), extras stop after the first chunk
    # Kernels, workspace layouts and compiled functions are specific to the device architecture.
    capability: tuple = field(default_factory=device_compute_capability)


def build_kernel(cfg: CombineGemmConfig):
    """Kernel for ``cfg`` that publishes finished column chunks to ``signal[]`` (none: n_chunk 0)."""
    comb, moe_utils = import_gemm_modules("moe_streaming_output_chunk_tile_scheduler")
    notify = None
    if cfg.n_chunk:
        notify = comb.OutputNotifyConfig(
            comb.OutputNotifyMode.N_CHUNK,
            cfg.n_chunk,
            comb.output_notify_capacity(cfg.n_chunk, cfg.n),
            caller_reset=True,
            clusters_beside_combine=cfg.num_clusters,
        )
    if cfg.mxfp8:
        rubin = is_sm107()
        module = import_gemm_modules(
            "moe_blockscaled_grouped_gemm_quant_rubin"
            if rubin
            else "moe_blockscaled_grouped_gemm_quant_sm100"
        )[0]
        return module.BlockScaledMoEGroupedGemmQuantKernel(
            generate_sfd=False,
            discrete_col_sfd=False,
            enable_bias=False,
            output_notify=notify,
            launch_dependents_at_entry=cfg.dependents_at_entry,
            **({"generate_c": False} if rubin else {}),
            **blockscaled_kwargs(cfg.num_experts),
            **mxfp8_tail_tile_kwargs(cfg.rows_alignment),
        )
    kernel_class, _ = import_bf16_kernel("combine")
    return kernel_class(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=MMA_TILER_MN,
        cluster_shape_mn=CLUSTER_MN,
        vectorized_f32=False,
        generate_c=False,
        enable_bias=False,
        expert_cnt=cfg.num_experts,
        weight_mode=moe_utils.MoEWeightMode.DENSE,
        use_dynamic_sched=True,
        output_notify=notify,
        launch_dependents_at_entry=cfg.dependents_at_entry,
        **tail_tile_kwargs(cfg.rows_alignment, fused=not cfg.backward),
    )


@functools.lru_cache(maxsize=None)
def workspace_layout(cfg: CombineGemmConfig):
    """(total bytes, byte offset of signal[], entries of signal[]) of ``cfg``'s workspace."""
    comb = import_gemm_modules("moe_streaming_output_chunk_tile_scheduler")[0]
    if not cfg.n_chunk:
        signal = -(-build_kernel(cfg).get_workspace_bytes() // 4) * 4
        return signal + 4, signal, 1
    offsets = comb.output_notify_offsets(build_kernel(cfg))
    return offsets["total_bytes"], offsets["signal"], offsets["capacity"]


def _compile(cfg: CombineGemmConfig):
    """Compile the launch wrapper of ``cfg`` with TVM-FFI."""
    kernel = build_kernel(cfg)
    rows, k, n, experts = cfg.rows, cfg.k, cfg.n, cfg.num_experts
    num_clusters, backward = cfg.grid_clusters or cfg.num_clusters, cfg.backward

    @cute.jit
    def launch(a, b, d, expert_row_ends, alpha, prob, workspace, stream):
        weight = weight_view(b, n, k, experts, n_major=backward)
        d3 = rows_view(d, rows, n)
        kernel(
            rows_view(a, rows, k), weight, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int64(0),
            OperandMajorMode.K, workspace.iterator, d3, d3, expert_row_ends, alpha, None,
            prob_view(prob, rows), num_clusters, stream,
        )  # fmt: skip

    return compile_launch(
        launch,
        fake_tensor(cutlass.BFloat16, (rows, k), (1, 0), 16),
        fake_tensor(
            cutlass.BFloat16, (experts, k, n) if backward else (experts, n, k), (2, 1, 0), 16
        ),
        fake_tensor(cutlass.BFloat16, (rows, n), (1, 0), 16),
        *tail_fakes(rows, experts, workspace_layout(cfg)[0]),
    )


def _compile_mxfp8(cfg: CombineGemmConfig):
    """Compile the MXFP8 launch wrapper of ``cfg`` with TVM-FFI."""
    kernel = build_kernel(cfg)
    rows, k, n, experts = cfg.rows, cfg.k, cfg.n, cfg.num_experts
    num_clusters, backward = cfg.grid_clusters or cfg.num_clusters, cfg.backward

    @cute.jit
    def launch(a, b, sfb, sfa, d, expert_row_ends, alpha, prob, workspace, stream):
        weight = weight_view(b, n, k, experts, n_major=backward)
        d3 = rows_view(d, rows, n)
        kernel(
            rows_view(a, rows, k), weight, sfb, cutlass.Int32(0), cutlass.Int32(0),
            cutlass.Int64(0), OperandMajorMode.K, workspace.iterator, d3, d3, sfa, None, None,
            None, None, expert_row_ends, alpha, None, None, prob_view(prob, rows), None,
            num_clusters, stream,
        )  # fmt: skip

    @cute.jit
    def launch_rubin(a, b, sfb, sfa, d, expert_row_ends, alpha, prob, workspace, stream):
        # The SM 10.7 kernel takes c (not stored) ahead of d, and the bias ahead of prob.
        weight = weight_view(b, n, k, experts, n_major=backward)
        d3 = rows_view(d, rows, n)
        kernel(
            rows_view(a, rows, k), weight, sfb, cutlass.Int32(0), cutlass.Int32(0),
            cutlass.Int64(0), OperandMajorMode.K, workspace.iterator, d3, d3, d3, sfa, None, None,
            None, None, expert_row_ends, alpha, None, prob_view(prob, rows), num_clusters, stream,
        )  # fmt: skip

    return compile_launch(
        launch_rubin if is_sm107() else launch,
        fp8_fake((rows, k), (1, 0)),
        fp8_fake((experts, k, n) if backward else (experts, n, k), (2, 1, 0)),
        scale_fake(experts * n, k),
        scale_fake(rows, k),
        fake_tensor(cutlass.BFloat16, (rows, n), (1, 0), 16),
        *tail_fakes(rows, experts, workspace_layout(cfg)[0]),
    )


def _config(
    rows, k, n, num_experts, num_clusters, n_chunk, dependents_at_entry, backward, mxfp8,
    rows_alignment, grid_clusters,
):  # fmt: skip
    """Build a CombineGemmConfig from the entry-point arguments."""
    return CombineGemmConfig(
        rows, k, n, num_experts, num_clusters, n_chunk, bool(dependents_at_entry), bool(backward),
        bool(mxfp8), rows_alignment, grid_clusters,
    )  # fmt: skip


def get_ep_combine_gemm_workspace_layout(
    rows, k, n, num_experts, num_clusters, n_chunk, dependents_at_entry, backward, mxfp8=False,
    rows_alignment=ROWS_PER_TILE, grid_clusters=0,
):  # fmt: skip
    """[total bytes, signal offset, signal entries] of the workspace of this configuration."""
    return list(workspace_layout(_config(
        rows, k, n, num_experts, num_clusters, n_chunk, dependents_at_entry, backward, mxfp8,
        rows_alignment, grid_clusters,
    )))  # fmt: skip


def get_ep_combine_gemm_function(
    fn_name, rows, k, n, num_experts, num_clusters, n_chunk, dependents_at_entry, backward,
    mxfp8=False, rows_alignment=ROWS_PER_TILE, grid_clusters=0,
) -> bool:  # fmt: skip
    """Compile and register the GEMM of this configuration as ``fn_name``; return whether it is."""
    cfg = _config(
        rows, k, n, num_experts, num_clusters, n_chunk, dependents_at_entry, backward, mxfp8,
        rows_alignment, grid_clusters,
    )  # fmt: skip
    compile_fn = _compile_mxfp8 if cfg.mxfp8 else _compile
    return register_function(fn_name, lambda: compile_fn(cfg), "EP grouped GEMM combine")
