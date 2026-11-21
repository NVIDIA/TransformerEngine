# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped weight gradient chained into the overlapped backward (SM 10.0, 10.3 and 10.7).

``weight_grad[e] = grad[rows of e]^T x[rows of e]`` per expert: ``grad`` [rows, m] and ``x``
[rows, n] BF16, or MXFP8 columnwise data with their GEMM-swizzled columnwise scales,
``weight_grad`` [experts, m, n] BF16 or FP32, written or added to. Operands: ``expert_row_ends``
[experts] int32, ``workspace`` uint8, stream; ``signal`` [rows / slot rows] int32 with
``ready_slot_rows`` (BF16 only).

Each launch is a persistent grid that claims output tiles from a counter in ``workspace``, so
launches sharing one workspace split the tiles. It is a programmatic dependent of the kernel
before it and lets the kernel behind it start early. MXFP8 launches of an added or FP32 weight
gradient, and on SM 10.7 of a written BF16 one with long experts, read per-expert descriptors
that a ``descriptors_only`` launch writes into ``workspace`` ahead of them; the others read global
descriptors.
"""

import functools
from dataclasses import dataclass, field

import cutlass
from cutlass import cute

from ._loader import (
    compile_launch,
    device_compute_capability,
    CLUSTER_MN,
    DELAYED_TENSOR_SCALING,
    MMA_TILER_MN,
    MXFP8_1D_SCALING,
    SF_VEC_SIZE,
    fake_tensor,
    fp8_fake,
    import_gemm_modules,
    is_sm107,
    register_function,
    scale_fake,
)


@dataclass(frozen=True)
class WgradConfig:
    """Static configuration of one compiled weight-gradient launch."""

    rows: int  # receive capacity in rows
    m: int  # columns of grad, rows of each expert's weight gradient
    n: int  # columns of x, columns of each expert's weight gradient
    num_experts: int  # local experts
    num_clusters: int  # persistent clusters
    fp32_out: bool = False  # FP32 instead of BF16 weight gradient
    accumulate: bool = False  # add to the weight gradient instead of writing it
    wait_primary_at_start: bool = False  # wait for the kernel before it ahead of any load
    ready_slot_rows: int = 0  # wait until signal[] is set for every slot of these rows
    scaling_mode: int = DELAYED_TENSOR_SCALING  # operand scaling (NVTEScalingMode)
    descriptors_only: bool = False  # MXFP8: write the per-expert descriptors, no GEMM
    rows_alignment: int = 0  # rows per expert slab are a multiple of it
    routed_rows: int = 0  # rows a rank receives on average
    # Kernels, workspace layouts and compiled functions are specific to the device architecture.
    capability: tuple = field(default_factory=device_compute_capability)

    @property
    def mxfp8(self) -> bool:
        """Whether the operands are MXFP8 rather than BF16."""
        return self.scaling_mode == MXFP8_1D_SCALING


def _mxfp8_kernels():
    """(per-expert descriptor, reuse, dense or None, moe_utils) MXFP8 kernel classes of the
    device. SM 10.7 has no dense one."""
    if is_sm107():
        stock, reuse, moe_utils = import_gemm_modules(
            "moe_blockscaled_grouped_gemm_wgrad_rubin",
            "moe_blockscaled_grouped_gemm_wgrad_reuse_rubin",
        )
        return (
            stock.BlockScaledMoEGroupedGemmWgradRubinKernel,
            reuse.BlockScaledMoEGroupedGemmWgradReuseRubinKernel,
            None,
            moe_utils,
        )
    blockscaled, dense, reuse, moe_utils = import_gemm_modules(
        "moe_blockscaled_grouped_gemm_wgrad",
        "moe_blockscaled_grouped_gemm_wgrad_dense",
        "moe_blockscaled_grouped_gemm_wgrad_reuse",
    )
    return (
        blockscaled.BlockScaledMoEGroupedGemmWgradKernel,
        reuse.BlockScaledMoEGroupedGemmWgradReuseKernel,
        dense.BlockScaledMoEGroupedGemmWgradDenseKernel,
        moe_utils,
    )


def _global_descriptor_kernel(cfg: WgradConfig):
    """Class of an MXFP8 weight gradient written in BF16 that reads global descriptors: the one
    keeping A across the N tiles of an expert when the experts are short on average, else the
    dense one; None for the others."""
    if cfg.accumulate or cfg.fp32_out:
        return None
    _, reuse, dense, _ = _mxfp8_kernels()
    # Each weight gradient picks for its own shape; EPCompOverlap keeps A only when both can, so
    # with ffn 256, 1280 or 1792 it runs the first one dense too.
    if reuse.can_implement(cfg.m, cfg.n, cfg.rows_alignment, cfg.routed_rows, cfg.num_experts):
        return reuse
    return dense


def reads_descriptors(cfg: WgradConfig) -> bool:
    """Whether the launches of ``cfg`` read the per-expert descriptors of a ``descriptors_only``
    launch."""
    return cfg.mxfp8 and _global_descriptor_kernel(cfg) is None


def build_kernel(cfg: WgradConfig):
    """Kernel for ``cfg``: 256 x 512 output tiles where they divide a BF16 weight gradient that is
    written, else 256 x 256 tiles. An MXFP8 weight gradient written in BF16 keeps A across the N
    tiles of an expert when the experts are short on average, else reads A, B and the scales
    through global descriptors (per-expert ones on SM 10.7)."""
    if cfg.mxfp8:
        stock, _, _, moe_utils = _mxfp8_kernels()
        kernel = _global_descriptor_kernel(cfg)
        if kernel is not None:
            # The SM 10.7 reuse kernel sizes its A ring to the average padded expert.
            padded_rows = cfg.routed_rows + cfg.num_experts * cfg.rows_alignment // 2
            a_stages = (
                {"a_stages": kernel.a_stages_for(padded_rows, cfg.num_experts)}
                if is_sm107()
                else {}
            )
            return kernel(
                sf_vec_size=SF_VEC_SIZE,
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=True,
                mma_tiler_mn=MMA_TILER_MN,
                cluster_shape_mn=CLUSTER_MN,
                expert_cnt=cfg.num_experts,
                weight_mode=moe_utils.MoEWeightMode.DENSE,
                input_order=moe_utils.WGradInputOrder.Tensor2D,
                tile_claim=moe_utils.WgradTileClaim.COUNTER,
                use_pdl=True,
                wait_primary_at_start=cfg.wait_primary_at_start,
                **a_stages,
            )
        if is_sm107() and cfg.rows <= 256:
            # The SM 10.7 kernel faults when the K extent is a single 256-row tile.
            raise ValueError(
                "The SM 10.7 MXFP8 weight gradient that is added, FP32 or of long experts needs "
                "a receive capacity above 256 rows."
            )
        return stock(
            sf_vec_size=SF_VEC_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=MMA_TILER_MN,
            cluster_shape_mn=CLUSTER_MN,
            accumulate_on_output=cfg.accumulate,
            expert_cnt=cfg.num_experts,
            weight_mode=moe_utils.MoEWeightMode.DENSE,
            input_order=moe_utils.WGradInputOrder.Tensor2D,
            tile_claim=moe_utils.WgradTileClaim.COUNTER,
            descriptors_only=cfg.descriptors_only,
            descriptors_prebuilt=not cfg.descriptors_only,
            use_pdl=not cfg.descriptors_only,
            wait_primary_at_start=cfg.wait_primary_at_start,
        )
    if is_sm107():
        # Both kernels, sized for the SM 10.7 shared memory.
        names = ("moe_grouped_gemm_sm107",) * 2
    else:
        names = ("moe_grouped_gemm_wgrad", "moe_grouped_gemm_wgrad_wide")
    stock, wide, moe_utils = import_gemm_modules(*names)
    options = {
        "expert_cnt": cfg.num_experts,
        "tile_claim": moe_utils.WgradTileClaim.COUNTER,
    }
    wide_kernel = wide.MoEGroupedGemmWgradWideBF16Kernel
    if not (cfg.accumulate or cfg.fp32_out) and wide_kernel.can_implement(cfg.m, cfg.n):
        return wide_kernel(
            use_pdl=True,
            wait_primary_at_start=cfg.wait_primary_at_start,
            ready_slot_rows=cfg.ready_slot_rows,
            **options,
        )
    return stock.MoEGroupedGemmWgradBF16Kernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=MMA_TILER_MN,
        cluster_shape_mn=CLUSTER_MN,
        accumulate_on_output=cfg.accumulate,
        weight_mode=moe_utils.MoEWeightMode.DENSE,
        input_order=moe_utils.WGradInputOrder.Tensor2D,
        **options,
    )


@functools.lru_cache(maxsize=None)
def workspace_layout(cfg: WgradConfig):
    """Total bytes and byte offset of the tile counter of ``cfg``'s workspace."""
    kernel = build_kernel(cfg)
    return kernel.get_workspace_bytes(), kernel.get_desc_workspace_bytes()


def _compile(cfg: WgradConfig):
    """Compile the launch wrapper of ``cfg`` with TVM-FFI."""
    kernel = build_kernel(cfg)
    rows, m, n, experts, num_clusters = cfg.rows, cfg.m, cfg.n, cfg.num_experts, cfg.num_clusters

    def views(grad, x, weight_grad):
        # grad^T as (m, rows), x as (rows, n), weight_grad as (experts, m, n).
        return (
            cute.make_tensor(grad.iterator, cute.make_layout((m, rows), stride=(1, m))),
            cute.make_tensor(x.iterator, cute.make_layout((rows, n), stride=(n, 1))),
            cute.make_tensor(
                weight_grad.iterator, cute.make_layout((experts, m, n), stride=(m * n, n, 1))
            ),
        )

    out_fake = fake_tensor(
        cutlass.Float32 if cfg.fp32_out else cutlass.BFloat16, (experts, m, n), (2, 1, 0), 16
    )
    tail = (
        fake_tensor(cutlass.Int32, (experts,), (0,), 16),
        fake_tensor(cutlass.Uint8, (workspace_layout(cfg)[0],), (0,), 128),
    )
    if cfg.mxfp8:
        k_scales = rows // SF_VEC_SIZE

        @cute.jit
        def launch_mxfp8(
            grad, x, grad_scales, x_scales, weight_grad, expert_row_ends, workspace, stream
        ):
            a, b, out = views(grad, x, weight_grad)
            # Columnwise scales as (columns, rows / 32) over the per-expert swizzled atoms.
            sfa = cute.make_tensor(
                grad_scales.iterator, cute.make_layout((m, k_scales), stride=(k_scales, 1))
            )
            sfb = cute.make_tensor(
                x_scales.iterator, cute.make_layout((n, k_scales), stride=(k_scales, 1))
            )
            kernel(a, b, sfa, sfb, out, expert_row_ends, workspace, num_clusters, stream)

        return compile_launch(
            launch_mxfp8,
            fp8_fake((rows, m), (1, 0)),
            fp8_fake((rows, n), (1, 0)),
            scale_fake(rows, m),
            scale_fake(rows, n),
            out_fake,
            *tail,
        )

    fakes = (
        fake_tensor(cutlass.BFloat16, (rows, m), (1, 0), 16),
        fake_tensor(cutlass.BFloat16, (rows, n), (1, 0), 16),
        out_fake,
        *tail,
    )
    if cfg.ready_slot_rows:
        # Only the wide kernel waits on the signal. The stock kernel, chosen for FP32 or
        # accumulated gradients and shapes the wide one cannot implement, is launched without PDL,
        # so it starts after the signaling kernels complete and has no flags to wait for.
        wait_on_signal = getattr(kernel, "ready_slot_rows", 0) > 0

        @cute.jit
        def launch_ready(grad, x, weight_grad, expert_row_ends, workspace, signal, stream):
            a, b, out = views(grad, x, weight_grad)
            if cutlass.const_expr(wait_on_signal):
                kernel(
                    a, b, out, expert_row_ends, workspace, num_clusters, stream, None,
                    signal.iterator,
                )  # fmt: skip
            else:
                kernel(a, b, out, expert_row_ends, workspace, num_clusters, stream)

        return compile_launch(
            launch_ready,
            *fakes,
            fake_tensor(cutlass.Int32, (rows // cfg.ready_slot_rows,), (0,), 4),
        )

    @cute.jit
    def launch(grad, x, weight_grad, expert_row_ends, workspace, stream):
        a, b, out = views(grad, x, weight_grad)
        kernel(a, b, out, expert_row_ends, workspace, num_clusters, stream)

    return compile_launch(launch, *fakes)


def _config(
    rows, m, n, num_experts, num_clusters, fp32_out, accumulate, wait_primary_at_start,
    ready_slot_rows, scaling_mode, descriptors_only, rows_alignment, routed_rows,
):  # fmt: skip
    """Build a WgradConfig from the entry-point arguments."""
    return WgradConfig(
        rows, m, n, num_experts, num_clusters, bool(fp32_out), bool(accumulate),
        bool(wait_primary_at_start), ready_slot_rows, int(scaling_mode), bool(descriptors_only),
        rows_alignment, routed_rows,
    )  # fmt: skip


def get_ep_wgrad_workspace_layout(
    rows, m, n, num_experts, num_clusters, fp32_out, accumulate, wait_primary_at_start,
    ready_slot_rows, scaling_mode, descriptors_only, rows_alignment, routed_rows,
):  # fmt: skip
    """[total bytes, counter offset] of the workspace of this configuration."""
    return list(workspace_layout(_config(
        rows, m, n, num_experts, num_clusters, fp32_out, accumulate, wait_primary_at_start,
        ready_slot_rows, scaling_mode, descriptors_only, rows_alignment, routed_rows,
    )))  # fmt: skip


def get_ep_wgrad_reads_descriptors(
    rows, m, n, num_experts, num_clusters, fp32_out, accumulate, wait_primary_at_start,
    ready_slot_rows, scaling_mode, descriptors_only, rows_alignment, routed_rows,
) -> bool:  # fmt: skip
    """Whether the launches of this configuration read the per-expert descriptors of a
    ``descriptors_only`` launch."""
    return reads_descriptors(_config(
        rows, m, n, num_experts, num_clusters, fp32_out, accumulate, wait_primary_at_start,
        ready_slot_rows, scaling_mode, descriptors_only, rows_alignment, routed_rows,
    ))  # fmt: skip


def get_ep_wgrad_function(
    fn_name, rows, m, n, num_experts, num_clusters, fp32_out, accumulate, wait_primary_at_start,
    ready_slot_rows, scaling_mode, descriptors_only, rows_alignment, routed_rows,
) -> bool:  # fmt: skip
    """Compile and register the weight gradient of this configuration as ``fn_name``; return
    whether it is."""
    cfg = _config(
        rows, m, n, num_experts, num_clusters, fp32_out, accumulate, wait_primary_at_start,
        ready_slot_rows, scaling_mode, descriptors_only, rows_alignment, routed_rows,
    )  # fmt: skip
    return register_function(fn_name, lambda: _compile(cfg), "EP weight-gradient grouped GEMM")
