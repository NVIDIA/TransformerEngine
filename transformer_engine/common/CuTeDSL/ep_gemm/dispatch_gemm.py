# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Signal-aware grouped GEMMs that follow an EP dispatch (BF16 or MXFP8, SM 10.0, 10.3 and 10.7).

An M tile starts once each of its slots s has ``signal[s] == s``. Operands (rows = receive rows):

    mode           a        b (weight)       outputs
    fwd BF16       [rows,k] [E,n,k]          c [rows,n]; d [rows,n/2] = silu(gate)*up*prob
    bwd BF16       [rows,k] [E,k,n]          d [rows,2n] = grad of c; dprob [rows] FP32 (added to)
    fwd MXFP8      E4M3     E4M3             c BF16 [rows,n]; d E4M3 [rows,n/2] + swizzled sfd
    bwd MXFP8      E4M3     columnwise E4M3  d rowwise + d_col columnwise, h [rows,n] columnwise

Forward is FC1 (gate and up alternate in 32-column blocks of ``c``). Backward is the FC2 data
gradient with the SwiGLU backward in the epilogue; its ``c`` is the FC1 output [rows, 2n].
Common operands: ``expert_row_ends`` [E] int32, ``alpha`` [E] FP32, ``prob`` [rows] FP32,
``workspace`` uint8, stream. With ``device_layout`` the forward also waits for the dispatch to
write ``expert_row_ends`` and the word before it (expert 0's offset), both -1 beforehand. The
kernel tree must wait until they are at least 0 (EPC fc1db0b8 or later).

MXFP8 scales: ``sfa`` is compact (as the dispatch delivers it) and ``sfb`` swizzled. The forward
takes no ``prob`` and a ``norm_const`` [1] FP32 of 1. The backward writes ``sfd_row`` (swizzled),
``sfd_col`` and ``sfh_col`` (swizzled per expert).
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
    device_smem_optin,
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

# Activation of the FC1 GLU to the activation function of its backward epilogue.
_BACKWARD_ACTIVATIONS = {"swiglu": "dswiglu"}
# A/B stages of the SM 10.7 forward GEMMs, BF16 and MXFP8 with compact scales read by TMA.
_RUBIN_FC1_AB_STAGES = 8


@dataclass(frozen=True)
class DispatchGemmConfig:
    """Static configuration of one compiled dispatch grouped GEMM."""

    rows: int  # receive capacity in rows, a multiple of slot_rows
    k: int  # reduction size
    n: int  # GEMM columns: forward 2 * ffn (gate and up), backward ffn
    num_experts: int  # local experts
    num_clusters: int  # persistent clusters beside the dispatch's SMs
    backward: bool = False
    mxfp8: bool = False
    rows_alignment: int = ROWS_PER_TILE  # rows per expert slab are a multiple of it
    device_layout: bool = False  # forward: the dispatch writes the expert row ends
    activation: str = "swiglu"
    # Kernels, workspace layouts and compiled functions are specific to the device architecture.
    capability: tuple = field(default_factory=device_compute_capability)

    def __post_init__(self):
        if self.activation not in _BACKWARD_ACTIVATIONS:
            raise NotImplementedError(
                f"Activation {self.activation!r} is not supported by the dispatch grouped GEMMs;"
                f" supported: {sorted(_BACKWARD_ACTIVATIONS)}."
            )

    @property
    def slot_rows(self) -> int:
        """Rows per ``signal[]`` entry: the expert alignment, at most one row tile."""
        return min(self.rows_alignment, ROWS_PER_TILE)

    @property
    def signal_capacity(self) -> int:
        """Entries of ``signal[]``, one per slot of the capacity."""
        return self.rows // self.slot_rows


def _tail_trigger(kernel_class) -> dict:
    """PDL trigger of a forward kernel for its tail on the dispatch's SMs: at slots ready or at
    entry, as the kernel class prefers."""
    if kernel_class.FC1_TAIL_AT_SLOTS_READY:
        return {"launch_dependents_at_slots_ready": True}
    return {"launch_dependents_at_entry": True}


def build_kernel(cfg: DispatchGemmConfig):
    """Kernel for ``cfg``; row tiles wait on ``signal[]``. A forward one lets its tail launch at the
    ``_tail_trigger`` point. The backward one lets the kernel behind it start at entry."""
    backward_activation = _BACKWARD_ACTIVATIONS[cfg.activation]
    rubin = is_sm107()
    if cfg.mxfp8:
        # The TMA warp turns the received compact scales into the GEMM's layout per tile.
        common = {
            **blockscaled_kwargs(cfg.num_experts),
            "signal_capacity": cfg.signal_capacity,
            "compact_sfa_sf_warp": True,
            "use_pdl": True,
            **mxfp8_tail_tile_kwargs(cfg.rows_alignment),
        }
        if cfg.backward and rubin:
            # 128-deep K tiles of one scale atom each: the TMA warp queues the compact scales ahead.
            module = import_gemm_modules("moe_blockscaled_grouped_gemm_dglu_rubin")[0]
            return module.BlockScaledMoEGroupedGemmDgluKernel(
                discrete_col_sfd=True,
                act_func=backward_activation,
                fp8_k_tile=128,
                compact_sfa_queue=True,
                **common,
            )
        if cfg.backward:
            module = import_gemm_modules("moe_blockscaled_grouped_gemm_dglu_dbias")[0]
            return module.BlockScaledMoEGroupedGemmDgluDbiasKernel(
                discrete_col_sfd=True,
                act_func=backward_activation,
                **common,
            )
        if rubin:
            # The fused SwiGLU GEMM. Compact scale rows of whole TMA boxes load by TMA, others
            # through registers with the opt-in shared memory.
            module = import_gemm_modules("moe_blockscaled_grouped_gemm_glu_rubin")[0]
            kernel_class = module.BlockScaledMoEGroupedGemmGluKernel
            scale_rows_tma = cfg.k % 1024 == 0
            return kernel_class(
                generate_sfd=True,
                discrete_col_sfd=False,
                act_func=cfg.activation,
                enable_bias=False,
                generate_c=True,
                generate_sfd_col=False,
                smem_capacity=None if scale_rows_tma else device_smem_optin(),
                device_layout=cfg.device_layout,
                max_ab_stages=_RUBIN_FC1_AB_STAGES if scale_rows_tma else None,
                compact_sfa_rows=scale_rows_tma,
                **_tail_trigger(kernel_class),
                **{**common, "vectorized_f32": True},
            )
        module = import_gemm_modules("moe_blockscaled_grouped_gemm_quant_sm100")[0]
        kernel_class = module.BlockScaledMoEGroupedGemmQuantKernel
        return kernel_class(
            generate_sfd=True,
            discrete_col_sfd=False,
            enable_bias=False,
            fuse_swiglu=True,
            device_layout=cfg.device_layout,
            **_tail_trigger(kernel_class),
            **common,
        )
    common = {
        "acc_dtype": cutlass.Float32,
        "use_2cta_instrs": True,
        "mma_tiler_mn": MMA_TILER_MN,
        "cluster_shape_mn": CLUSTER_MN,
        "vectorized_f32": False,
        "expert_cnt": cfg.num_experts,
        "use_dynamic_sched": True,
        "signal_capacity": cfg.signal_capacity,
        "use_pdl": True,
        **tail_tile_kwargs(cfg.rows_alignment, fused=not cfg.backward),
    }
    if cfg.backward:
        kernel_class, moe_utils = import_bf16_kernel("dispatch_bwd")
        return kernel_class(
            weight_mode=moe_utils.MoEWeightMode.DENSE,
            act_func=backward_activation,
            **common,
        )
    kernel_class, moe_utils = import_bf16_kernel("dispatch_fwd")
    return kernel_class(
        weight_mode=moe_utils.MoEWeightMode.DENSE,
        generate_c=True,
        enable_bias=False,
        device_layout=cfg.device_layout,
        **({"max_ab_stages": _RUBIN_FC1_AB_STAGES} if rubin else {}),
        **_tail_trigger(kernel_class),
        **common,
    )


@functools.lru_cache(maxsize=None)
def workspace_layout(cfg: DispatchGemmConfig):
    """Total bytes and byte offsets of the claim counter, ``signal[]`` and scheduler state."""
    helpers, _ = import_gemm_modules("moe_signal_helpers")
    offsets = helpers.signal_workspace_offsets(build_kernel(cfg))
    return offsets["total_bytes"], offsets["counter"], offsets["signal"], offsets["state"]


def _prologue(cfg: DispatchGemmConfig):
    """Kernel, (rows, k, n, experts, num_clusters) and workspace tail fakes shared by a compiler."""
    kernel = build_kernel(cfg)
    dims = (cfg.rows, cfg.k, cfg.n, cfg.num_experts, cfg.num_clusters)
    tail = tail_fakes(cfg.rows, cfg.num_experts, workspace_layout(cfg)[0])
    return kernel, dims, tail


def _compile_forward(cfg: DispatchGemmConfig):
    """FC1: SwiGLU GEMM with the weight [experts, n, k], writing ``c`` and ``d``."""
    kernel, (rows, k, n, experts, num_clusters), tail = _prologue(cfg)
    d_cols = n // 2

    @cute.jit
    def launch(a, b, c, d, expert_row_ends, alpha, prob, workspace, stream):
        weight = weight_view(b, n, k, experts, n_major=False)
        kernel(
            rows_view(a, rows, k), weight, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int64(0),
            OperandMajorMode.K, workspace.iterator, rows_view(c, rows, n),
            rows_view(d, rows, d_cols), expert_row_ends, alpha, prob_view(prob, rows), None,
            num_clusters, stream,
        )  # fmt: skip

    return compile_launch(
        launch,
        fake_tensor(cutlass.BFloat16, (rows, k), (1, 0), 16),
        fake_tensor(cutlass.BFloat16, (experts, n, k), (2, 1, 0), 16),
        fake_tensor(cutlass.BFloat16, (rows, n), (1, 0), 16),
        fake_tensor(cutlass.BFloat16, (rows, d_cols), (1, 0), 16),
        *tail,
    )


def _compile_backward(cfg: DispatchGemmConfig):
    """FC2 data gradient with the forward weight [experts, k, n] read N-major, and SwiGLU backward
    in the epilogue (unit alpha and beta)."""
    kernel, (rows, k, n, experts, num_clusters), tail = _prologue(cfg)

    @cute.jit
    def launch(a, b, c, d, dprob, expert_row_ends, alpha, prob, workspace, stream):
        weight = weight_view(b, n, k, experts, n_major=True)
        kernel(
            rows_view(a, rows, k), weight, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int64(0),
            OperandMajorMode.K, workspace.iterator, rows_view(c, rows, 2 * n),
            rows_view(d, rows, 2 * n), expert_row_ends, alpha, alpha, prob_view(prob, rows),
            prob_view(dprob, rows), cutlass.Float32(0.0), None, num_clusters, stream,
        )  # fmt: skip

    return compile_launch(
        launch,
        fake_tensor(cutlass.BFloat16, (rows, k), (1, 0), 16),
        fake_tensor(cutlass.BFloat16, (experts, k, n), (2, 1, 0), 16),
        fake_tensor(cutlass.BFloat16, (rows, 2 * n), (1, 0), 16),
        fake_tensor(cutlass.BFloat16, (rows, 2 * n), (1, 0), 16),
        fake_tensor(cutlass.Float32, (rows,), (0,), 16),
        *tail,
    )


def _compile_forward_mxfp8(cfg: DispatchGemmConfig):
    """MXFP8 FC1: SwiGLU GEMM writing the BF16 ``c`` and the E4M3 ``d`` with its scales."""
    kernel, (rows, k, n, experts, num_clusters), tail = _prologue(cfg)
    d_cols = n // 2

    @cute.jit
    def launch(a, b, sfb, sfa, c, d, sfd, expert_row_ends, alpha, norm_const, workspace, stream):
        weight = weight_view(b, n, k, experts, n_major=False)
        d3 = rows_view(d, rows, d_cols)
        kernel(
            rows_view(a, rows, k), weight, sfb, cutlass.Int32(0), cutlass.Int32(0),
            cutlass.Int64(0), OperandMajorMode.K, workspace.iterator, d3, d3, sfa, sfd, None, None,
            norm_const, expert_row_ends, alpha, None, None, None, rows_view(c, rows, n),
            num_clusters, stream,
        )  # fmt: skip

    @cute.jit
    def launch_rubin(
        a, b, sfb, sfa, c, d, sfd, expert_row_ends, alpha, norm_const, workspace, stream
    ):
        # The SM 10.7 kernel takes c ahead of d and no row scale.
        weight = weight_view(b, n, k, experts, n_major=False)
        d3 = rows_view(d, rows, d_cols)
        kernel(
            rows_view(a, rows, k), weight, sfb, cutlass.Int32(0), cutlass.Int32(0),
            cutlass.Int64(0), OperandMajorMode.K, workspace.iterator, rows_view(c, rows, n), d3,
            d3, sfa, sfd, None, None, norm_const, expert_row_ends, alpha, None, None, num_clusters,
            stream,
        )  # fmt: skip

    row_ends, alpha, _, workspace = tail
    return compile_launch(
        launch_rubin if is_sm107() else launch,
        fp8_fake((rows, k), (1, 0)),
        fp8_fake((experts, n, k), (2, 1, 0)),
        scale_fake(experts * n, k),
        scale_fake(rows, k),
        fake_tensor(cutlass.BFloat16, (rows, n), (1, 0), 16),
        fp8_fake((rows, d_cols), (1, 0)),
        scale_fake(rows, d_cols),
        row_ends,
        alpha,
        fake_tensor(cutlass.Float32, (1,), (0,), 4),
        workspace,
    )


def _compile_backward_mxfp8(cfg: DispatchGemmConfig):
    """MXFP8 FC2 data gradient with the columnwise forward weight [experts, k, n] read N-major,
    SwiGLU backward in the epilogue, and the forward activation recomputed columnwise."""
    kernel, (rows, k, n, experts, num_clusters), tail = _prologue(cfg)
    # The SM 10.7 kernel takes the GeGLU linear offset (unused by SwiGLU) ahead of the bias.
    rubin = is_sm107()

    @cute.jit
    def launch(
        a, b, sfb, sfa, c, d, d_col, sfd_row, sfd_col, h, sfh_col, dprob, expert_row_ends, alpha,
        prob, norm_const, workspace, stream,
    ):  # fmt: skip
        weight = weight_view(b, n, k, experts, n_major=True)
        kernel(
            rows_view(a, rows, k), weight, sfb, cutlass.Int32(0), cutlass.Int32(0),
            cutlass.Int64(0), OperandMajorMode.K, workspace.iterator, rows_view(c, rows, 2 * n),
            rows_view(d, rows, 2 * n), rows_view(d_col, rows, 2 * n), sfa, sfd_row, sfd_col, None,
            norm_const, expert_row_ends, alpha, alpha, prob_view(prob, rows),
            prob_view(dprob, rows), *((cutlass.Float32(1.0),) if rubin else ()), None,
            rows_view(h, rows, n), sfh_col, num_clusters, stream,
        )  # fmt: skip

    row_ends, alpha, prob, workspace = tail
    return compile_launch(
        launch,
        fp8_fake((rows, k), (1, 0)),
        fp8_fake((experts, k, n), (2, 1, 0)),
        scale_fake(experts * n, k),
        scale_fake(rows, k),
        fake_tensor(cutlass.BFloat16, (rows, 2 * n), (1, 0), 16),
        fp8_fake((rows, 2 * n), (1, 0)),
        fp8_fake((rows, 2 * n), (1, 0)),
        scale_fake(rows, 2 * n),
        scale_fake(rows, 2 * n),
        fp8_fake((rows, n), (1, 0)),
        scale_fake(rows, n),
        fake_tensor(cutlass.Float32, (rows,), (0,), 16),
        row_ends,
        alpha,
        prob,
        fake_tensor(cutlass.Float32, (1,), (0,), 4),
        workspace,
    )


_COMPILERS = {
    (False, False): _compile_forward,
    (True, False): _compile_backward,
    (False, True): _compile_forward_mxfp8,
    (True, True): _compile_backward_mxfp8,
}


def _config(rows, k, n, num_experts, num_clusters, backward, mxfp8, rows_alignment, device_layout):
    """Build a DispatchGemmConfig from the entry-point arguments."""
    return DispatchGemmConfig(
        rows, k, n, num_experts, num_clusters, bool(backward), bool(mxfp8), rows_alignment,
        bool(device_layout),
    )  # fmt: skip


def get_ep_dispatch_gemm_workspace_layout(
    rows, k, n, num_experts, num_clusters, backward, mxfp8=False, rows_alignment=ROWS_PER_TILE,
    device_layout=False,
):  # fmt: skip
    """[total bytes, counter, signal, state] byte offsets of the workspace of this configuration."""
    return list(workspace_layout(_config(
        rows, k, n, num_experts, num_clusters, backward, mxfp8, rows_alignment, device_layout,
    )))  # fmt: skip


def get_ep_dispatch_gemm_function(
    fn_name, rows, k, n, num_experts, num_clusters, backward, mxfp8=False,
    rows_alignment=ROWS_PER_TILE, device_layout=False,
) -> bool:  # fmt: skip
    """Compile and register the GEMM of this configuration as ``fn_name``; return whether it is."""
    cfg = _config(
        rows, k, n, num_experts, num_clusters, backward, mxfp8, rows_alignment, device_layout,
    )  # fmt: skip
    compile_fn = _COMPILERS[cfg.backward, cfg.mxfp8]
    return register_function(fn_name, lambda: compile_fn(cfg), "EP dispatch grouped GEMM")
