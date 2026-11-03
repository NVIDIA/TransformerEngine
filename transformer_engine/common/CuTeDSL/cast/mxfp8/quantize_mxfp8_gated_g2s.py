# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated-activation MXFP8 casts in CuTeDSL, ported from gated_mxfp8.cuh.

The kernel reuses MXFP8QuantizeGeneralG2SKernel's TMA pipeline and its plain-cast SMEM tile
quantizer. What it adds is an elementwise pass that runs on each staged tile before it is
quantized: it combines the act and gate halves of the input (and the incoming gradient, in
backward) and writes the result back into the act (and gate) input buffer in the input dtype.
That is exactly the truncation the CUDA kernel applies before computing block amaxes, so the
plain cast that follows reproduces its output.

Gated input is [M, 2N] with [act | gate] along the last dim. Every operand is addressed through
(M, N) views (stride 2N, column offset 0 or N), mirroring the CUDA kernel's tensor maps:
  forward : out[M, N] = act(x[:, :N]) * gate(x[:, N:])
  backward: out[M, 2N] = [d/d act | d/d gate] given grad[M, N]
"""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Boolean, Float32, Int32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    fabs_f32,
    is_packed16,
    truncate_f32_bf16,
    truncate_f32_f16,
)
from transformer_engine.common.CuTeDSL.activations import (
    act_clamped_silu,
    act_situ_gate,
    dact_dsitu_gate,
    situ_up,
    dsitu_up,
    sigmoidf,
)
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    GATED_ACTIVATIONS,
    MXFP8GatedQuantizeConfig,
    MXFP8_BLOCK_SCALING_SIZE,
    derive_swizzled_scale_layout,
)
from .quantize_mxfp8_g2s_general import (
    MXFP8QuantizeGeneralG2SKernel,
    quantize_mxfp8_smem_tile_general,
)


@cute.jit
def _select_f32(pred: Boolean, if_true: Float32, if_false: Float32) -> Float32:
    result = if_false
    if pred:
        result = if_true
    return result


@cute.jit
def _gated_element(
    act_elt: Float32,
    gate_elt: Float32,
    grad_elt: Optional[Float32],
    limit: Optional[Float32],
    alpha: Optional[Float32],
    glu_linear_offset: Optional[Float32],
    beta1: Optional[Float32],
    beta2: Optional[Float32],
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
):
    """One element of the gated op in FP32; returns (act output, gate output or None).

    Mirrors the elementwise math of gated_mxfp8.cuh's columnwise pass (forward and backward),
    operation by operation.
    """
    out_gate = None
    # Transform the gate. Gating is ideally the identity, with derivative 1.
    gate = gate_elt
    dgate = None
    if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
        # |g| > limit is gated_mxfp8.cuh's (g > limit || g < -limit), NaN included
        dgate = _select_f32(fabs_f32(gate_elt) > limit, Float32(0.0), Float32(1.0))
        gate = cute.arch.fmin(cute.arch.fmax(-limit, gate_elt), limit) + glu_linear_offset
    elif cutlass.const_expr(ACTIVATION == "situglu"):
        dgate = dsitu_up(gate_elt, beta2)
        gate = situ_up(gate_elt, beta2)

    if cutlass.const_expr(not IS_BWD):
        if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
            act = act_clamped_silu(act_elt, limit, alpha)
        elif cutlass.const_expr(ACTIVATION == "situglu"):
            act = act_situ_gate(act_elt, beta1)
        else:
            act = GATED_ACTIVATIONS[ACTIVATION][0](act_elt)
        out_act = act * gate
    else:
        if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
            x = cute.arch.fmin(act_elt, limit)
            s = sigmoidf(alpha * x)
            act_x = x * s
            dact_x = _select_f32(
                act_elt <= limit, s + s * (Float32(1.0) - s) * alpha * x, Float32(0.0)
            )
        elif cutlass.const_expr(ACTIVATION == "situglu"):
            act_x = act_situ_gate(act_elt, beta1)
            dact_x = dact_dsitu_gate(act_elt, beta1)
        elif cutlass.const_expr(ACTIVATION == "swiglu"):
            s = sigmoidf(act_elt)
            act_x = act_elt * s
            dact_x = act_elt * s * (Float32(1.0) - s) + s
        else:
            act_fn, dact_fn = GATED_ACTIVATIONS[ACTIVATION]
            act_x = act_fn(act_elt)
            dact_x = dact_fn(act_elt)
        out_act = dact_x * grad_elt * gate
        if cutlass.const_expr(dgate is None):
            out_gate = act_x * grad_elt
        else:
            out_gate = dgate * act_x * grad_elt
    return out_act, out_gate


@cute.jit
def gated_elementwise_smem_tile(
    sAct_tile: cute.Tensor,
    sGate_tile: cute.Tensor,
    sGrad_tile: Optional[cute.Tensor],
    limit: Optional[Float32],
    alpha: Optional[Float32],
    glu_linear_offset: Optional[Float32],
    beta1: Optional[Float32],
    beta2: Optional[Float32],
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
    DTYPE: cutlass.Constexpr[type],
    TILE_X: cutlass.Constexpr[int],
):
    """Compute the gated op on one staged tile, in place, rounded to the input dtype.

    Thread tidx owns column tidx of the tile, which is also the element set it owns in the
    colwise quantization pass, so a colwise-only consumer needs no barrier after this.
    """
    tidx, _, _ = cute.arch.thread_idx()

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE, 1), stride=(1, 1)),
    )
    sAct_thread = cute.composition(sAct_tile, tv_layout)[tidx, None]
    sGate_thread = cute.composition(sGate_tile, tv_layout)[tidx, None]
    if cutlass.const_expr(IS_BWD):
        sGrad_thread = cute.composition(sGrad_tile, tv_layout)[tidx, None]
    if cutlass.const_expr(is_packed16(DTYPE)):
        truncate_f32 = truncate_f32_f16 if DTYPE is cutlass.Float16 else truncate_f32_bf16

    for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
        grad_elt = None
        if cutlass.const_expr(IS_BWD):
            grad_elt = Float32(sGrad_thread[i])
        out_act, out_gate = _gated_element(
            Float32(sAct_thread[i]),
            Float32(sGate_thread[i]),
            grad_elt,
            limit,
            alpha,
            glu_linear_offset,
            beta1,
            beta2,
            ACTIVATION=ACTIVATION,
            IS_BWD=IS_BWD,
        )
        # Numerical truncation: round to the input dtype, like the CUDA kernels do before
        # taking the amax. The cast below is then exact.
        if cutlass.const_expr(is_packed16(DTYPE)):
            out_act = truncate_f32(out_act)
            if cutlass.const_expr(IS_BWD):
                out_gate = truncate_f32(out_gate)
        sAct_thread[i] = DTYPE(out_act)
        if cutlass.const_expr(IS_BWD):
            sGate_thread[i] = DTYPE(out_gate)


@cute.jit
def quantize_mxfp8_smem_tile_general_partial(
    sX_tile: cute.Tensor,
    sO_row_tile: Optional[cute.Tensor],
    mS_row_stage: Optional[cute.Tensor],
    sO_col_tile: Optional[cute.Tensor],
    mS_col_stage: Optional[cute.Tensor],
    tile_row_start: Int32,
    tile_col_start: Int32,
    M: Int32,
    N: Int32,
    ROWWISE: cutlass.Constexpr[bool],
    COLWISE: cutlass.Constexpr[bool],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    DTYPE: cutlass.Constexpr[type],
    FP8_DTYPE: cutlass.Constexpr[type],
    WITH_GEMM_SWIZZLED_SCALES: cutlass.Constexpr[bool],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],
):
    """A partial quantize_mxfp8_smem_tile_general function with some args fixed."""
    quantize_mxfp8_smem_tile_general(
        sX_tile,
        None,
        sO_row_tile,
        mS_row_stage,
        sO_col_tile,
        mS_col_stage,
        None,
        tile_row_start,
        tile_col_start,
        M,
        N,
        ROWWISE=ROWWISE,
        COLWISE=COLWISE,
        WITH_AMAX=False,
        WITH_ACT=False,
        WITH_DACT=False,
        ACTIVATION=None,
        ACTIVATION_TABLE="none",
        MAX_NORM_RCP=MAX_NORM_RCP,
        DTYPE=DTYPE,
        FP8_DTYPE=FP8_DTYPE,
        WITH_GEMM_SWIZZLED_SCALES=WITH_GEMM_SWIZZLED_SCALES,
        TILE_X=TILE_X,
        TILE_Y=TILE_Y,
        # The gated op maps the zeros TMA fills out-of-bounds elements with to zero.
        SKIP_INPUT_MASKING=True,
        SKIP_SCALE_BOUNDS=False,
        CACHE_ACTIVATION=False,
        DBIAS_REDUCTION_IN_COLWISE=False,
        COLWISE_DBIAS_REDUCTION_ONLY=False,
        DBIAS_REDUCTION_IN_ROWWISE=False,
    )


@cute.jit
def _split_gated_scales(
    mS_row: Optional[cute.Tensor],
    mS_col: Optional[cute.Tensor],
    M: Int32,
    N: Int32,
    ROWWISE: cutlass.Constexpr[bool],
    COLWISE: cutlass.Constexpr[bool],
    SWIZZLED: cutlass.Constexpr[bool],
):
    """Act and gate views of the backward output's scales; returns
    (act_row, gate_row, act_col, gate_col).

    The [M, 2N] output is [d act | d gate] and its scales are laid out for the whole output. Each
    view has the shape of an [M, N] tensor's scales and the strides of the whole output's, and
    starts where the whole output's layout places that half's first scale. For swizzled scales
    this needs the halves to be whole 128x4 tiles (N % 128 == 0, required by the C++ dispatcher).
    """
    act_row = gate_row = act_col = gate_col = None
    if cutlass.const_expr(SWIZZLED):
        mS_row, mS_col = derive_swizzled_scale_layout(M, 2 * N, ROWWISE, COLWISE, mS_row, mS_col)
        half_tiles = N // 128
        if cutlass.const_expr(ROWWISE):
            row_tiler = (
                cute.make_layout((32, 4, cute.ceil_div(M, 128))),  # all rows
                cute.make_layout((4, half_tiles)),  # one half's scale columns
            )
            act_row = cute.local_tile(mS_row, row_tiler, (0, 0))
            gate_row = cute.local_tile(mS_row, row_tiler, (0, 1))
        if cutlass.const_expr(COLWISE):
            col_tiler = (
                cute.make_layout((4, cute.ceil_div(cute.ceil_div(M, 32), 4))),  # all scale rows
                cute.make_layout((32, 4, half_tiles)),  # one half's columns
            )
            act_col = cute.local_tile(mS_col, col_tiler, (0, 0))
            gate_col = cute.local_tile(mS_col, col_tiler, (0, 1))
    else:
        # Compact scales are row-major, so each half is a block of scale columns
        if cutlass.const_expr(ROWWISE):
            row_tiler = (mS_row.shape[0], N // MXFP8_BLOCK_SCALING_SIZE)
            act_row = cute.local_tile(mS_row, row_tiler, (0, 0))
            gate_row = cute.local_tile(mS_row, row_tiler, (0, 1))
        if cutlass.const_expr(COLWISE):
            col_tiler = (mS_col.shape[0], N)
            act_col = cute.local_tile(mS_col, col_tiler, (0, 0))
            gate_col = cute.local_tile(mS_col, col_tiler, (0, 1))
    return act_row, gate_row, act_col, gate_col


class MXFP8GatedQuantizeG2SKernel(MXFP8QuantizeGeneralG2SKernel):
    """gated_mxfp8.cuh in CuTeDSL. The general kernel's pipeline, loading the act and gate halves
    (and grad, backward) each stage and storing one (forward) or two (backward) quantized halves.

    Inherits the plain-cast tile config, which is gated_mxfp8.cuh's: 64 threads per CTA, each CTA
    covering two vertically stacked 32x64 tiles.
    """

    def __init__(self, cfg: MXFP8GatedQuantizeConfig):
        super().__init__(cfg)
        # Same occupancy hint as the general kernel with a fused activation
        self._MIN_BLOCKS_PER_MP = 1
        # Every pipeline stage has its own SMEM buffers, so a CTA with fewer tiles than stages
        # would allocate buffers it never fills
        assert (
            self._NUM_STAGES <= self._NUM_TILES
        ), f"{self._NUM_STAGES} pipeline stages for {self._NUM_TILES} tiles per CTA"

    @cute.jit
    def __call__(
        self,
        mGatedInput: cute.Tensor,  # [M, 2N] = [act | gate]
        mGrad: Optional[cute.Tensor],  # [M, N], bwd only
        # Outputs, with C = N (FWD) or 2N (BWD)
        mO_row: Optional[cute.Tensor],  # [M, C]
        mS_row: Optional[cute.Tensor],  # [roundup(M, 128), roundup(C / 32, 4)]
        mO_col: Optional[cute.Tensor],  # [M, C], not transposed
        mS_col: Optional[cute.Tensor],  # [roundup(ceil(M / 32), 4), roundup(C, 128)]
        # Special activation params, None if not used by the activation
        limit: Optional[Float32],  # Clamped SwiGLU only
        alpha: Optional[Float32],  # Clamped SwiGLU only
        glu_linear_offset: Optional[Float32],  # Clamped SwiGLU only
        beta1: Optional[Float32],  # SiTUGLU only
        beta2: Optional[Float32],  # SiTUGLU only
        stream: CUstream,
    ):
        """
        mGatedInput: [a | g] with shape M x 2N, where a is the activation half and g is the gate half.
        mGrad: None for FWD, [dy] with shape [M, N] for BWD
        FWD mode:
        out = quantize(act(a) * gate(g)) with shape [M, N]
        BWD mode:
        out = quantize([ dy * gate(g) * dact(a)  |  dy * act(a) * dgate(g) ]) with shape [M, 2N]
        """
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                f"[CuTeDSL] MXFP8GatedQuantizeG2SKernel.__call__() with config: {self.cfg}\n"
            )

        M = mGatedInput.shape[0]
        # Pick any mO that is used just to get N
        mO = mO_row if mO_row is not None else mO_col
        # FWD: mO is [M, N]; BWD: mGrad is [M, N]
        N = mO.shape[1] if not self.cfg.IS_BWD else mGrad.shape[1]

        # The act and gate halves of the [M, 2N] input
        mAct = cute.local_tile(mGatedInput, (M, N), (0, 0))
        mGate = cute.local_tile(mGatedInput, (M, N), (0, 1))

        mO_row_dact = mO_row_dgate = mO_col_dact = mO_col_dgate = None  # BWD only
        mS_row_dact = mS_row_dgate = mS_col_dact = mS_col_dgate = None  # BWD only
        if cutlass.const_expr(self.cfg.IS_BWD):
            # The [M, 2N] output is [d act | d gate]; the kernel takes its two halves
            # TODO(kainingz): how to implement this in a more elegant way
            mS_row_dact, mS_row_dgate, mS_col_dact, mS_col_dgate = _split_gated_scales(
                mS_row,
                mS_col,
                M,
                N,
                self.cfg.ROWWISE,
                self.cfg.COLWISE,
                self.cfg.WITH_GEMM_SWIZZLED_SCALES,
            )
            if cutlass.const_expr(self.cfg.ROWWISE):
                mO_row_dact = cute.local_tile(mO_row, (M, N), (0, 0))
                mO_row_dgate = cute.local_tile(mO_row, (M, N), (0, 1))
            if cutlass.const_expr(self.cfg.COLWISE):
                mO_col_dact = cute.local_tile(mO_col, (M, N), (0, 0))
                mO_col_dgate = cute.local_tile(mO_col, (M, N), (0, 1))
            # In BWD mode we won't use these, and we should set them to None so we don't build TMA atoms for them
            mO_row = mO_col = mS_row = mS_col = None
        elif cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            mS_row, mS_col = derive_swizzled_scale_layout(
                M, N, self.cfg.ROWWISE, self.cfg.COLWISE, mS_row, mS_col
            )

        atoms = self._make_tma_atoms(
            mGrad,
            mAct,
            mGate,
            mO_row,
            mO_col,
            mO_row_dact,
            mO_row_dgate,
            mO_col_dact,
            mO_col_dgate,
        )
        # The TMA atoms are spread from a list, which pylint cannot count
        self.kernel(  # pylint: disable=no-value-for-parameter
            M,
            N,
            mS_row,
            mS_col,
            mS_row_dact,
            mS_row_dgate,
            mS_col_dact,
            mS_col_dgate,
            limit,
            alpha,
            glu_linear_offset,
            beta1,
            beta2,
            *atoms,
        ).launch(
            grid=[
                cute.ceil_div(N, self._TILE_COLS),
                cute.ceil_div(M, self._TILE_ROWS * self._NUM_TILES),
            ],
            block=[self._THREADS_PER_CTA],
            min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
            stream=stream,
        )

    @cute.jit
    def _make_tma_atoms(
        self,
        mGrad: Optional[cute.Tensor],
        mAct: cute.Tensor,
        mGate: cute.Tensor,
        mO_row: Optional[cute.Tensor],  # FWD only
        mO_col: Optional[cute.Tensor],  # FWD only
        mO_row_dact: Optional[cute.Tensor],  # BWD only
        mO_row_dgate: Optional[cute.Tensor],  # BWD only
        mO_col_dact: Optional[cute.Tensor],  # BWD only
        mO_col_dgate: Optional[cute.Tensor],  # BWD only
    ):
        """Build TMA atoms for input / quantized output; scales will not be loaded / stored by TMA."""
        smem_tile_layout = cute.make_ordered_layout(
            (self._TILE_ROWS, self._TILE_COLS), order=(1, 0)
        )
        cta_tiler = (self._TILE_ROWS, self._TILE_COLS)
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        atoms = []
        for op, mT in (
            (op_load, mGrad),
            (op_load, mAct),
            (op_load, mGate),
            (op_store, mO_row),
            (op_store, mO_col),
            (op_store, mO_row_dact),
            (op_store, mO_row_dgate),
            (op_store, mO_col_dact),
            (op_store, mO_col_dgate),
        ):
            if cutlass.const_expr(mT is None):
                atoms.extend([None, None])
            else:
                atoms.extend(
                    cute.nvgpu.cpasync.make_tiled_tma_atom(
                        op, mT, smem_tile_layout, cta_tiler, num_multicast=1
                    )
                )
        return atoms

    @cute.kernel
    def kernel(
        self,
        M: Int32,
        N: Int32,
        mS_row: Optional[cute.Tensor],  # FWD only
        mS_col: Optional[cute.Tensor],  # FWD only
        mS_row_dact: Optional[cute.Tensor],  # BWD only
        mS_row_dgate: Optional[cute.Tensor],  # BWD only
        mS_col_dact: Optional[cute.Tensor],  # BWD only
        mS_col_dgate: Optional[cute.Tensor],  # BWD only
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        beta1: Optional[Float32],
        beta2: Optional[Float32],
        # TMA for mGrad, only used in BWD
        tma_atom_grad: Optional[cute.CopyAtom],
        tma_src_grad: Optional[cute.Tensor],
        # TMA for mAct and mGate, used in both FWD and BWD
        tma_atom_act: cute.CopyAtom,
        tma_src_act: cute.Tensor,
        tma_atom_gate: cute.CopyAtom,
        tma_src_gate: cute.Tensor,
        # TMA for mO_row/col, only used in FWD
        tma_atom_row: Optional[cute.CopyAtom],
        tma_dst_row: Optional[cute.Tensor],
        tma_atom_col: Optional[cute.CopyAtom],
        tma_dst_col: Optional[cute.Tensor],
        # TMA for mO_row/col_dact/dgate, only used in BWD
        tma_atom_row_dact: Optional[cute.CopyAtom],
        tma_dst_row_dact: Optional[cute.Tensor],
        tma_atom_row_dgate: Optional[cute.CopyAtom],
        tma_dst_row_dgate: Optional[cute.Tensor],
        tma_atom_col_dact: Optional[cute.CopyAtom],
        tma_dst_col_dact: Optional[cute.Tensor],
        tma_atom_col_dgate: Optional[cute.CopyAtom],
        tma_dst_col_dgate: Optional[cute.Tensor],
    ):
        """
        FWD mode:
        mAct: activation half input
        mGate: gate half input
        mO_row/col: quantized output for row / column directions
        mS_row/col: output scales for row / column directions

        BWD mode:
        mGrad: incoming gradient
        mAct: activation half input from FWD pass
        mGate: gate half input from FWD pass
        mO_row/col_dact: quantized activation gradient for row / column directions
        mO_row/col_dgate: quantized gate gradient for row / column directions
        mS_row/col_dact: output scales for activation gradient row / column directions
        mS_row/col_dgate: output scales for gate gradient row / column directions
        mO_row/col: quantized gradient (both activation and gate) for row / column directions
        mS_row/col: output scales for gradient (both activation and gate) row / column directions

        mAct: both FWD and BWD
        mGate: both FWD and BWD
        mGrad: BWD only
        mO_row/col: both FWD (N) and BWD (2N)
        mS_row/col: both FWD (N) and BWD (2N)
        mO_row/col_dact: BWD only
        mO_row/col_dgate: BWD only
        mS_row/col_dact: BWD only
        mS_row/col_dgate: BWD only
        """

        cfg = self.cfg
        DTYPE = cfg.DTYPE
        TILE_ROWS = self._TILE_ROWS
        TILE_COLS = self._TILE_COLS
        NUM_STAGES = self._NUM_STAGES
        NUM_TILES = self._NUM_TILES

        # The scales of one (TILE_ROWS, TILE_COLS) tile
        row_scale_tile = (TILE_ROWS, TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        col_scale_tile = (TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, TILE_COLS)

        # Shared memory: one ((TILE_ROWS, TILE_COLS), NUM_STAGES) ring buffer per operand
        smem = cutlass.memory.SmemAllocator()

        @cute.struct
        class BarrierStorage:  # pylint: disable=missing-class-docstring
            mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]

        storage = smem.allocate(BarrierStorage)
        stages_layout = cute.make_layout(
            ((TILE_ROWS, TILE_COLS), NUM_STAGES),
            stride=((TILE_COLS, 1), TILE_ROWS * TILE_COLS),
        )

        # Allocate SMEM buffers for inputs
        sAct = smem.allocate_tensor(DTYPE, stages_layout, byte_alignment=128)
        sGate = smem.allocate_tensor(DTYPE, stages_layout, byte_alignment=128)
        sGrad = None
        if cutlass.const_expr(cfg.IS_BWD):
            sGrad = smem.allocate_tensor(DTYPE, stages_layout, byte_alignment=128)

        # Allocate SMEM buffers for outputs
        sO_row = sO_col = None
        sO_row_dact = sO_row_dgate = sO_col_dact = sO_col_dgate = None
        if cutlass.const_expr(cfg.ROWWISE):
            if cutlass.const_expr(cfg.IS_BWD):
                sO_row_dact = smem.allocate_tensor(cfg.FP8_DTYPE, stages_layout, byte_alignment=128)
                sO_row_dgate = smem.allocate_tensor(
                    cfg.FP8_DTYPE, stages_layout, byte_alignment=128
                )
            else:
                sO_row = smem.allocate_tensor(cfg.FP8_DTYPE, stages_layout, byte_alignment=128)
        if cutlass.const_expr(cfg.COLWISE):
            if cutlass.const_expr(cfg.IS_BWD):
                sO_col_dact = smem.allocate_tensor(cfg.FP8_DTYPE, stages_layout, byte_alignment=128)
                sO_col_dgate = smem.allocate_tensor(
                    cfg.FP8_DTYPE, stages_layout, byte_alignment=128
                )
            else:
                sO_col = smem.allocate_tensor(cfg.FP8_DTYPE, stages_layout, byte_alignment=128)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()

        # Issue TMA descriptor prefetch using the TMA warp (warp 0)
        if warp_idx == 0:
            for load_atom in (tma_atom_act, tma_atom_gate, tma_atom_grad):
                if cutlass.const_expr(load_atom is not None):
                    cute.nvgpu.cpasync.prefetch_descriptor(load_atom)

        # TMA warp (warp 0)'s elected thread produces (issues TMA), every warp consumes
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._NUM_WARPS)
        # Input has 2 tiles (act + gate) in FWD, 3 tiles (grad + act + gate) in BWD
        num_inputs = 3 if cfg.IS_BWD else 2
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            # Every input tile of a stage arrives on the same barrier
            tx_count=num_inputs * TILE_ROWS * TILE_COLS * DTYPE.width // 8,
            cta_layout_vmnk=None,
        )
        prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_STAGES)
        cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_STAGES)

        cta_coord = (bidy, bidx)
        # Shape of each individual tile. A CTA processes one tile at a time
        tile_shape = (TILE_ROWS, TILE_COLS)
        # The block processed by a CTA, consisting of NUM_TILES vertically stacked tiles
        cta_block = (NUM_TILES * TILE_ROWS, TILE_COLS)

        # Prepare parameters to pass to the SMEM tile quantize function by dact / dgate / output, where
        # for each group, input, quantized output are SMEM tiles, and scales are GMEM tiles
        cta_block_row_scales = (NUM_TILES * TILE_ROWS, TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        cta_block_col_scales = (NUM_TILES * TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, TILE_COLS)
        if cutlass.const_expr(cfg.IS_BWD):
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row_dact = cute.local_tile(mS_row_dact, cta_block_row_scales, cta_coord)
                mS_row_dgate = cute.local_tile(mS_row_dgate, cta_block_row_scales, cta_coord)
            if cutlass.const_expr(cfg.COLWISE):
                mS_col_dact = cute.local_tile(mS_col_dact, cta_block_col_scales, cta_coord)
                mS_col_dgate = cute.local_tile(mS_col_dgate, cta_block_col_scales, cta_coord)
            # In BWD mode we output gradient for both the act and gate halves, which are stored in sAct and sGate respectively
            tile_quantizer_params = [
                (
                    sAct,
                    sO_row_dact,
                    mS_row_dact,
                    sO_col_dact,
                    mS_col_dact,
                ),  # We will write unquantized dact to sAct in-place
                (
                    sGate,
                    sO_row_dgate,
                    mS_row_dgate,
                    sO_col_dgate,
                    mS_col_dgate,
                ),  # We will write unquantized dgate to sGate in-place
            ]
        else:
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row = cute.local_tile(mS_row, cta_block_row_scales, cta_coord)
            if cutlass.const_expr(cfg.COLWISE):
                mS_col = cute.local_tile(mS_col, cta_block_col_scales, cta_coord)
            # In FWD mode we only output the result after the gated op, which is stored in sAct
            tile_quantizer_params = [
                (sAct, sO_row, mS_row, sO_col, mS_col)
            ]  # We will write unquantized output to sAct in-place

        # Prepare input TMA (atom, smem partition, gmem partition) triples: act and gate, and BWD's grad
        inputs = []
        for atom_in, src_in, sX_in in (
            (tma_atom_act, tma_src_act, sAct),
            (tma_atom_gate, tma_src_gate, sGate),
            (tma_atom_grad, tma_src_grad, sGrad),
        ):
            if cutlass.const_expr(atom_in is not None):
                tXsX_in, tXgX_in = cute.nvgpu.cpasync.tma_partition(
                    atom_in,
                    0,
                    cute.make_layout(1),
                    sX_in,
                    cute.zipped_divide(cute.local_tile(src_in, cta_block, cta_coord), tile_shape),
                )
                inputs.append((atom_in, tXsX_in, tXgX_in))

        # Prepare output TMA (atom, smem partition, gmem partition) triples of the outputs this config has
        TMA_atom_and_partitions = []
        for atom_out, dst_out, sO_out in (
            (tma_atom_row, tma_dst_row, sO_row),  # FWD only
            (tma_atom_col, tma_dst_col, sO_col),  # FWD only
            (tma_atom_row_dact, tma_dst_row_dact, sO_row_dact),  # BWD only
            (tma_atom_row_dgate, tma_dst_row_dgate, sO_row_dgate),  # BWD only
            (tma_atom_col_dact, tma_dst_col_dact, sO_col_dact),  # BWD only
            (tma_atom_col_dgate, tma_dst_col_dgate, sO_col_dgate),  # BWD only
        ):
            if cutlass.const_expr(atom_out is not None):
                tXsO_out, tXgO_out = cute.nvgpu.cpasync.tma_partition(
                    atom_out,
                    0,
                    cute.make_layout(1),
                    sO_out,
                    cute.zipped_divide(cute.local_tile(dst_out, cta_block, cta_coord), tile_shape),
                )
                TMA_atom_and_partitions.append((atom_out, tXsO_out, tXgO_out))

        # Barrier init must be visible to every thread
        cute.arch.sync_threads()

        # Prologue: fill the pipeline
        if warp_idx == 0:
            for s in cutlass.range_constexpr(NUM_STAGES):
                mainloop_pipeline.producer_acquire(prod_state)
                for load_atom, load_sX, load_gX in inputs:
                    cute.copy(
                        load_atom,
                        load_gX[(None, (s, 0))],
                        load_sX[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                mainloop_pipeline.producer_commit(prod_state)
                prod_state.advance()

        for tile_idx in cutlass.range(NUM_TILES, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            # We only reuse previously used SMEM buffers when we have more tiles to process than pipeline stages, and only
            # after we've already finished one round and wrap around to the start of the ring buffer
            if cutlass.const_expr(NUM_TILES > NUM_STAGES):
                if tile_idx >= NUM_STAGES:
                    # Only allow at most NUM_STAGES - 1 store groups reading SMEM on the fly, so the tile NUM_STAGES
                    # ago must finish before we reuse its SMEM buffer
                    if warp_idx == 0:
                        cute.arch.cp_async_bulk_wait_group(NUM_STAGES - 1, read=True)
                    cute.arch.sync_threads()
            # The tile should be ready to use now

            stage_idx = cons_state.index
            tile_coord = (tile_idx, 0)  # within this CTA's block

            # Select current stage's SMEM tile
            sAct_tile = sAct[(None, stage_idx)]
            sGate_tile = sGate[(None, stage_idx)]
            sGrad_tile = None
            if cutlass.const_expr(cfg.IS_BWD):
                sGrad_tile = sGrad[(None, stage_idx)]

            # Compute the gated op on this tile in fp32, cast back to the original dtype,
            # and store the result in-place in the SMEM tile without quantization.
            # FWD: write the output into sAct_tile
            # BWD: write the act half's gradient to sAct_tile, and write the gate half's gradient to sGate_tile
            gated_elementwise_smem_tile(
                sAct_tile,  # we also store the FWD output / BWD act half's gradient here
                sGate_tile,  # we also store the BWD gate half's gradient here
                sGrad_tile,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                ACTIVATION=cfg.GATED_ACTIVATION,
                IS_BWD=cfg.IS_BWD,
                DTYPE=DTYPE,
                TILE_X=TILE_COLS,
            )

            # The rowwise quantization pass reads elements other threads wrote so we need to sync here
            if cutlass.const_expr(cfg.ROWWISE):
                cute.arch.sync_threads()

            # Quantize every output of this tile into its SMEM buffers and scales
            for sX_out, sO_row_out, mS_row_cta, sO_col_out, mS_col_cta in tile_quantizer_params:
                # Slice the SMEM / GMEM tile for the current iteration
                sO_row_tile = None
                mS_row_stage = None
                if cutlass.const_expr(cfg.ROWWISE):
                    sO_row_tile = sO_row_out[(None, stage_idx)]
                    mS_row_stage = cute.flatten(
                        cute.local_tile(mS_row_cta, row_scale_tile, tile_coord)
                    )
                sO_col_tile = None
                mS_col_stage = None
                if cutlass.const_expr(cfg.COLWISE):
                    sO_col_tile = sO_col_out[(None, stage_idx)]
                    mS_col_stage = cute.flatten(
                        cute.local_tile(mS_col_cta, col_scale_tile, tile_coord)
                    )
                # Use the general MXFP8 SMEM tile quantize function to quantize it
                quantize_mxfp8_smem_tile_general_partial(
                    sX_out[(None, stage_idx)],  # The SMEM tile to quantize
                    sO_row_tile,  # SMEM tile for rowwise quantized output
                    mS_row_stage,  # GMEM tile for rowwise quantized scales
                    sO_col_tile,  # SMEM tile for columnwise quantized output
                    mS_col_stage,  # GMEM tile for columnwise quantized scales
                    # The tile's first row and column in the whole tensor, for bounds checks
                    (bidy * NUM_TILES + tile_idx) * TILE_ROWS,
                    bidx * TILE_COLS,
                    M,
                    N,
                    ROWWISE=cfg.ROWWISE,
                    COLWISE=cfg.COLWISE,
                    MAX_NORM_RCP=cfg.MAX_NORM_RCP,
                    DTYPE=DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    WITH_GEMM_SWIZZLED_SCALES=cfg.WITH_GEMM_SWIZZLED_SCALES,
                    TILE_X=TILE_COLS,
                    TILE_Y=TILE_ROWS,
                )

            # Make the SMEM writes visible to the TMA's async proxy before it reads them
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            mainloop_pipeline.consumer_release(cons_state)

            if warp_idx == 0:
                # The tile quantize function has already written the quantized outputs to SMEM,
                # so we can issue TMA stores from SMEM to GMEM
                for store_atom, store_sO, store_gO in TMA_atom_and_partitions:
                    cute.copy(
                        store_atom,
                        store_sO[(None, stage_idx)],
                        store_gO[(None, (tile_idx, 0))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()

            # Refill the stage just released with the tile NUM_STAGES ahead
            if warp_idx == 0:
                next_tile_idx = tile_idx + NUM_STAGES
                if next_tile_idx < NUM_TILES:
                    mainloop_pipeline.producer_acquire(prod_state)
                    for load_atom, load_sX, load_gX in inputs:
                        cute.copy(
                            load_atom,
                            load_gX[(None, (next_tile_idx, 0))],
                            load_sX[(None, prod_state.index)],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                        )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        # Only wait until the in-flight TMA stores have read SMEM, so the CTA retires early
        cute.arch.cp_async_bulk_wait_group(0, read=True)
