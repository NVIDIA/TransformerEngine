# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL MXFP8 dequantization kernel, with its compilation and TVM-FFI registration."""

# Local @cute.struct classes are shared-memory storage descriptors.
# pylint: disable=missing-class-docstring

import logging
import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Float8E8M0FNU, Float32, Int32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    device_compute_capability,
    exp2f,
    str_to_cutlass_dtype,
)
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    MXFP8_BLOCK_SCALING_SIZE,
    SYM_N_DIVISIBILITY,
    THREADS_PER_WARP,
    derive_swizzled_scale_layout,
)

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")


@cute.jit
def dequantize_mxfp8_smem_tile(
    sQ: cute.Tensor,  # Input quantized SMEM tile
    sO: cute.Tensor,  # Output dequantized SMEM tile
    gS: cute.Tensor,  # This tile's E8M0 scales in GMEM
    TILE_ROWS: cutlass.Constexpr[int],  # How many rows in the tile
    TILE_COLS: cutlass.Constexpr[int],  # How many columns in the tile
    ELEMS_PER_THREAD: cutlass.Constexpr[int],  # How many values each thread dequantizes
    DEQUANTIZE_FROM_ROWWISE: cutlass.Constexpr[bool],  # Which direction to dequantize from
):
    """Dequantize a smem tile sQ into sO using the scales in gS.

    Requires CTA threads multiplied by ELEMS_PER_THREAD to cover the tile.

    sQ, sO: (TILE_ROWS, TILE_COLS)
    gS:
    - rowwise: (TILE_ROWS, max(1, TILE_COLS // MXFP8_BLOCK_SCALING_SIZE))
    - colwise: (max(1, TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE), TILE_COLS)
    """

    tidx, _, _ = cute.arch.thread_idx()

    # MXFP8_BLOCK_SCALING_SIZE must be divisible by ELEMS_PER_THREAD
    assert MXFP8_BLOCK_SCALING_SIZE % ELEMS_PER_THREAD == 0
    FRAGS_SHARING_SAME_SCALE = MXFP8_BLOCK_SCALING_SIZE // ELEMS_PER_THREAD
    if cutlass.const_expr(DEQUANTIZE_FROM_ROWWISE):
        assert TILE_COLS % ELEMS_PER_THREAD == 0
        THR_ROWS = TILE_ROWS
        THR_COLS = TILE_COLS // ELEMS_PER_THREAD
        # Row-major threads: a warp reads contiguous 16 B chunks along a row (no bank conflicts)
        thr_layout = cute.make_layout((THR_ROWS, THR_COLS), stride=(THR_COLS, 1))
        val_layout = cute.make_layout((1, ELEMS_PER_THREAD), stride=(0, 1))
        scale_y = tidx // THR_COLS
        scale_x = tidx % THR_COLS
        # Every `FRAGS_SHARING_SAME_SCALE` threads along x direction share the same scale
        scale_e8m0 = gS[scale_y, scale_x // FRAGS_SHARING_SAME_SCALE]
    else:
        assert TILE_ROWS % ELEMS_PER_THREAD == 0 and MXFP8_BLOCK_SCALING_SIZE % TILE_ROWS == 0
        THR_ROWS = TILE_ROWS // ELEMS_PER_THREAD
        THR_COLS = TILE_COLS
        thr_layout = cute.make_layout((THR_ROWS, THR_COLS))
        val_layout = cute.make_layout((ELEMS_PER_THREAD, 1), stride=(1, 0))
        scale_y = tidx % THR_ROWS
        scale_x = tidx // THR_ROWS
        # Every `FRAGS_SHARING_SAME_SCALE` threads along y direction share the same scale
        scale_e8m0 = gS[scale_y // FRAGS_SHARING_SAME_SCALE, scale_x]

    _, tv_layout = cute.make_layout_tv(
        thr_layout=thr_layout,
        val_layout=val_layout,
    )

    sQ_frag = cute.composition(sQ, tv_layout)[tidx, None]
    sO_frag = cute.composition(sO, tv_layout)[tidx, None]

    rQ_frag = cute.make_rmem_tensor(cute.make_layout(ELEMS_PER_THREAD), sQ.element_type)
    rO_frag = cute.make_rmem_tensor(cute.make_layout(ELEMS_PER_THREAD), sO.element_type)

    cute.autovec_copy(sQ_frag, rQ_frag)

    scale_f32 = exp2f(scale_e8m0)
    rO_frag.store((rQ_frag.load().to(Float32) * scale_f32).to(sO.element_type))

    cute.autovec_copy(rO_frag, sO_frag)


class MXFP8DequantizeConfig:
    """Compile-time configuration of the CuTeDSL MXFP8 dequantize kernel."""

    def __init__(
        self,
        fp8_dtype: str,
        out_dtype: str,
        rowwise: bool,
        colwise: bool,
        with_gemm_swizzled_scales: bool,
    ):
        if not (rowwise or colwise):
            raise ValueError("at least one of rowwise and columnwise data must be present")
        if fp8_dtype not in ("Float8E4M3", "Float8E5M2"):
            raise ValueError(
                f"unknown FP8 dtype {fp8_dtype!r}; expected 'Float8E4M3' or 'Float8E5M2'"
            )
        if out_dtype not in ("Float32", "Float16", "BFloat16"):
            raise ValueError(
                f"unknown output dtype {out_dtype!r}; expected Float32|Float16|BFloat16"
            )
        self.FP8_DTYPE = str_to_cutlass_dtype(fp8_dtype)
        self.FP8_DTYPE_STR = fp8_dtype
        self.OUT_DTYPE = str_to_cutlass_dtype(out_dtype)
        self.OUT_DTYPE_STR = out_dtype
        # Which data copies (each with its scales) the caller passes. Both are (M, N) row-major.
        self.ROWWISE = rowwise
        self.COLWISE = colwise
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales

    def __str__(self):
        return (
            f"MXFP8DequantizeConfig(fp8_dtype={self.FP8_DTYPE_STR}, out_dtype={self.OUT_DTYPE_STR},"
            f" rowwise={self.ROWWISE}, colwise={self.COLWISE},"
            f" swizzled={self.WITH_GEMM_SWIZZLED_SCALES})"
        )

    __repr__ = __str__


class MXFP8DequantizeKernel:
    """MXFP8 dequantize with staged TMA I/O, a 1:1 port of dequantize_mxfp8_kernel in
    cast/mxfp8/dequantize_mxfp8.cuh (same tiling, thread mapping, barriers and pipelining).

    Each CTA processes _NUM_TILES (_TILE_ROWS, _TILE_COLS) tiles stacked vertically through a
    _NUM_STAGES-deep smem ring. The next tile's TMA load is issued before the current one is
    consumed, and the output is written back with a TMA store.
    """

    _NUM_STAGES = 2
    _NUM_TILES = 8  # Each CTA handles 8 tiles stacked vertically

    _THREADS_PER_CTA = 128
    _NUM_WARPS = _THREADS_PER_CTA // THREADS_PER_WARP
    # How many rows and columns are processed by a CTA at a time
    # Rowwise: each thread owns _ELEMS_PER_THREAD elements in a row
    # Colwise: each thread owns _ELEMS_PER_THREAD elements in a column
    _TILE_ROWS = 16
    _TILE_COLS = _THREADS_PER_CTA
    _ELEMS_PER_THREAD = _TILE_ROWS * _TILE_COLS // _THREADS_PER_CTA  # 16

    # How many rows and columns are processed by a CTA in total
    CTA_BLOCK_ROWS = _TILE_ROWS * _NUM_TILES
    CTA_BLOCK_COLS = _TILE_COLS

    def __init__(self, cfg: MXFP8DequantizeConfig):
        self.cfg = cfg
        # Prefer using rowwise quantization result to dequantize
        self.DEQUANTIZE_FROM_ROWWISE = cfg.ROWWISE

    @cute.jit
    def __call__(
        self,
        mX_row: cute.Tensor,
        mS_row: cute.Tensor,
        mX_col: cute.Tensor,
        mS_col: cute.Tensor,
        mO: cute.Tensor,
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8DequantizeKernel.__call__() with config: {self.cfg}\n")

        mX = mX_row if cutlass.const_expr(self.DEQUANTIZE_FROM_ROWWISE) else mX_col
        mS = mS_row if cutlass.const_expr(self.DEQUANTIZE_FROM_ROWWISE) else mS_col
        M, N = mX.shape
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            if cutlass.const_expr(self.DEQUANTIZE_FROM_ROWWISE):
                mS, _ = derive_swizzled_scale_layout(M, N, True, False, mS, None)
            else:
                _, mS = derive_swizzled_scale_layout(M, N, False, True, None, mS)

        # Unswizzled boxes, like create_2D_tensor_map's default.
        tiler = (self._TILE_ROWS, self._TILE_COLS)
        smem_tile_layout = cute.make_ordered_layout(tiler, order=(1, 0))
        tma_atom_in, tma_in = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mX, smem_tile_layout, tiler
        )
        tma_atom_out, tma_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), mO, smem_tile_layout, tiler
        )

        grid = [
            cute.ceil_div(Int32(N), self.CTA_BLOCK_COLS),
            cute.ceil_div(M, self.CTA_BLOCK_ROWS),
            1,
        ]
        self.kernel(mS, tma_atom_in, tma_in, tma_atom_out, tma_out).launch(
            grid=grid, block=[self._THREADS_PER_CTA, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mS: cute.Tensor,
        tma_atom_in: cute.CopyAtom,
        tma_in: cute.Tensor,
        tma_atom_out: cute.CopyAtom,
        tma_out: cute.Tensor,
    ):
        """Device entry: out = exp2(scale - 127) * x, computed in FP32."""

        TILE_SIZE = self._TILE_ROWS * self._TILE_COLS

        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # This CTA's block, split into _NUM_TILES tiles: ((_TILE_ROWS, _TILE_COLS), (_NUM_TILES, 1)).
        cta_block_tiler = (self.CTA_BLOCK_ROWS, self.CTA_BLOCK_COLS)
        tiler = (self._TILE_ROWS, self._TILE_COLS)

        # Select the block of tiles that belongs to this CTA
        tma_in = cute.local_tile(tma_in, cta_block_tiler, (bidy, bidx))
        tma_out = cute.local_tile(tma_out, cta_block_tiler, (bidy, bidx))

        gQ = cute.zipped_divide(tma_in, tiler)
        gO = cute.zipped_divide(tma_out, tiler)

        if cutlass.const_expr(self.DEQUANTIZE_FROM_ROWWISE):
            scale_cta_block_tiler = (
                self.CTA_BLOCK_ROWS,
                self.CTA_BLOCK_COLS // MXFP8_BLOCK_SCALING_SIZE,
            )
            # self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE might be fractional meaning a tile only owns a
            # partial MXFP8 block. In which case this tile still own at least 1 scale on that direction
            scale_tiler = (
                self._TILE_ROWS,
                max(1, self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE),
            )
            # Each data row has its own scales, so tiles never share them
            tiles_sharing_same_scale = 1
        else:
            scale_cta_block_tiler = (
                self.CTA_BLOCK_ROWS // MXFP8_BLOCK_SCALING_SIZE,
                self.CTA_BLOCK_COLS,
            )
            # self._TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE might be fractional meaning a tile only owns a
            # partial MXFP8 block. In which case this tile still own at least 1 scale on that direction
            scale_tiler = (
                max(1, self._TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE),
                self._TILE_COLS,
            )
            # A tile shorter than an MXFP8 block shares its scale row with the other tiles in it
            tiles_sharing_same_scale = max(1, MXFP8_BLOCK_SCALING_SIZE // self._TILE_ROWS)
        # If tiles_sharing_same_scale > 1, then this tile owns only a fraction of the MXFP8 block and
        # shares its scale with the other tiles, so we need to index with tile_idx // tiles_sharing_same_scale
        # to obtain the scale slice that belongs to this tile
        mS = cute.local_tile(mS, scale_cta_block_tiler, (bidy, bidx))

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.cfg.FP8_DTYPE, self._NUM_STAGES * TILE_SIZE], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.cfg.OUT_DTYPE, self._NUM_STAGES * TILE_SIZE], 128
            ]

        storage = cutlass.memory.SmemAllocator().allocate(SharedStorage)
        ring_layout = cute.zipped_product(
            cute.make_ordered_layout(tiler, order=(1, 0)), cute.make_layout(self._NUM_STAGES)
        )
        sQ = storage.sQ.get_tensor(ring_layout)
        sO = storage.sO.get_tensor(ring_layout)

        tXs_in, tXg_in = cute.nvgpu.cpasync.tma_partition(
            tma_atom_in, 0, cute.make_layout(1), sQ, gQ
        )
        tXs_out, tXg_out = cute.nvgpu.cpasync.tma_partition(
            tma_atom_out, 0, cute.make_layout(1), sO, gO
        )

        # Only warp 0 is the producer (issues TMA); every warp is a consumer.
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self._NUM_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self._NUM_WARPS),
            tx_count=TILE_SIZE * self.cfg.FP8_DTYPE.width // 8,
            cta_layout_vmnk=None,
        )
        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._NUM_STAGES
        )

        # Prologue: fill the pipeline.
        if warp_idx == 0:
            for it in cutlass.range_constexpr(self._NUM_STAGES):
                mainloop_pipeline.producer_acquire(prod_state)
                cute.copy(
                    tma_atom_in,
                    tXg_in[(None, it)],
                    tXs_in[(None, prod_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                )
                mainloop_pipeline.producer_commit(prod_state)
                prod_state.advance()

        for tile_idx in cutlass.range_constexpr(self._NUM_TILES):
            stage = cons_state.index
            # The first _NUM_STAGES tiles don't have to wait for previous TMA read because they are not used yet
            if cutlass.const_expr(tile_idx >= self._NUM_STAGES):
                if warp_idx == 0:
                    # Wait for previous TMA to finish their read so we can write to this output tile
                    cute.arch.cp_async_bulk_wait_group(self._NUM_STAGES - 1, read=True)
                cute.arch.sync_threads()

            mainloop_pipeline.consumer_wait(cons_state)

            gS = cute.local_tile(mS, scale_tiler, (tile_idx // tiles_sharing_same_scale, 0))

            dequantize_mxfp8_smem_tile(
                sQ[(None, None), stage],
                sO[(None, None), stage],
                gS,
                self._TILE_ROWS,
                self._TILE_COLS,
                self._ELEMS_PER_THREAD,
                self.DEQUANTIZE_FROM_ROWWISE,
            )

            # Make the smem writes visible to the TMA engine before the store.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            mainloop_pipeline.consumer_release(cons_state)
            cons_state.advance()

            if warp_idx == 0:
                cute.copy(
                    tma_atom_out,
                    tXs_out[(None, stage)],
                    tXg_out[(None, tile_idx)],
                )
                cute.arch.cp_async_bulk_commit_group()
                if cutlass.const_expr(tile_idx + self._NUM_STAGES < self._NUM_TILES):
                    mainloop_pipeline.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom_in,
                        tXg_in[(None, tile_idx + self._NUM_STAGES)],
                        tXs_in[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        if warp_idx == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=False)
        cute.arch.sync_threads()


def compile_dequantize_function_from_cfg(cfg: MXFP8DequantizeConfig):
    """Compile the dequantize kernel for cfg with symbolic shapes and a TVM-FFI entrypoint."""
    sym_M = cute.sym_int32()
    sym_N = cute.sym_int32(divisibility=SYM_N_DIVISIBILITY)
    # Scales are allocated at the padded shapes described in compile_cutedsl_function_from_cfg.
    scale_row_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
    scale_col_shape = (cute.sym_int32(divisibility=4), cute.sym_int32(divisibility=128))

    def fake(dtype, shape, align, present=True):
        if not present:
            return None
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=align
        )

    return cute.compile(
        MXFP8DequantizeKernel(cfg),
        fake(cfg.FP8_DTYPE, (sym_M, sym_N), 16, cfg.ROWWISE),  # mX_row
        fake(Float8E8M0FNU, scale_row_shape, 4, cfg.ROWWISE),  # mS_row
        fake(cfg.FP8_DTYPE, (sym_M, sym_N), 16, cfg.COLWISE),  # mX_col
        fake(Float8E8M0FNU, scale_col_shape, 4, cfg.COLWISE),  # mS_col
        fake(cfg.OUT_DTYPE, (sym_M, sym_N), 16),  # mO
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def get_mxfp8_dequantization_function(
    fn_name: str,
    fp8_dtype: str,
    out_dtype: str,
    rowwise: bool,
    colwise: bool,
    with_gemm_swizzled_scales: bool,
) -> bool:
    """Compile the MXFP8 dequantize kernel for this config and register it in the TVM-FFI global
    registry under `fn_name`. rowwise/colwise say which data copies the C++ caller passes; the
    direction actually dequantized is chosen in MXFP8DequantizeConfig. Returns False if the config is unsupported, so the C++ dispatcher
    falls back to the CUDA kernel. Mirrors get_mxfp8_quantization_function.
    """
    try:
        if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
            return True

        major, minor = device_compute_capability()
        if major < 10:
            logger.warning(
                "CuTeDSL MXFP8 backend requires compute capability >= 10.0 (Blackwell), "
                "but detected %d.%d; falling back to the CUDA C++ kernel.",
                major,
                minor,
            )
            return False

        try:
            cfg = MXFP8DequantizeConfig(
                fp8_dtype, out_dtype, rowwise, colwise, with_gemm_swizzled_scales
            )
        except ValueError as e:
            logger.warning(
                "CuTeDSL MXFP8 backend does not support this dequantize config, "
                "falling back to the CUDA C++ kernel: %s",
                e,
            )
            return False

        logger.debug("Compiling CuTeDSL MXFP8 dequantization kernel for %s", cfg)
        compiled = compile_dequantize_function_from_cfg(cfg)
        # Register the native function to skip the Python argument-parsing wrapper
        # (see get_mxfp8_quantization_function).
        native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
        tvm_ffi.register_global_func(
            fn_name, native if native is not None else compiled, override=True
        )
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL MXFP8 dequantize kernel compilation & registration failed, falling back to"
            " the CUDA C++ kernel: %s",
            e,
        )
        return False
