# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTeDSL grouped MXFP8 dequantization kernel, with its compilation and TVM-FFI registration.

Port of group_dequantize_mxfp8_kernel in cast/mxfp8/group_dequantize_mxfp8.cuh. The C++ bridge
is group_dequantize_mxfp8_cutedsl.cuh.
"""

# Local @cute.struct classes are shared-memory storage descriptors.
# pylint: disable=missing-class-docstring

import logging
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Float8E8M0FNU, Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.tensor_utils import TensorMapManager, TensorMapUpdateMode
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    device_compute_capability,
    str_to_cutlass_dtype,
)
from .dequantize_mxfp8 import dequantize_mxfp8_smem_tile
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    MXFP8_BLOCK_SCALING_SIZE,
    SYM_N_DIVISIBILITY,
    THREADS_PER_WARP,
)
from .group_quantize_mxfp8_common import (
    SAME_BOTH_DIMS,
    VARYING_FIRST_DIM,
    VARYING_LAST_DIM,
    VARYING_BOTH_DIMS,
    SUPPORTED_SHAPE_REPS,
    MAX_SUPPORTED_TENSORS,
    BYTES_PER_TENSORMAP,
    INT64_PER_TENSORMAP,
    find_tensor_from_offsets,
    member_shape,
    update_member_tensormaps,
)

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")


class MXFP8GroupDequantizeConfig:
    """Compile-time configuration of the CuTeDSL grouped MXFP8 dequantize kernel."""

    def __init__(self, fp8_dtype: str, out_dtype: str, rowwise: bool, shape_rep: str):
        if fp8_dtype not in ("Float8E4M3", "Float8E5M2"):
            raise ValueError(
                f"unknown FP8 dtype {fp8_dtype!r}; expected 'Float8E4M3' or 'Float8E5M2'"
            )
        if out_dtype not in ("Float32", "Float16", "BFloat16"):
            raise ValueError(
                f"unknown output dtype {out_dtype!r}; expected Float32|Float16|BFloat16"
            )
        if shape_rep not in SUPPORTED_SHAPE_REPS:
            raise ValueError(
                f"unsupported shape representation {shape_rep!r}; expected one of"
                f" {SUPPORTED_SHAPE_REPS}"
            )
        self.FP8_DTYPE = str_to_cutlass_dtype(fp8_dtype)
        self.FP8_DTYPE_STR = fp8_dtype
        self.OUT_DTYPE = str_to_cutlass_dtype(out_dtype)
        self.OUT_DTYPE_STR = out_dtype
        # Dequantize from the rowwise data and scales, else from the columnwise ones.
        self.ROWWISE = rowwise
        self.SHAPE_REP = shape_rep
        # As in the CUDA kernel, a group with a common last dim is dequantized as one tensor.
        self.IS_SINGLE_TENSOR = shape_rep in (SAME_BOTH_DIMS, VARYING_FIRST_DIM)

    def __str__(self):
        return (
            f"MXFP8GroupDequantizeConfig(fp8_dtype={self.FP8_DTYPE_STR},"
            f" out_dtype={self.OUT_DTYPE_STR}, rowwise={self.ROWWISE},"
            f" shape_rep={self.SHAPE_REP})"
        )

    __repr__ = __str__


class MXFP8GroupDequantizeKernel:
    """Grouped MXFP8 dequantize with staged TMA I/O.

    Mirrors the CUDA kernel's work mapping: each CTA takes one (JOB_ROWS, JOB_COLS) chunk of one
    member, processed as NUM_TILES (TILE_ROWS, TILE_COLS) tiles stacked vertically through a
    NUM_STAGES-deep SMEM ring. A group with a common last dim is dequantized as one tensor.
    Otherwise each member gets its own TMA descriptors, and a CTA finds its member by a binary
    search over the offsets, which balances the work across members of any size.
    """

    NUM_STAGES = 2
    NUM_TILES = 8
    THREADS_PER_CTA = 128
    NUM_WARPS = THREADS_PER_CTA // THREADS_PER_WARP
    TILE_ROWS = 16
    TILE_COLS = THREADS_PER_CTA
    ELEMS_PER_THREAD = TILE_ROWS * TILE_COLS // THREADS_PER_CTA  # 16
    JOB_ROWS = TILE_ROWS * NUM_TILES  # 128
    JOB_COLS = TILE_COLS  # 128
    JOB_ELTS = JOB_ROWS * JOB_COLS

    def __init__(self, cfg: MXFP8GroupDequantizeConfig):
        self.cfg = cfg
        # Prefer using rowwise quantization result to dequantize
        self.DEQUANTIZE_FROM_ROWWISE = cfg.ROWWISE

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # FP8 data, logical (M, N) view of the group
        mS: cute.Tensor,  # E8M0 scales, flat; per-member layouts are derived from the shapes
        mO: cute.Tensor,  # High-precision output, logical (M, N) view of the group
        mOffsets: Optional[cute.Tensor],  # int64[num_tensors + 1] element offsets
        mFirstDims: Optional[cute.Tensor],  # int64[num_tensors] (VARYING_BOTH_DIMS)
        mLastDims: Optional[cute.Tensor],  # int64[num_tensors] (VARYING_LAST/BOTH_DIMS)
        mInputTensormaps: Optional[cute.Tensor],  # per-member input descriptors, flat int64
        mOutputTensormaps: Optional[cute.Tensor],  # per-member output descriptors, flat int64
        num_tensors: Int32,
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8GroupDequantizeKernel.__call__() cfg: {self.cfg}\n")

        first_logical_dim, last_logical_dim = mX.shape

        # Unswizzled boxes, like create_2D_tensor_map's default.
        tiler = (self.TILE_ROWS, self.TILE_COLS)
        smem_tile_layout = cute.make_ordered_layout(tiler, order=(1, 0))
        tma_atom_in, tma_in = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX, smem_tile_layout, tiler
        )
        tma_atom_out, tma_out = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, smem_tile_layout, tiler
        )

        # Jobs per row of jobs. With varying last dims, the stacked members' 128-column strips.
        jobs_X = cute.ceil_div(Int64(last_logical_dim), self.JOB_COLS)
        if cutlass.const_expr(self.cfg.SHAPE_REP == VARYING_BOTH_DIMS):
            # Members of 128-aligned dims are whole jobs laid out back to back.
            num_jobs = cute.ceil_div(
                Int64(first_logical_dim) * Int64(last_logical_dim), self.JOB_ELTS
            )
        else:
            num_jobs = jobs_X * cute.ceil_div(Int64(first_logical_dim), self.JOB_ROWS)

        if cutlass.const_expr(not self.cfg.IS_SINGLE_TENSOR):
            self._update_descriptors_kernel(
                mX,
                mO,
                mOffsets,
                mFirstDims,
                mLastDims,
                mInputTensormaps,
                mOutputTensormaps,
                first_logical_dim,
                last_logical_dim,
                tma_atom_in,
                tma_atom_out,
            ).launch(grid=[num_tensors, 1, 1], block=[THREADS_PER_WARP, 1, 1], stream=stream)

        self.kernel(
            mS,
            mOffsets,
            mFirstDims,
            mLastDims,
            mInputTensormaps,
            mOutputTensormaps,
            first_logical_dim,
            last_logical_dim,
            num_tensors,
            jobs_X,
            tma_atom_in,
            tma_in,
            tma_atom_out,
            tma_out,
        ).launch(grid=[num_jobs, 1, 1], block=[self.THREADS_PER_CTA, 1, 1], stream=stream)

    @staticmethod
    def _tensormap_ptr(tmap: TensorMapManager, mTensormaps: cute.Tensor, tensor_id: Int32):
        return tmap.get_tensormap_ptr(mTensormaps.iterator + tensor_id * INT64_PER_TENSORMAP)

    @cute.kernel
    def _update_descriptors_kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        mOffsets: cute.Tensor,
        mFirstDims: Optional[cute.Tensor],
        mLastDims: cute.Tensor,
        mInputTensormaps: cute.Tensor,
        mOutputTensormaps: cute.Tensor,
        first_logical_dim: Int64,
        last_logical_dim: Int64,
        tma_atom_in: cute.CopyAtom,
        tma_atom_out: cute.CopyAtom,
    ):
        """Point member tensor_id's input and output descriptors at its (rows, cols) block."""
        tensor_id, _, _ = cute.arch.block_idx()
        rows, cols = member_shape(
            self.cfg.SHAPE_REP,
            mFirstDims,
            mLastDims,
            tensor_id,
            first_logical_dim,
            last_logical_dim,
        )
        update_member_tensormaps(
            (mX, mO),
            (tma_atom_in, tma_atom_out),
            (
                mInputTensormaps.iterator + tensor_id * INT64_PER_TENSORMAP,
                mOutputTensormaps.iterator + tensor_id * INT64_PER_TENSORMAP,
            ),
            Int64(mOffsets[tensor_id]),
            rows,
            cols,
        )

    @cute.jit
    def _scale_tiles(self, mS: cute.Tensor, base: Int64, rows: Int64, cols: Int64) -> cute.Tensor:
        """Compact scales of a (rows, cols) tensor starting at element `base` of mS, tiled per
        (TILE_ROWS, TILE_COLS) data tile. The padded extents cover every tile of every job."""
        if cutlass.const_expr(self.DEQUANTIZE_FROM_ROWWISE):
            scale_shape = (
                cute.round_up(rows, 128),
                cute.round_up(cute.ceil_div(cols, MXFP8_BLOCK_SCALING_SIZE), 4),
            )
            # One scale per row and 32 columns
            tile_scale_shape = (self.TILE_ROWS, self.TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        else:
            scale_shape = (
                cute.round_up(cute.ceil_div(rows, MXFP8_BLOCK_SCALING_SIZE), 4),
                cute.round_up(cols, 128),
            )
            # A tile is shorter than a scaling block, so it owns a single scale row
            tile_scale_shape = (1, self.TILE_COLS)
        mS_member = cute.make_tensor(
            cute.make_ptr(
                Float8E8M0FNU,
                mS.iterator.toint() + base,
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            cute.make_ordered_layout(scale_shape, order=(1, 0)),
        )
        return cute.zipped_divide(mS_member, tile_scale_shape)

    @cute.jit
    def _member_scale_base(self, base_offset: Int64, rows: Int64) -> Int64:
        """Element offset of a multi-tensor member's scales, as computed by the CUDA kernel."""
        scale_base = base_offset // MXFP8_BLOCK_SCALING_SIZE
        if cutlass.const_expr(self.cfg.SHAPE_REP == VARYING_LAST_DIM):
            # Every member has `rows` rows, so the members before it hold base_offset / rows cols.
            sum_prev_cols = base_offset // rows
            if cutlass.const_expr(self.DEQUANTIZE_FROM_ROWWISE):
                scale_base = cute.round_up(rows, 128) // MXFP8_BLOCK_SCALING_SIZE * sum_prev_cols
            else:
                scale_base = (
                    cute.round_up(cute.ceil_div(rows, MXFP8_BLOCK_SCALING_SIZE), 4) * sum_prev_cols
                )
        return scale_base

    @cute.kernel
    def kernel(
        self,
        mS: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mLastDims: Optional[cute.Tensor],
        mInputTensormaps: Optional[cute.Tensor],
        mOutputTensormaps: Optional[cute.Tensor],
        first_logical_dim: Int64,
        last_logical_dim: Int64,
        num_tensors: Int32,
        jobs_X: Int64,
        tma_atom_in: cute.CopyAtom,
        tma_in: cute.Tensor,
        tma_atom_out: cute.CopyAtom,
        tma_out: cute.Tensor,
    ):
        """Device entry: out = exp2(scale - 127) * x, computed in FP32."""
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tiler = (self.TILE_ROWS, self.TILE_COLS)
        TILE_SIZE = self.TILE_ROWS * self.TILE_COLS

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cute.Int64, 2 * self.NUM_STAGES]
            sQ: cute.struct.Align[
                cute.struct.MemRange[cfg.FP8_DTYPE, self.NUM_STAGES * TILE_SIZE], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[cfg.OUT_DTYPE, self.NUM_STAGES * TILE_SIZE], 128
            ]

        storage = cutlass.memory.SmemAllocator().allocate(SharedStorage)
        ring_layout = cute.zipped_product(
            cute.make_ordered_layout(tiler, order=(1, 0)), cute.make_layout(self.NUM_STAGES)
        )
        sQ = storage.sQ.get_tensor(ring_layout)
        sO = storage.sO.get_tensor(ring_layout)

        # Tile coordinates are shared by every member; per-member descriptors supply the geometry.
        tXs_in, tXg_in = cpasync.tma_partition(
            tma_atom_in, 0, cute.make_layout(1), sQ, cute.zipped_divide(tma_in, tiler)
        )
        tXs_out, tXg_out = cpasync.tma_partition(
            tma_atom_out, 0, cute.make_layout(1), sO, cute.zipped_divide(tma_out, tiler)
        )

        # Only warp 0 is the producer (issues TMA); every warp is a consumer.
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.NUM_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.NUM_WARPS),
            tx_count=TILE_SIZE * cfg.FP8_DTYPE.width // 8,
            cta_layout_vmnk=None,
        )
        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.NUM_STAGES
        )
        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        job_args = (warp_idx, sQ, sO, tXs_in, tXg_in, tXs_out, tXg_out, tma_atom_in, tma_atom_out)

        job_id = Int64(bidx)
        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            scale_tiles = self._scale_tiles(
                mS, Int64(0), Int64(first_logical_dim), Int64(last_logical_dim)
            )
            self._process_job(
                (job_id // jobs_X) * self.NUM_TILES,
                job_id % jobs_X,
                scale_tiles,
                None,
                None,
                tmap,
                mainloop_pipeline,
                prod_state,
                cons_state,
                job_args,
            )
        else:
            # Find this job's member and its job coordinates in it, as the CUDA kernel does.
            if cutlass.const_expr(cfg.SHAPE_REP == VARYING_LAST_DIM):
                strip = job_id % jobs_X
                job_y = job_id // jobs_X
                tensor_id = find_tensor_from_offsets(
                    mOffsets, num_tensors, strip * self.JOB_COLS * Int64(first_logical_dim)
                )
                rows, cols = member_shape(
                    cfg.SHAPE_REP,
                    mFirstDims,
                    mLastDims,
                    tensor_id,
                    first_logical_dim,
                    last_logical_dim,
                )
                job_x = strip - Int64(mOffsets[tensor_id]) // (rows * self.JOB_COLS)
            else:
                tensor_id = find_tensor_from_offsets(mOffsets, num_tensors, job_id * self.JOB_ELTS)
                rows, cols = member_shape(
                    cfg.SHAPE_REP,
                    mFirstDims,
                    mLastDims,
                    tensor_id,
                    first_logical_dim,
                    last_logical_dim,
                )
                job_in_tensor = job_id - Int64(mOffsets[tensor_id]) // self.JOB_ELTS
                member_jobs_X = cutlass.max(cute.ceil_div(cols, self.JOB_COLS), Int64(1))
                job_y = job_in_tensor // member_jobs_X
                job_x = job_in_tensor % member_jobs_X

            # Jobs past the group's end (e.g. a buffer sized ahead for CUDA graphs) have no work.
            if job_x * self.JOB_COLS < cols and job_y * self.JOB_ROWS < rows:
                desc_in = self._tensormap_ptr(tmap, mInputTensormaps, tensor_id)
                desc_out = self._tensormap_ptr(tmap, mOutputTensormaps, tensor_id)
                # Only thread 0 issues TMA, so only it acquires the updated descriptors.
                if tidx == 0:
                    tmap.fence_tensormap_update(desc_in)
                    tmap.fence_tensormap_update(desc_out)
                scale_tiles = self._scale_tiles(
                    mS,
                    self._member_scale_base(Int64(mOffsets[tensor_id]), rows),
                    rows,
                    cols,
                )
                self._process_job(
                    job_y * self.NUM_TILES,
                    job_x,
                    scale_tiles,
                    desc_in,
                    desc_out,
                    tmap,
                    mainloop_pipeline,
                    prod_state,
                    cons_state,
                    job_args,
                )

        # Drain every TMA store before the CTA releases its shared-memory source buffers.
        if warp_idx == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=False)
        cute.arch.sync_threads()

    def _issue_load(
        self,
        mainloop_pipeline: pipeline.PipelineTmaAsync,
        prod_state: pipeline.PipelineState,
        tile_y: Int64,
        tile_x: Int64,
        desc_in: Optional[cute.Pointer],
        tmap: TensorMapManager,
        tXs_in: cute.Tensor,
        tXg_in: cute.Tensor,
        tma_atom_in: cute.CopyAtom,
    ):
        """Load data tile (tile_y, tile_x) into the producer's ring slot. Called by warp 0; the
        caller advances prod_state outside that branch."""
        mainloop_pipeline.producer_acquire(prod_state)
        tma_desc_kwargs = {}
        if cutlass.const_expr(desc_in is not None):
            tma_desc_kwargs["tma_desc_ptr"] = tmap.get_tensormap_ptr(
                desc_in, cute.AddressSpace.generic
            )
        cute.copy(
            tma_atom_in,
            tXg_in[(None, (tile_y, tile_x))],
            tXs_in[(None, prod_state.index)],
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
            **tma_desc_kwargs,
        )
        mainloop_pipeline.producer_commit(prod_state)

    @cute.jit
    def _process_job(
        self,
        job_tile_y: Int64,  # Data tile row of the job's first tile
        job_tile_x: Int64,  # Data tile column of the job
        scale_tiles: cute.Tensor,  # The member's scales, tiled per data tile
        desc_in: Optional[cute.Pointer],  # Member descriptors, None for a single tensor
        desc_out: Optional[cute.Pointer],
        tmap: TensorMapManager,
        mainloop_pipeline: pipeline.PipelineTmaAsync,
        prod_state: pipeline.PipelineState,
        cons_state: pipeline.PipelineState,
        job_args: tuple,
    ):
        """Dequantize the job's NUM_TILES tiles."""
        warp_idx, sQ, sO, tXs_in, tXg_in, tXs_out, tXg_out, tma_atom_in, tma_atom_out = job_args
        load_args = (desc_in, tmap, tXs_in, tXg_in, tma_atom_in)

        # Prologue: fill the pipeline.
        for tile in cutlass.range_constexpr(self.NUM_STAGES):
            if warp_idx == 0:
                self._issue_load(
                    mainloop_pipeline, prod_state, job_tile_y + tile, job_tile_x, *load_args
                )
            prod_state.advance()

        for tile in cutlass.range_constexpr(self.NUM_TILES):
            stage = cons_state.index
            # Reusing an output slot waits for the TMA store that last read it.
            if cutlass.const_expr(tile >= self.NUM_STAGES):
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(self.NUM_STAGES - 1, read=True)
                cute.arch.sync_threads()
            mainloop_pipeline.consumer_wait(cons_state)

            tile_y = job_tile_y + tile
            if cutlass.const_expr(self.DEQUANTIZE_FROM_ROWWISE):
                scale_tile_coord = (tile_y, job_tile_x)
            else:
                # Every MXFP8_BLOCK_SCALING_SIZE // TILE_ROWS tiles share one scale row
                scale_tile_coord = (
                    tile_y // (MXFP8_BLOCK_SCALING_SIZE // self.TILE_ROWS),
                    job_tile_x,
                )
            dequantize_mxfp8_smem_tile(
                sQ[(None, None), stage],
                sO[(None, None), stage],
                scale_tiles[((None, None), scale_tile_coord)],
                self.TILE_ROWS,
                self.TILE_COLS,
                self.ELEMS_PER_THREAD,
                self.DEQUANTIZE_FROM_ROWWISE,
            )

            # Make the SMEM writes visible to the TMA engine before the store.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            mainloop_pipeline.consumer_release(cons_state)
            cons_state.advance()

            if warp_idx == 0:
                tma_desc_kwargs = {}
                if cutlass.const_expr(desc_out is not None):
                    tma_desc_kwargs["tma_desc_ptr"] = tmap.get_tensormap_ptr(
                        desc_out, cute.AddressSpace.generic
                    )
                cute.copy(
                    tma_atom_out,
                    tXs_out[(None, stage)],
                    tXg_out[(None, (tile_y, job_tile_x))],
                    **tma_desc_kwargs,
                )
                cute.arch.cp_async_bulk_commit_group()
                if cutlass.const_expr(tile + self.NUM_STAGES < self.NUM_TILES):
                    self._issue_load(
                        mainloop_pipeline,
                        prod_state,
                        tile_y + self.NUM_STAGES,
                        job_tile_x,
                        *load_args,
                    )
            if cutlass.const_expr(tile + self.NUM_STAGES < self.NUM_TILES):
                prod_state.advance()


def compile_group_dequantize_function_from_cfg(cfg: MXFP8GroupDequantizeConfig):
    """Compile the grouped dequantize kernel for cfg with symbolic shapes and a TVM-FFI
    entrypoint."""
    logical_shape = (cute.sym_int64(), cute.sym_int64(divisibility=SYM_N_DIVISIBILITY))

    def fake(dtype, shape, align, present=True):
        if not present:
            return None
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    multi_tensor = not cfg.IS_SINGLE_TENSOR
    tensormaps_shape = (MAX_SUPPORTED_TENSORS * INT64_PER_TENSORMAP,)
    return cute.compile(
        MXFP8GroupDequantizeKernel(cfg),
        fake(cfg.FP8_DTYPE, logical_shape, 16),  # mX
        fake(Float8E8M0FNU, (cute.sym_int64(),), 4),  # mS
        fake(cfg.OUT_DTYPE, logical_shape, 16),  # mO
        fake(Int64, (cute.sym_int32(),), 8, multi_tensor),  # mOffsets
        fake(Int64, (cute.sym_int32(),), 8, cfg.SHAPE_REP == VARYING_BOTH_DIMS),  # mFirstDims
        fake(Int64, (cute.sym_int32(),), 8, multi_tensor),  # mLastDims
        fake(Int64, tensormaps_shape, 128, multi_tensor),  # mInputTensormaps
        fake(Int64, tensormaps_shape, 128, multi_tensor),  # mOutputTensormaps
        Int32(1),  # num_tensors, supplied at runtime
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def get_mxfp8_group_dequantization_function(
    fn_name: str,
    fp8_dtype: str,
    out_dtype: str,
    rowwise: bool,
    shape_rep: str,
) -> bool:
    """Compile the grouped MXFP8 dequantize kernel for this config and register it in the TVM-FFI
    global registry under `fn_name`. Returns False if the config is unsupported, so the C++
    dispatcher falls back to the CUDA kernel. Mirrors get_mxfp8_group_quantization_function.
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
            cfg = MXFP8GroupDequantizeConfig(fp8_dtype, out_dtype, rowwise, shape_rep)
        except ValueError as e:
            logger.warning(
                "CuTeDSL grouped MXFP8 backend does not support this dequantize config, "
                "falling back to the CUDA C++ kernel: %s",
                e,
            )
            return False

        logger.debug("Compiling CuTeDSL grouped MXFP8 dequantization kernel for %s", cfg)
        compiled = compile_group_dequantize_function_from_cfg(cfg)
        # Register the native function to skip the Python argument-parsing wrapper
        # (see get_mxfp8_quantization_function).
        native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
        tvm_ffi.register_global_func(
            fn_name, native if native is not None else compiled, override=True
        )
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL grouped MXFP8 dequantize kernel compilation & registration failed, falling"
            " back to the CUDA C++ kernel: %s",
            e,
        )
        return False
