# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Generic grouped MXFP8 quantization kernel implemented in CuTeDSL (TMA -> SMEM).

Handles every grouped config. MXFP8GroupQuantizeEntry in group_quantize_mxfp8_common.py
dispatches plain casts to faster kernels when they apply and falls back to this one otherwise.
"""

# pylint: disable=missing-class-docstring

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Boolean, Float32, Int32, Int64, Float8E8M0FNU
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.testing import assert_ as runtime_assert
from cutlass.utils import TensorMapUpdateMode
from cutlass.tensor_utils import TensorMapManager
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from .quantize_mxfp8_common import (
    MXFP8_BLOCK_SCALING_SIZE,
    derive_swizzled_scale_layout,
    noop_flag_is_set,
)
from .quantize_mxfp8_g2s_general import (
    reduce_rowwise_dbias,
    _quantize_rowwise_mxfp8,
    _quantize_colwise_mxfp8,
)
from .group_quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    THREADS_PER_WARP,
    INPUT_SLOT,
    ACT_INPUT_SLOT,
    ROWWISE_OUTPUT_SLOT,
    COLWISE_OUTPUT_SLOT,
    SAME_BOTH_DIMS,
    VARYING_FIRST_DIM,
    VARYING_LAST_DIM,
    VARYING_BOTH_DIMS,
    MAX_SUPPORTED_TENSORS,
    BYTES_PER_TENSORMAP,
    INT64_PER_TENSORMAP,
    ROWS_OFFSET,
    COLS_OFFSET,
    OFFSETS_OFFSET,
    MXFP8GroupQuantizeConfig,
    MXFP8GroupQuantizeKernelBase,
)


class MXFP8GroupQuantizeKernel(MXFP8GroupQuantizeKernelBase):
    """Grouped MXFP8 quantize mirroring group_quantize_mxfp8_kernel's strategy."""

    # Target persistent CTA count per SM for sizing the grid
    STATIC_PERSISTENT_WORKERS_PER_SM = 24
    # The shape of one pipeline stage processed by a CTA
    TILE_ROWS = 32
    TILE_COLS = 128
    PIPELINE_DEPTH = 2
    # The number of elements in a tile
    ELTS_PER_TILE = TILE_ROWS * TILE_COLS
    # CTA shape
    THREADS_PER_CTA = 128
    NUM_WARPS = THREADS_PER_CTA // THREADS_PER_WARP  # 4
    THREADS_X = TILE_COLS // MXFP8_BLOCK_SCALING_SIZE  # 4
    THREADS_Y = THREADS_PER_CTA // THREADS_X  # 32

    @staticmethod
    def supports(cfg: MXFP8GroupQuantizeConfig) -> bool:  # pylint: disable=unused-argument
        """Every grouped config."""
        return True

    @cute.jit
    def eligible_to_dispatch(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mLastDims: Optional[cute.Tensor],
        num_tensors: Int32,
        mNoop: cute.Pointer,
    ) -> Boolean:
        """Every call."""
        return Boolean(True)

    def __init__(self, cfg: MXFP8GroupQuantizeConfig, SM_COUNT: int) -> None:
        self.cfg = cfg
        self.SM_COUNT = SM_COUNT
        # A CTA processes (NUM_TILES_Y, NUM_TILES_X) tiles, NUM_STAGES tiles in total
        self.NUM_TILES_X = 2 if cfg.SHAPE_REP == VARYING_BOTH_DIMS else 1
        self.NUM_TILES_Y = 4
        self.NUM_STAGES = self.NUM_TILES_X * self.NUM_TILES_Y
        self.ELTS_PER_CTA = self.ELTS_PER_TILE * self.NUM_STAGES
        # The CUDA kernel honors the noop flag only without fused activations or dbias.
        self.CHECK_NOOP_FLAG = not (cfg.WITH_ACT or cfg.WITH_DACT or cfg.WITH_DBIAS)
        # The colwise pass caches the activation in the input tile for the rowwise pass.
        # Note: ReLU is fused into the conversion instead, which yields the same bytes.
        self.CACHE_ACTIVATION = (
            (cfg.WITH_ACT or cfg.WITH_DACT)
            and cfg.ROWWISE
            and cfg.COLWISE
            and cfg.ACTIVATION != "relu"
        )
        # Prefer to reduce dbias in the colwise pass if quantized in columnwise, else in the rowwise pass.
        self.DBIAS_IN_COLWISE = cfg.WITH_DBIAS and cfg.COLWISE
        self.DBIAS_IN_ROWWISE = cfg.WITH_DBIAS and not cfg.COLWISE
        # Rowwise dbias reduces through SMEM. Reuse the input tiles for it when they are large
        # enough (FP32 input), as the CUDA kernel does, and cap registers at 128, so it runs 4
        # CTAs per SM like the CUDA kernel instead of 3.
        dbias_bytes = self.THREADS_Y * self.THREADS_X * (MXFP8_BLOCK_SCALING_SIZE + 1) * 4
        input_bytes = self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH * cfg.DTYPE.width // 8
        self.ALIAS_DBIAS = self.DBIAS_IN_ROWWISE and dbias_bytes <= input_bytes
        self.MIN_BLOCKS_PER_MP = 4 if self.ALIAS_DBIAS else 0

    @cute.jit
    def _find_tensor_from_offsets(
        self, mOffsets: cute.Tensor, num_tensors: Int32, offset: Int64
    ) -> Int32:
        """Index of the tensor whose element range holds `offset` (find_tensor_from_offsets)."""
        low = Int32(1)
        hi = Int32(num_tensors)
        # Enough bisection steps for any group of up to MAX_SUPPORTED_TENSORS members.
        for _ in cutlass.range_constexpr(MAX_SUPPORTED_TENSORS.bit_length()):
            if low < hi:
                mid = low + (hi - low) // 2
                if Int64(mOffsets[mid]) <= offset:
                    low = mid + 1
                else:
                    hi = mid
        return low - 1

    @cute.jit
    def _scale_tensor(self, mS: cute.Tensor, base: Int64, layout: cute.Layout) -> cute.Tensor:
        """View the scale buffer from element `base` on with `layout`."""
        return cute.make_tensor(
            cute.make_ptr(
                Float8E8M0FNU,
                mS.iterator.toint() + base,
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            layout,
        )

    @cute.jit
    def _rowwise_scales(
        self, mS_row: cute.Tensor, base: Int64, rows: Int64, cols: Int64
    ) -> cute.Tensor:
        """Rowwise scales of a (rows, cols) tensor at `base`, tiled per 32x128 stage."""
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            mS_t, _ = derive_swizzled_scale_layout(
                rows, cols, True, False, self._scale_tensor(mS_row, base, cute.make_layout(1)), None
            )
        else:
            # Rowwise scale's divisibility guarantee: (128, 4)
            stride = cute.round_up(cute.ceil_div(cols, MXFP8_BLOCK_SCALING_SIZE), 4)
            mS_t = self._scale_tensor(
                mS_row, base, cute.make_layout((rows, stride), stride=(stride, 1))
            )
        return cute.zipped_divide(
            mS_t, (self.TILE_ROWS, self.TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        )

    @cute.jit
    def _colwise_scales(
        self, mS_col: cute.Tensor, base: Int64, rows: Int64, cols: Int64
    ) -> cute.Tensor:
        """Colwise scales of a (rows, cols) tensor at `base`, tiled per 32x128 stage."""
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            _, mS_t = derive_swizzled_scale_layout(
                rows, cols, False, True, None, self._scale_tensor(mS_col, base, cute.make_layout(1))
            )
        else:
            # Colwise scale's divisibility guarantee: (4, 128)
            stride = cute.round_up(cols, 128)
            mS_t = self._scale_tensor(
                mS_col,
                base,
                cute.make_layout((rows // MXFP8_BLOCK_SCALING_SIZE, stride), stride=(stride, 1)),
            )
        return cute.zipped_divide(
            mS_t, (self.TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, self.TILE_COLS)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],  # int64[num_tensors + 1], CSR element offsets
        mFirstDims: Optional[
            cute.Tensor
        ],  # int64[num_tensors] (VARYING_FIRST_DIM / VARYING_BOTH_DIMS)
        mLastDims: Optional[
            cute.Tensor
        ],  # int64[num_tensors] (VARYING_LAST_DIM / VARYING_BOTH_DIMS)
        mTensormaps: Optional[cute.Tensor],  # flat int64 view of CUDA TensorMapStorage
        num_tensors: Int32,
        mNoop: cute.Pointer,  # f32 cast_noop flag; may be null, checked on device
        mActInput: Optional[cute.Tensor],  # activation input, only with WITH_DACT
        mWorkspace: Optional[cute.Tensor],  # f32 partial dbias, only with WITH_DBIAS
        stream: CUstream,
    ) -> None:
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8GroupQuantizeKernel.__call__() cfg: {self.cfg}\n")

        cfg = self.cfg
        first_logical_dim = mX.shape[0]
        last_logical_dim = mX.shape[1]
        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_BOTH_DIMS):
            runtime_assert(
                first_logical_dim == 1, "VARYING_BOTH_DIMS requires logical shape [1, total]"
            )

        runtime_assert(num_tensors > 0, "Grouped quantization requires at least one tensor")

        # A TMA atom copies a TILE at a time
        smem_tile_layout = cute.make_ordered_layout((self.TILE_ROWS, self.TILE_COLS), order=(1, 0))
        cta_tiler = (self.TILE_ROWS, self.TILE_COLS)

        # TMA atom for loading the input tensor and the activation input (if WITH_DACT)
        op_load = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_x, tma_src = cpasync.make_tiled_tma_atom(
            op_load, mX, smem_tile_layout, cta_tiler, num_multicast=1
        )
        tma_atom_act = None
        tma_src_act = None
        if cutlass.const_expr(cfg.WITH_DACT):
            tma_atom_act, tma_src_act = cpasync.make_tiled_tma_atom(
                op_load, mActInput, smem_tile_layout, cta_tiler, num_multicast=1
            )

        # TMA atom for storing the rowwise and colwise outputs (if enabled)
        op_store = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_out_row = None
        tma_dst_out_row = None
        if cutlass.const_expr(cfg.ROWWISE):
            tma_atom_out_row, tma_dst_out_row = cpasync.make_tiled_tma_atom(
                op_store, mO_row, smem_tile_layout, cta_tiler, num_multicast=1
            )
        tma_atom_out_col = None
        tma_dst_out_col = None
        if cutlass.const_expr(cfg.COLWISE):
            tma_atom_out_col, tma_dst_out_col = cpasync.make_tiled_tma_atom(
                op_store, mO_col, smem_tile_layout, cta_tiler, num_multicast=1
            )

        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            # How many CTAs does the grouped tensor have in both directions
            jobs_Y = cute.ceil_div(Int64(first_logical_dim), (self.TILE_ROWS * self.NUM_TILES_Y))
            jobs_X = cute.ceil_div(Int64(last_logical_dim), self.TILE_COLS * self.NUM_TILES_X)
            # Flatten it to an 1D grid
            grid = [jobs_X * jobs_Y, 1, 1]
        else:
            # A placeholder for the kernel signature only; we won't use it in non-single tensor cases
            jobs_X = None

            # Below is just some heuristic to decide how many CTAs we should use per tensor

            # Estimate the total jobs across the group; each job has NUM_TILES_X * NUM_TILES_Y tiles
            estimated_jobs = cute.ceil_div(
                Int64(first_logical_dim) * Int64(last_logical_dim), self.ELTS_PER_CTA
            )
            # Divide the persistent worker budget evenly across tensors, with at least
            # one worker per tensor. Each worker may process several jobs.
            requested_CTAs_per_tensor = cutlass.max(
                Int64(1),
                Int64(self.SM_COUNT * self.STATIC_PERSISTENT_WORKERS_PER_SM) // Int64(num_tensors),
            )
            # In average how many jobs per tensor (only an average, the actual jobs per tensor may vary)
            average_jobs_per_tensor = cutlass.max(
                Int64(1), (estimated_jobs + Int64(num_tensors) - 1) // Int64(num_tensors)
            )
            # Don't launch more CTAs than the average jobs per tensor in case
            # STATIC_PERSISTENT_WORKERS_PER_SM causes redundancy
            CTAs_per_tensor = cutlass.min(requested_CTAs_per_tensor, average_jobs_per_tensor)
            grid = [Int32(CTAs_per_tensor), num_tensors, 1]

        # Only the multi-tensor representations need per-tensor descriptors.
        if cutlass.const_expr(not cfg.IS_SINGLE_TENSOR):
            self._update_descriptors_kernel(
                mX,
                mO_row,
                mO_col,
                mActInput,
                mOffsets,
                mFirstDims,
                mLastDims,
                mTensormaps,
                first_logical_dim,
                last_logical_dim,
                mX.element_type,
                tma_atom_x,
                tma_atom_out_row,
                tma_atom_out_col,
                tma_atom_act,
            ).launch(grid=[num_tensors, 1, 1], block=[THREADS_PER_WARP, 1, 1], stream=stream)

        self.kernel(
            mS_row,
            mS_col,
            mOffsets,
            mFirstDims,
            mTensormaps,
            mNoop,
            mWorkspace,
            first_logical_dim,
            last_logical_dim,
            num_tensors,
            jobs_X,
            mX.element_type,
            tma_atom_x,
            tma_src,
            tma_atom_act,
            tma_src_act,
            tma_atom_out_row,
            tma_dst_out_row,
            tma_atom_out_col,
            tma_dst_out_col,
        ).launch(
            grid=grid,
            block=[self.THREADS_PER_CTA, 1, 1],
            min_blocks_per_mp=self.MIN_BLOCKS_PER_MP,
            # Left unset, the carveout chosen with min_blocks_per_mp fits one CTA fewer
            preferred_smem_carveout=100 if self.MIN_BLOCKS_PER_MP else None,
            stream=stream,
        )

    @cute.kernel
    def _update_descriptors_kernel(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mActInput: Optional[cute.Tensor],
        mOffsets: cute.Tensor,
        mFirstDims: Optional[cute.Tensor],
        mLastDims: Optional[cute.Tensor],
        mTensormaps: cute.Tensor,
        first_logical_dim: Int64,
        last_logical_dim: Int64,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x: cute.CopyAtom,
        tma_atom_orow: Optional[cute.CopyAtom],
        tma_atom_ocol: Optional[cute.CopyAtom],
        tma_atom_act: Optional[cute.CopyAtom],
    ) -> None:
        """Update the per-tensor TMA descriptors for the group quantization kernel.

        mTensormaps is a flat int64 view of CUDA's TensorMapStorage: four descriptor
        arrays followed by the rows, cols, and offsets arrays.
        """
        cfg = self.cfg

        # For the descriptor update kernel, grid=[num_tensors, 1, 1]
        tensor_id, _, _ = cute.arch.block_idx()
        # Figure out how many rows and columns this tensor has, and where its first element is in the group.
        if cutlass.const_expr(cfg.SHAPE_REP in (VARYING_FIRST_DIM, VARYING_BOTH_DIMS)):
            rows = Int64(mFirstDims[tensor_id])
        else:
            rows = Int64(first_logical_dim)
        if cutlass.const_expr(cfg.SHAPE_REP in (VARYING_LAST_DIM, VARYING_BOTH_DIMS)):
            cols = Int64(mLastDims[tensor_id])
        else:
            cols = Int64(last_logical_dim)
        base_offset = Int64(mOffsets[tensor_id])

        # Same diagnostics as get_tensor_rows_num / get_tensor_cols_num. Like NVTE_DEVICE_ERROR
        # in a release build, they only print.
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            if rows % 128 != 0:
                cute.printf(
                    "tensor %d: First dimension of each tensor in a group must be divisible"
                    " by 128.\n",
                    tensor_id,
                )
            if cols % 128 != 0:
                cute.printf(
                    "tensor %d: For varying last dimensions support, the last dimension of each"
                    " tensor in a group must be divisible by 128.\n",
                    tensor_id,
                )

        mTensormaps[ROWS_OFFSET + tensor_id] = Int64(rows)
        mTensormaps[COLS_OFFSET + tensor_id] = Int64(cols)
        mTensormaps[OFFSETS_OFFSET + tensor_id] = base_offset

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        # Obtain the pointers of these descriptors
        desc_x = tmap.get_tensormap_ptr(
            mTensormaps.iterator
            + (INPUT_SLOT * MAX_SUPPORTED_TENSORS + tensor_id) * INT64_PER_TENSORMAP
        )
        desc_orow = tmap.get_tensormap_ptr(
            mTensormaps.iterator
            + (ROWWISE_OUTPUT_SLOT * MAX_SUPPORTED_TENSORS + tensor_id) * INT64_PER_TENSORMAP
        )
        desc_ocol = tmap.get_tensormap_ptr(
            mTensormaps.iterator
            + (COLWISE_OUTPUT_SLOT * MAX_SUPPORTED_TENSORS + tensor_id) * INT64_PER_TENSORMAP
        )
        desc_act = tmap.get_tensormap_ptr(
            mTensormaps.iterator
            + (ACT_INPUT_SLOT * MAX_SUPPORTED_TENSORS + tensor_id) * INT64_PER_TENSORMAP
        )

        # Zero-sized groups: creating a descriptor with a zero extent is invalid,
        # so skip (the main kernel skips these tensors as well).
        if rows > 0 and cols > 0:
            member_layout = cute.make_layout((rows, cols), stride=(cols, 1))

            def member_view(tensor: cute.Tensor, elt_dtype: Type[cutlass.Numeric]) -> cute.Tensor:
                return cute.make_tensor(
                    cute.make_ptr(
                        elt_dtype,
                        tensor.iterator.toint() + base_offset * (elt_dtype.width // 8),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    member_layout,
                )

            views = [member_view(mX, dtype)]
            atoms = [tma_atom_x]
            descs = [desc_x]
            tmap.init_tensormap_from_atom(tma_atom_x, desc_x, 0)
            if cutlass.const_expr(cfg.ROWWISE):
                views.append(member_view(mO_row, cfg.FP8_DTYPE))
                atoms.append(tma_atom_orow)
                tmap.init_tensormap_from_atom(tma_atom_orow, desc_orow, 0)
                descs.append(desc_orow)
            if cutlass.const_expr(cfg.COLWISE):
                views.append(member_view(mO_col, cfg.FP8_DTYPE))
                atoms.append(tma_atom_ocol)
                tmap.init_tensormap_from_atom(tma_atom_ocol, desc_ocol, 0)
                descs.append(desc_ocol)
            if cutlass.const_expr(cfg.WITH_DACT):
                views.append(member_view(mActInput, dtype))
                atoms.append(tma_atom_act)
                tmap.init_tensormap_from_atom(tma_atom_act, desc_act, 0)
                descs.append(desc_act)
            tmap.fence_tensormap_initialization()
            # Update descriptors in global memory with views and atoms
            tmap.update_tensormap(
                tuple(views),
                tuple(atoms),
                tuple(descs),
                0,
                (),  # smem staging is unused in GMEM update mode
            )

    @cute.jit
    def _make_shared_storage(
        self,
        smem: cutlass.Constexpr[cutlass.utils.SmemAllocator],
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    ) -> tuple[
        cute.Pointer,
        cute.Tensor,
        Optional[cute.Tensor],
        Optional[cute.Tensor],
        Optional[cute.Tensor],
        Optional[cute.Tensor],
    ]:
        """Allocate pipeline buffers and optional activation input and dbias storage."""
        FP8_DTYPE = self.cfg.FP8_DTYPE
        tile_layout = cute.make_layout(
            ((self.TILE_ROWS, self.TILE_COLS), self.PIPELINE_DEPTH),
            stride=((self.TILE_COLS, 1), self.TILE_ROWS * self.TILE_COLS),
        )

        sX = None
        sO_row = None
        sO_col = None
        sActInput = None

        if cutlass.const_expr(self.cfg.ROWWISE and self.cfg.COLWISE):

            @cute.struct
            class SharedStorage:
                mbar: cute.struct.MemRange[cute.Int64, 2 * self.PIPELINE_DEPTH]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]

            storage = smem.allocate(SharedStorage)
            sX = storage.sX.get_tensor(tile_layout)
            sO_row = storage.sO_row.get_tensor(tile_layout)
            sO_col = storage.sO_col.get_tensor(tile_layout)

        elif cutlass.const_expr(self.cfg.ROWWISE):

            @cute.struct
            class SharedStorage:
                mbar: cute.struct.MemRange[cute.Int64, 2 * self.PIPELINE_DEPTH]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]

            storage = smem.allocate(SharedStorage)
            sX = storage.sX.get_tensor(tile_layout)
            sO_row = storage.sO_row.get_tensor(tile_layout)

        elif cutlass.const_expr(self.cfg.COLWISE):

            @cute.struct
            class SharedStorage:
                mbar: cute.struct.MemRange[cute.Int64, 2 * self.PIPELINE_DEPTH]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]

            storage = smem.allocate(SharedStorage)
            sX = storage.sX.get_tensor(tile_layout)
            sO_col = storage.sO_col.get_tensor(tile_layout)

        if cutlass.const_expr(self.cfg.WITH_DACT):

            @cute.struct
            class DactStorage:
                sActInput: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]

            sActInput = smem.allocate(DactStorage).sActInput.get_tensor(tile_layout)

        sDbias = None
        if cutlass.const_expr(self.DBIAS_IN_ROWWISE):
            # (THREADS_Y, TILE_COLS) partial sums. Like the CUDA kernel's partial_dbias_rowwise,
            # every 32-column group is padded by one float so the groups fall in different banks.
            DBIAS_BUFF_WIDTH = self.THREADS_X * (MXFP8_BLOCK_SCALING_SIZE + 1)
            dbias_layout = cute.make_layout(
                (self.THREADS_Y, (MXFP8_BLOCK_SCALING_SIZE, self.THREADS_X)),
                stride=(DBIAS_BUFF_WIDTH, (1, MXFP8_BLOCK_SCALING_SIZE + 1)),
            )
            if cutlass.const_expr(self.ALIAS_DBIAS):
                # Every load of a dbias job is consumed before the reduction (one tile column
                # per job), so the input tiles are free then; the reduction ends with a sync
                # before the next job loads into them again.
                assert self.NUM_TILES_X == 1, "dbias reduction would overlap in-flight loads"
                sDbias = cute.make_tensor(cute.recast_ptr(sX.iterator, dtype=Float32), dbias_layout)
            else:

                @cute.struct
                class DbiasStorage:
                    sDbias: cute.struct.MemRange[Float32, self.THREADS_Y * DBIAS_BUFF_WIDTH]

                sDbias = smem.allocate(DbiasStorage).sDbias.get_tensor(dbias_layout)

        return storage.mbar.data_ptr(), sX, sO_row, sO_col, sActInput, sDbias

    @cute.kernel
    def kernel(
        self,
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mTensormaps: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mWorkspace: Optional[cute.Tensor],
        first_logical_dim: Int64,
        last_logical_dim: Int64,
        num_tensors: Int32,
        jobs_X: Optional[Int64],
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x: cute.CopyAtom,
        tma_src: cute.Tensor,
        tma_atom_act: Optional[cute.CopyAtom],
        tma_src_act: Optional[cute.Tensor],
        tma_atom_out_row: Optional[cute.CopyAtom],
        tma_dst_out_row: Optional[cute.Tensor],
        tma_atom_out_col: Optional[cute.CopyAtom],
        tma_dst_out_col: Optional[cute.Tensor],
    ) -> None:
        """No-op the CTA when the noop flag is set, else run the quantize main loop."""
        skip_execution = Boolean(False)
        if cutlass.const_expr(self.CHECK_NOOP_FLAG):
            skip_execution = noop_flag_is_set(mNoop)
        if not skip_execution:
            self._kernel_main(
                mS_row,
                mS_col,
                mOffsets,
                mFirstDims,
                mTensormaps,
                mWorkspace,
                first_logical_dim,
                last_logical_dim,
                num_tensors,
                jobs_X,
                dtype,
                tma_atom_x,
                tma_src,
                tma_atom_act,
                tma_src_act,
                tma_atom_out_row,
                tma_dst_out_row,
                tma_atom_out_col,
                tma_dst_out_col,
            )

    @cute.jit
    def _kernel_main(
        self,
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mTensormaps: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        first_logical_dim: Int64,
        last_logical_dim: Int64,
        num_tensors: Int32,
        jobs_X: Optional[Int64],
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x: cute.CopyAtom,
        tma_src: cute.Tensor,
        tma_atom_act: Optional[cute.CopyAtom],
        tma_src_act: Optional[cute.Tensor],
        tma_atom_out_row: Optional[cute.CopyAtom],
        tma_dst_out_row: Optional[cute.Tensor],
        tma_atom_out_col: Optional[cute.CopyAtom],
        tma_dst_out_col: Optional[cute.Tensor],
    ) -> None:
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
            # The first CTA validates every member's rows, as the CUDA kernel does. Like
            # NVTE_DEVICE_ERROR in a release build, this only prints.
            if bidx == 0:
                if tidx < num_tensors:
                    if Int64(mFirstDims[tidx]) % 128 != 0:
                        cute.printf(
                            "tensor %d: First dimension of each tensor in a group must be"
                            " divisible by 128.\n",
                            tidx,
                        )

        smem = cutlass.utils.SmemAllocator()
        # Layouts for sX, sO_row, sO_col, sActInput:
        # ((TILE_ROWS, TILE_COLS), PIPELINE_DEPTH):((TILE_COLS, 1), TILE_ROWS * TILE_COLS)
        # Layout for sDbias:
        # (THREADS_Y, TILE_COLS):(THREADS_X * (MXFP8_BLOCK_SCALING_SIZE + 1), 1)
        mbar_ptr, sX, sO_row, sO_col, sActInput, sDbias = self._make_shared_storage(smem, dtype)

        tx_count = self.TILE_ROWS * self.TILE_COLS * dtype.width // 8
        if cutlass.const_expr(cfg.WITH_DACT):
            tx_count *= 2
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self.PIPELINE_DEPTH,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.NUM_WARPS),
            tx_count=tx_count,
            cta_layout_vmnk=None,
        )
        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.PIPELINE_DEPTH
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.PIPELINE_DEPTH
        )

        gX_tiled = cute.zipped_divide(tma_src, (self.TILE_ROWS, self.TILE_COLS))
        tXsX, tXgX = cpasync.tma_partition(tma_atom_x, 0, cute.make_layout(1), sX, gX_tiled)

        tXsA = None
        tXgA = None
        if cutlass.const_expr(cfg.WITH_DACT):
            gA_tiled = cute.zipped_divide(tma_src_act, (self.TILE_ROWS, self.TILE_COLS))
            tXsA, tXgA = cpasync.tma_partition(
                tma_atom_act, 0, cute.make_layout(1), sActInput, gA_tiled
            )

        tXsO_row = None
        tXgO_row = None
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self.TILE_ROWS, self.TILE_COLS))
            tXsO_row, tXgO_row = cpasync.tma_partition(
                tma_atom_out_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
            )

        tXsO_col = None
        tXgO_col = None
        if cutlass.const_expr(cfg.COLWISE):
            gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self.TILE_ROWS, self.TILE_COLS))
            tXsO_col, tXgO_col = cpasync.tma_partition(
                tma_atom_out_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
            )

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        cute.arch.sync_threads()

        # If the CTA has work to do
        has_work = Boolean(True)
        # Metadata of the tensor that owns this job
        member_rows = Int64(0)
        member_cols = Int64(0)
        # Element offset of this tensor within the group: Int64 (CUDA uses size_t), since a
        # group can exceed 2^31 elements even when every individual extent is small.
        member_base = Int64(0)
        # Job's starting row and id in this individual tensor / global single tensor
        job_start_row = Int64(0)
        job_id_X = Int64(0)

        first_job_id = Int64(0)
        jobs_in_tensor = Int64(1)
        job_stride = Int64(1)
        jobs_X_in_tensor = Int64(1)

        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            # grid = [jobs_X * jobs_Y, 1, 1]
            job_id_Y = Int64(bidx) // jobs_X
            job_id_X = Int64(bidx) % jobs_X
            # All members share the stacked tensor's column count.
            member_cols = Int64(last_logical_dim)
            # Which row does this job start from
            job_start_row = job_id_Y * (self.TILE_ROWS * self.NUM_TILES_Y)
            if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
                total_elts = Int64(mOffsets[mOffsets.shape[0] - 1])
                # If the starting element is already beyond the last element of the group, this CTA can stop
                if Int64(job_start_row) * Int64(last_logical_dim) >= total_elts:
                    has_work = Boolean(False)
            # When SAME_BOTH_DIM, M is always divisible by 128, which is exactly TILE_ROWS * NUM_TILES_Y,
            # so no need to check for the last job's starting row being beyond the last row of the tensor.
        else:
            # grid = [workers_per_tensor, num_tensors, 1]
            tensor_id = Int32(bidy)
            # Extract tensor's metadata
            member_rows = Int64(mTensormaps[ROWS_OFFSET + tensor_id])
            member_cols = Int64(mTensormaps[COLS_OFFSET + tensor_id])
            member_base = Int64(mTensormaps[OFFSETS_OFFSET + tensor_id])
            if member_rows > 0 and member_cols > 0:
                # How many jobs does this tensor have in both directions
                jobs_X_in_tensor = cute.ceil_div(member_cols, (self.TILE_COLS * self.NUM_TILES_X))
                jobs_Y_in_tensor = cute.ceil_div(member_rows, (self.TILE_ROWS * self.NUM_TILES_Y))
                # How many jobs does this tensor have
                jobs_in_tensor = jobs_X_in_tensor * jobs_Y_in_tensor
                # Which job (1D index) does this CTA start from, which is also their worker ID
                first_job_id = Int64(bidx)
                # gdx is how many CTAs are assigned to this tensor
                job_stride = Int64(gdx)
                # If my first job is already beyond the tensor's last job, I have no work to do
                if first_job_id >= jobs_in_tensor:
                    has_work = Boolean(False)
            else:
                # This tensor is empty, so this CTA has no work to do
                has_work = Boolean(False)

        partitions = (tXsX, tXgX, tXsA, tXgA, tXsO_row, tXgO_row, tXsO_col, tXgO_col)
        atoms = (tma_atom_x, tma_atom_act, tma_atom_out_row, tma_atom_out_col)

        if has_work:
            # For single tensor case, each CTA only processes one job
            if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                # For single tensor case we don't use tensor descriptors
                descs = (None, None, None, None)

                # Find how many rows this tensor member has and which row the member starts from
                if cutlass.const_expr(cfg.COLWISE and cfg.WITH_GEMM_SWIZZLED_SCALES):
                    # If colwise scale is swizzled, we can't view the grouped tensor as one tensor as a whole
                    # in which case where the colwise is written to depends on the tensor member's first dimension,
                    # so we need to actually find out the start of the tensor member the job belongs to
                    if cutlass.const_expr(cfg.SHAPE_REP == SAME_BOTH_DIMS):
                        # If SAME_BOTH_DIMS, the tensor member's start offset and rows can be obtained with division
                        member_rows = Int64(first_logical_dim) // Int64(num_tensors)
                        member_start_row = job_start_row - job_start_row % member_rows
                    else:  # VARYING_FIRST_DIM
                        # Otherwise if VARYING_FIRST_DIM, we need to use find these tensor member metadata
                        # from mFirstDims & mOffsets
                        member_id = self._find_tensor_from_offsets(
                            mOffsets, num_tensors, job_start_row * member_cols
                        )
                        member_rows = Int64(mFirstDims[member_id])
                        member_start_row = Int64(mOffsets[member_id]) // member_cols
                else:
                    # If no swizzled colwise scale, quantizing a grouped tensor is equivalent to
                    # quantizing a single tensor as a whole as long as M % 128 == 0 holds,
                    # so we just pretend the whole tensor is the tensor member this CTA is working on
                    member_rows = Int64(first_logical_dim)
                    member_start_row = Int64(0)

                # The starting offset for this tensor member's scale
                member_scale_base = (
                    member_start_row * cute.round_up(member_cols, 128) // MXFP8_BLOCK_SCALING_SIZE
                )

                row_scales = None
                # Starting data row of this job relative to the member
                job_start_row_from_member = job_start_row - member_start_row
                job_start_col_from_member = job_id_X * self.TILE_COLS
                if cutlass.const_expr(cfg.ROWWISE):
                    row_scales = self._rowwise_scales(
                        mS_row, member_scale_base, member_rows, member_cols
                    )

                col_scales = None
                if cutlass.const_expr(cfg.COLWISE):
                    col_scales = self._colwise_scales(
                        mS_col, member_scale_base, member_rows, member_cols
                    )
                cute.arch.sync_threads()

                self._process_job(
                    member_start_row,
                    job_start_row_from_member,
                    job_start_col_from_member,
                    member_rows,
                    member_cols,  # all tensors have the same last dimension in single tensor case
                    row_scales,
                    col_scales,
                    mWorkspace,
                    sDbias,
                    descs,
                    tmap,
                    warp_idx,
                    tidx,
                    sX,
                    sActInput,
                    sO_row,
                    sO_col,
                    partitions,
                    atoms,
                    mainloop_pipeline,
                    prod_state,
                    cons_state,
                )
            else:
                # For non-single tensor case, we use persistent kernel so each CTA keeps processing jobs until none is left
                desc_x = tmap.get_tensormap_ptr(
                    mTensormaps.iterator
                    + (INPUT_SLOT * MAX_SUPPORTED_TENSORS + tensor_id) * INT64_PER_TENSORMAP
                )
                desc_out_row = tmap.get_tensormap_ptr(
                    mTensormaps.iterator
                    + (ROWWISE_OUTPUT_SLOT * MAX_SUPPORTED_TENSORS + tensor_id)
                    * INT64_PER_TENSORMAP
                )
                desc_out_col = tmap.get_tensormap_ptr(
                    mTensormaps.iterator
                    + (COLWISE_OUTPUT_SLOT * MAX_SUPPORTED_TENSORS + tensor_id)
                    * INT64_PER_TENSORMAP
                )
                desc_act = tmap.get_tensormap_ptr(
                    mTensormaps.iterator
                    + (ACT_INPUT_SLOT * MAX_SUPPORTED_TENSORS + tensor_id) * INT64_PER_TENSORMAP
                )

                if tidx == 0:
                    tmap.fence_tensormap_update(desc_x)
                    if cutlass.const_expr(cfg.WITH_DACT):
                        tmap.fence_tensormap_update(desc_act)
                    if cutlass.const_expr(cfg.ROWWISE):
                        tmap.fence_tensormap_update(desc_out_row)
                    if cutlass.const_expr(cfg.COLWISE):
                        tmap.fence_tensormap_update(desc_out_col)
                descs = (desc_x, desc_act, desc_out_row, desc_out_col)

                # This member's scales start at member_base / 32 in both directions.
                member_scale_base = member_base // Int64(MXFP8_BLOCK_SCALING_SIZE)
                row_scales = None
                if cutlass.const_expr(cfg.ROWWISE):
                    row_scales = self._rowwise_scales(
                        mS_row, member_scale_base, member_rows, member_cols
                    )
                col_scales = None
                if cutlass.const_expr(cfg.COLWISE):
                    col_scales = self._colwise_scales(
                        mS_col, member_scale_base, member_rows, member_cols
                    )

                # Make sure all threads see the updated descriptors and scales before processing any jobs.
                cute.arch.sync_threads()

                # Grid-stride over this tensor's own jobs; the descriptors never change.
                job_id = first_job_id
                job_finished = Boolean(False)
                while not job_finished:
                    job_id_Y_in_tensor = job_id // jobs_X_in_tensor
                    job_id_X_in_tensor = job_id % jobs_X_in_tensor
                    job_start_row_from_member = job_id_Y_in_tensor * (
                        self.TILE_ROWS * self.NUM_TILES_Y
                    )
                    job_start_col_from_member = job_id_X_in_tensor * (
                        self.TILE_COLS * self.NUM_TILES_X
                    )
                    self._process_job(
                        Int64(0),  # For non-single tensor case, each TMA loads one member
                        job_start_row_from_member,
                        job_start_col_from_member,
                        member_rows,
                        member_cols,
                        row_scales,
                        col_scales,
                        mWorkspace,
                        sDbias,
                        descs,
                        tmap,
                        warp_idx,
                        tidx,
                        sX,
                        sActInput,
                        sO_row,
                        sO_col,
                        partitions,
                        atoms,
                        mainloop_pipeline,
                        prod_state,
                        cons_state,
                    )
                    # Find the next job to process
                    job_id = job_id + job_stride
                    if job_id >= jobs_in_tensor:
                        job_finished = Boolean(True)

        # Drain every TMA store before the CTA releases its shared-memory source buffers.
        if warp_idx == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=False)
        cute.arch.sync_threads()

    def _issue_load(
        self,
        pipeline_obj: pipeline.PipelineTmaAsync,
        prod_state: pipeline.PipelineState,
        tile_y: Int64,
        tile_x: Int64,
        atoms: tuple[Optional[cute.CopyAtom], ...],
        partitions: tuple[Optional[cute.Tensor], ...],
        tmap: TensorMapManager,
        descs: tuple[Optional[cute.Pointer], ...],
    ) -> None:
        """Emit the 32x128 TMA load(s) of one stage into the current pipeline buffer.

        Caller gates this on warp 0 and advances `prod_state` afterwards -- the advance
        must happen outside the gate or the mutated SSA values stay trapped in the scf.if.
        """
        tma_atom_x, tma_atom_act, _, _ = atoms
        tXsX, tXgX, tXsA, tXgA, _, _, _, _ = partitions
        desc_x, desc_act, _, _ = descs
        # Wait for the consumer to finish using this SMEM buffer
        pipeline_obj.producer_acquire(prod_state)
        barrier = pipeline_obj.producer_get_barrier(prod_state)
        loads = [(tma_atom_x, tXgX, tXsX, desc_x)]
        if cutlass.const_expr(self.cfg.WITH_DACT):
            loads.append((tma_atom_act, tXgA, tXsA, desc_act))
        for atom, tXg, tXs, desc in loads:
            if cutlass.const_expr(self.cfg.IS_SINGLE_TENSOR):
                cute.copy(
                    atom,
                    tXg[(None, (tile_y, tile_x))],
                    tXs[(None, prod_state.index)],
                    tma_bar_ptr=barrier,
                )
            else:
                # Every member shares tXg's tile-coordinate arithmetic (the coefficients are
                # just the tile size); tma_desc_ptr supplies this member's geometry.
                cute.copy(
                    atom,
                    tXg[(None, (tile_y, tile_x))],
                    tXs[(None, prod_state.index)],
                    tma_bar_ptr=barrier,
                    tma_desc_ptr=tmap.get_tensormap_ptr(desc, cute.AddressSpace.generic),
                )
        # Notify the consumer that this SMEM buffer is ready for consumption
        pipeline_obj.producer_commit(prod_state)

    @cute.jit
    def _process_job(
        self,
        member_start_row: Int64,  # Which row the member starts from the tensor loaded by TMA
        job_start_row_from_member: Int64,  # Rows between this job and the start of this member
        job_start_col_from_member: Int64,  # Columns between this job and the start of this member
        member_rows: Int64,  # Number of data rows in this member
        member_cols: Int64,  # Number of columns in this member
        row_scales: Optional[cute.Tensor],  # Member-local rowwise scales tiled per stage
        col_scales: Optional[cute.Tensor],  # Colwise scales tiled per stage
        mWorkspace: Optional[cute.Tensor],  # f32 partial dbias workspace (WITH_DBIAS)
        sDbias: Optional[
            cute.Tensor
        ],  # SMEM buffer for the rowwise dbias reduction (rowwise-only dbias)
        descs: tuple[
            Optional[cute.Pointer], ...
        ],  # Per-tensor descriptors (x, act, out_row, out_col), None if single-tensor
        tmap: TensorMapManager,  # TensorMapManager for managing TMA descriptors
        warp_idx: Int32,
        tidx: Int32,
        sX: cute.Tensor,  # SMEM input ring
        sActInput: Optional[cute.Tensor],  # SMEM activation input ring (WITH_DACT)
        sO_row: Optional[cute.Tensor],  # SMEM rowwise output ring
        sO_col: Optional[cute.Tensor],  # SMEM colwise output ring
        partitions: tuple[Optional[cute.Tensor], ...],  # TMA partitions (x, act, out_row, out_col)
        atoms: tuple[Optional[cute.CopyAtom], ...],  # TMA atoms (x, act, out_row, out_col)
        mainloop_pipeline: cutlass.pipeline.PipelineTmaAsync,
        prod_state: pipeline.PipelineState,
        cons_state: pipeline.PipelineState,
    ) -> None:
        """Quantize a job with one continuous pipeline across its column and row tiles."""
        cfg = self.cfg
        _, _, tma_atom_out_row, tma_atom_out_col = atoms
        _, _, _, _, tXsO_row, tXgO_row, tXsO_col, tXgO_col = partitions
        _, _, desc_out_row, desc_out_col = descs

        job_start_row = member_start_row + job_start_row_from_member
        job_tile_Y = job_start_row // self.TILE_ROWS
        job_tile_X = job_start_col_from_member // self.TILE_COLS
        member_tile_Y = job_start_row_from_member // self.TILE_ROWS
        tiles_X = cutlass.min(
            Int64(self.NUM_TILES_X),
            cute.ceil_div(member_cols - job_start_col_from_member, self.TILE_COLS),
        )
        num_tiles = tiles_X * self.NUM_TILES_Y

        # Fill every buffer up front, then issue one more each time a stage is consumed.
        for prologue_stage in cutlass.range_constexpr(self.PIPELINE_DEPTH):
            if warp_idx == 0:
                self._issue_load(
                    mainloop_pipeline,
                    prod_state,
                    job_tile_Y + prologue_stage % self.NUM_TILES_Y,
                    job_tile_X + prologue_stage // self.NUM_TILES_Y,
                    atoms,
                    partitions,
                    tmap,
                    descs,
                )
            prod_state.advance()

        for tile_X in cutlass.range(tiles_X, unroll=1):
            tile_id_X = job_tile_X + tile_X
            tile_start_col = job_start_col_from_member + tile_X * self.TILE_COLS
            partial_dbias = Float32(0.0)
            dbias_row_acc = None
            if cutlass.const_expr(self.DBIAS_IN_ROWWISE):
                dbias_row_acc = cute.make_rmem_tensor(
                    layout_or_shape=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE,), stride=(1,)),
                    dtype=Float32,
                )
                for c in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                    dbias_row_acc[c] = Float32(0.0)

            for tile_Y in cutlass.range_constexpr(self.NUM_TILES_Y):
                stage = tile_X * self.NUM_TILES_Y + tile_Y
                # Wait for at most DEPTH-1 iters on the fly, which means the the last DEPTH iter has finished
                # so we can reuse its SMEM output buffer
                # (input buffer is managed by the producer and consumer pipeline states)
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(self.PIPELINE_DEPTH - 1, read=True)
                # Wait for this stage's input buffer to be filled by the producer
                mainloop_pipeline.consumer_wait(cons_state)
                cute.arch.sync_threads()
                sX_tile = sX[(None, cons_state.index)]
                sAct_tile = None
                if cutlass.const_expr(cfg.WITH_DACT):
                    sAct_tile = sActInput[(None, cons_state.index)]
                row_tile = job_tile_Y + tile_Y

                if cutlass.const_expr(cfg.COLWISE):
                    _, partial_dbias = _quantize_colwise_mxfp8(
                        sX_tile,
                        sAct_tile,
                        sO_col[(None, cons_state.index)],
                        cute.flatten(col_scales[(None, (member_tile_Y + tile_Y, tile_id_X))]),
                        cfg.MAX_NORM_RCP,
                        (member_tile_Y + tile_Y) * self.TILE_ROWS,
                        tile_start_col,
                        member_rows,
                        member_cols,
                        ACTIVATION=cfg.ACTIVATION,
                        DTYPE=cfg.DTYPE,
                        FP8_DTYPE=cfg.FP8_DTYPE,
                        SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
                        TILE_X=self.TILE_COLS,
                        TILE_Y=self.TILE_ROWS,
                        WITH_ACT=cfg.WITH_ACT,
                        WITH_DACT=cfg.WITH_DACT,
                        WITH_DBIAS=self.DBIAS_IN_COLWISE,
                        CACHE_ACTIVATION=self.CACHE_ACTIVATION,
                        ZERO_OOB_SCALES=True,
                        dbias_init=partial_dbias,
                    )
                if cutlass.const_expr(self.CACHE_ACTIVATION):
                    # The rowwise pass reads the activation the colwise pass cached in sX.
                    cute.arch.sync_threads()

                if cutlass.const_expr(cfg.ROWWISE):
                    _quantize_rowwise_mxfp8(
                        sX_tile,
                        None if self.CACHE_ACTIVATION else sAct_tile,
                        sO_row[(None, cons_state.index)],
                        cute.flatten(row_scales[(None, (member_tile_Y + tile_Y, tile_id_X))]),
                        cfg.MAX_NORM_RCP,
                        (member_tile_Y + tile_Y) * self.TILE_ROWS,
                        tile_start_col,
                        member_rows,
                        member_cols,
                        ACTIVATION=None if self.CACHE_ACTIVATION else cfg.ACTIVATION,
                        DTYPE=cfg.DTYPE,
                        FP8_DTYPE=cfg.FP8_DTYPE,
                        TILE_X=self.TILE_COLS,
                        TILE_Y=self.TILE_ROWS,
                        WITH_ACT=cfg.WITH_ACT and not self.CACHE_ACTIVATION,
                        WITH_DACT=cfg.WITH_DACT and not self.CACHE_ACTIVATION,
                        WITH_DBIAS=self.DBIAS_IN_ROWWISE,
                        dbias_acc=dbias_row_acc,
                        ZERO_OOB_SCALES=True,
                    )

                # Force consumer's write to SMEM to be visible to TMA stores later
                cute.arch.fence_proxy("async.shared", space="cta")
                # Only after everyone finishes computation then this stage can be considered as "consumed"
                cute.arch.sync_threads()
                # I'm done with my input SMEM buffer, so the producer can write the next stage's data into it
                mainloop_pipeline.consumer_release(cons_state)

                # I just freed my input SMEM buffer (stage), so the producer now can use it for writing
                # (stage+DEPTH) stage's data if that stage exists
                if stage + self.PIPELINE_DEPTH < num_tiles:
                    if warp_idx == 0:
                        self._issue_load(
                            mainloop_pipeline,
                            prod_state,
                            job_tile_Y + (stage + self.PIPELINE_DEPTH) % self.NUM_TILES_Y,
                            job_tile_X + (stage + self.PIPELINE_DEPTH) // self.NUM_TILES_Y,
                            atoms,
                            partitions,
                            tmap,
                            descs,
                        )
                    prod_state.advance()

                # Write result to GMEM via TMA
                if warp_idx == 0:
                    stores = []
                    if cutlass.const_expr(cfg.ROWWISE):
                        stores.append((tma_atom_out_row, tXsO_row, tXgO_row, desc_out_row))
                    if cutlass.const_expr(cfg.COLWISE):
                        stores.append((tma_atom_out_col, tXsO_col, tXgO_col, desc_out_col))
                    for atom, tXs, tXg, desc in stores:
                        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                            cute.copy(
                                atom,
                                tXs[(None, cons_state.index)],
                                tXg[(None, (row_tile, tile_id_X))],
                            )
                        else:
                            cute.copy(
                                atom,
                                tXs[(None, cons_state.index)],
                                tXg[(None, (row_tile, tile_id_X))],
                                tma_desc_ptr=tmap.get_tensormap_ptr(
                                    desc, cute.AddressSpace.generic
                                ),
                            )
                    # Commit all TMA operations of this iteration
                    cute.arch.cp_async_bulk_commit_group()

                cons_state.advance()

            if cutlass.const_expr(cfg.WITH_DBIAS):
                if cutlass.const_expr(self.DBIAS_IN_ROWWISE):
                    partial_dbias = reduce_rowwise_dbias(
                        sDbias,
                        tidx,
                        dbias_row_acc,
                        self.TILE_ROWS,
                        self.TILE_COLS,
                    )
                    # All threads must finish reading before the next job rewrites the buffer.
                    cute.arch.sync_threads()

                # A job has TILE_ROWS * NUM_TILES_Y rows, and dbias reduces them to one row
                dbias_row = job_start_row // (self.TILE_ROWS * self.NUM_TILES_Y)
                dbias_col = tile_start_col + tidx
                if dbias_col < member_cols:
                    mWorkspace[(dbias_row, dbias_col)] = partial_dbias
