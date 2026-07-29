# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVFP4 quantization kernel implemented in CuTeDSL.

Replicates the core logic of quantize_transpose_nvfp4_tuned_1D.cuh: given a 2D tensor of BF16
values, quantize to NVFP4 (FP4E2M1 data + E4M3 per-block scales) and optionally also emit the
transposed (columnwise-scaled) output.

Main kernel logic resides in NVFP4QuantizeTransposeTuned1DKernel.

"""

import logging
import os
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int64, Float4E2M1FN, Float8E4M3FN, BFloat16
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import device_is_blackwell

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.nvfp4")

# Number of elements per NVFP4 scale block (they share one E4M3 scale factor).
NVFP4_BLOCK_SCALING_SIZE = 16
# Input row/col divisibility the CUDA tuned-1D kernel requires (16B TMA alignment).
NVFP4_SHAPE_ALIGNMENT = 32
# Padding TE applies to the scale tensors' outer / inner dim (NVFP4Quantizer::get_scale_shape).
NVFP4_SCALE_PAD_OUTER = 128
NVFP4_SCALE_PAD_INNER = 4


class NVFP4QuantizeConfig:
    """Instantiation parameters of the CuTE DSL kernel"""

    def __init__(
        self,
        use_stochastic_rounding: bool,
        use_fast_math: bool,
        row_scaled_nvfp4: bool,
        return_transpose: bool,
    ):
        self.USE_STOCHASTIC_ROUNDING = use_stochastic_rounding
        self.USE_FAST_MATH = use_fast_math
        if row_scaled_nvfp4 and return_transpose:
            raise ValueError("row-scaled NVFP4 quantization does not produce a transposed output")
        self.ROW_SCALED_NVFP4 = row_scaled_nvfp4
        self.RETURN_TRANSPOSE = return_transpose

    def __str__(self):
        return (
            f"NVFP4QuantizeConfig(use_stochastic_rounding={self.USE_STOCHASTIC_ROUNDING}, "
            f"use_fast_math={self.USE_FAST_MATH}, "
            f"row_scaled_nvfp4={self.ROW_SCALED_NVFP4}, "
            f"return_transpose={self.RETURN_TRANSPOSE})"
        )

    __repr__ = __str__

# Runs if CUTE_DSL_ENABLE_ASSERTIONS=1 or --enable-assertions present in cute.compile
def validate_tensor(tensor: Optional[cute.Tensor], expected_layout: cute.Layout, expected_dtype):
    if tensor is None:
        return
    cute.testing.assert_(tensor.layout == expected_layout, "Tensor layout does not match")
    cute.testing.assert_(tensor.dtype == expected_dtype, "Tensor dtype does not match")

@cute.jit
def noop_flag_is_set(mNoop: cute.Pointer) -> Boolean:
    """Whether the cast_noop flag says this quantization is a no-op and must be skipped.

    mNoop is a pointer rather than a tensor so that one compiled kernel serves both a present and
    an absent flag, hence the address is checked before it is dereferenced, exactly like the CUDA
    C++ kernel's `noop != nullptr && noop[0] == 1.0f`. The two checks cannot be joined with `and`,
    which the DSL lowers to a non-short-circuiting op that would load from the null pointer.
    """
    flag_is_set = Boolean(False)
    if mNoop.toint() != Int64(0):
        flag_is_set = cute.make_tensor(mNoop, cute.make_layout((1,)))[0] == Float32(1.0)
    return flag_is_set

class NVFP4QuantizeTransposeTuned1DKernel:
    """Tuned kernel to cast to NVFP4 and transpose"""

    # Each thread block processes a _chunk_ of the input tensor, which is a 2D sub-tensor.
    # The quantization this kernel performs is almost a pure point-wise operation, except that
    # NVFP4_BLOCK_SCALING_SIZE elements share a single block scaling factor and the entire tensor
    # has a global scaling factor.
    # Therefore, the chunk also correponds to a 2D sub-tensor of all the other tensors.
    #
    # A chunk is made of multiple _tiles_, which are processed sequentially by the thread block.
    # While the tiles are processed sequentially, there may be more than one tile in-flight at a time.
    # PREFETCH_STAGES specifies how many tiles to prefetch to SMEM, in addition to the current tile.
    #
    # Optionally, the kernel can be PERSISTENT, in which case it will use Cluster Launch Control to
    # achieve dynamic persistent tile scheduling, through work stealing.

    # Tunable config (mirroring the CUDA version)
    CHUNK_DIM_Y = 128
    CHUNK_DIM_X = 128
    PREFETCH_STAGES = 1
    PERSISTENT = False

    THREADS_NUM = 128
    ELTS_PER_THREAD = NVFP4_BLOCK_SCALING_SIZE
    TILE_DIM_Y = 64
    TILE_DIM_X = 64

    def __init__(self, cfg):
        self.cfg = cfg

    # Host-side kernel launch
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, N) bf16 input
        mO_row: cute.Tensor,  # (M, N) fp4 rowwise output
        mS_row: cute.Tensor,  # (roundup(M, 128), roundup(ceil(N / 16), 4)) e4m3 scales
        mO_col: Optional[cute.Tensor],  # (N, M) fp4 transposed output
        mS_col: Optional[cute.Tensor],  # (roundup(N, 128), roundup(ceil(M / 16), 4)) e4m3 scales
        mAmaxRow: cute.Tensor,  # (M,) f32 per-row amax if ROW_SCALED_NVFP4, else (1,) global amax
        mAmaxCol: Optional[cute.Tensor],  # (1,) f32 global amax of the transposed output
        mNoop: cute.Pointer,  # f32 cast-noop flag; may be null, checked on device
        mRngState: Optional[cute.Tensor],  # (2,) i64 Philox {seed, offset}
        stream: CUstream,
    ):
        """AOT-compiled host entrypoint. `quantize_transpose_nvfp4_cutedsl.cuh` passes these
        arguments in this exact order via tvm-ffi, and the config fixes which of the optional
        ones are present (see `compile_cutedsl_function_from_cfg`).

        M, N are the input's *flattened* 2D dims, both multiples of NVFP4_SHAPE_ALIGNMENT; a
        rank > 2 input already arrives flattened.
        All tensors are row-major (with rightmost stride 1).
        FP4 extents are logical element counts, not the halved extents of the uint8 buffer TE
        actually allocates.
        """
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] NVFP4QuantizeTransposeTuned1DKernel.__call__() with config:"
                f" {self.cfg}\n"
            )

        ## Validation

        # Validate input and output tensor layouts
        (M, N) = mX.shape
        mX_layout = cute.make_ordered_layout((M, N), order=(1, 0))
        mO_row_layout = mX_layout
        mO_col_layout = cute.make_ordered_layout((N, M), order=(1, 0))
        validate_tensor(mX, mX_layout, BFloat16)
        validate_tensor(mO_row, mO_row_layout, Float4E2M1FN)
        validate_tensor(mO_col, mO_col_layout, Float4E2M1FN)
        
        # Validate scaling factor tensor layouts
        mS_row_layout = cute.make_ordered_layout(
            (cute.round_up(M, NVFP4_SCALE_PAD_OUTER), cute.round_up(cute.ceil_div(N, NVFP4_BLOCK_SCALING_SIZE), NVFP4_SCALE_PAD_INNER)),
            order=(1, 0)
        )
        mS_col_layout = cute.make_ordered_layout(
            (cute.round_up(N, NVFP4_SCALE_PAD_OUTER), cute.round_up(cute.ceil_div(M, NVFP4_BLOCK_SCALING_SIZE), NVFP4_SCALE_PAD_INNER)),
            order=(1, 0)
        )
        validate_tensor(mS_row, mS_row_layout, Float8E4M3FN)
        validate_tensor(mS_col, mS_col_layout, Float8E4M3FN)

        # Validate amax tensor layouts
        if cutlass.const_expr(self.cfg.row_scaled_nvfp4):
            mAmaxRow_layout = cute.make_layout((M,))
            validate_tensor(mAmaxRow, mAmaxRow_layout, Float32)
        else:
            mAmaxRow_layout = cute.make_layout((1,))
            validate_tensor(mAmaxRow, mAmaxRow_layout, Float32)
        mAmaxCol_layout = cute.make_layout((1,))
        validate_tensor(mAmaxCol, mAmaxCol_layout, Float32)

        # Validate RNG state tensor layout
        mRngState_layout = cute.make_layout((2,))
        validate_tensor(mRngState, mRngState_layout, Int64)
        
        ## Grid and block size calculation
        chunk_shape = (self.CHUNK_DIM_Y, self.CHUNK_DIM_X)
        grid = cute.ceil_div(mX.shape, chunk_shape)
        block = (self.THREADS_NUM, 1, 1)

        # todo: setup TMA atoms

        self.kernel(
            mX,
            mO_row,
            mS_row,
            mO_col,
            mS_col,
            mAmaxRow,
            mAmaxCol,
            mNoop,
            mRngState,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N) bf16 input
        mO_row: cute.Tensor,  # (M, N) fp4 rowwise output
        mS_row: cute.Tensor,  # (roundup(M, 128), roundup(ceil(N / 16), 4)) e4m3 scales
        mO_col: Optional[cute.Tensor],  # (N, M) fp4 transposed output
        mS_col: Optional[cute.Tensor],  # (roundup(N, 128), roundup(ceil(M / 16), 4)) e4m3 scales
        mAmaxRow: cute.Tensor,  # (M,) f32 per-row amax if ROW_SCALED_NVFP4, else (1,) global amax
        mAmaxCol: Optional[cute.Tensor],  # (1,) f32 global amax of the transposed output
        mNoop: cute.Pointer,  # f32 cast-noop flag; may be null, checked on device
        mRngState: Optional[cute.Tensor],  # (2,) i64 Philox {seed, offset}
        stream: CUstream,
    ):
        """Device entry for the NVFP4 tuned-1D quantize-transpose kernel."""
        if not noop_flag_is_set(mNoop):
            ...

def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given NVFP4 quantization config.

    The fake tensors below are what fix the compiled ABI, so they must agree with `__call__`'s
    signature.
    """

    if not device_is_blackwell():
        raise RuntimeError("CuTeDSL NVFP4 backend requires compute capability >= 10.0 (Blackwell)")

    kernel_obj = NVFP4QuantizeTransposeTuned1DKernel(cfg)

    def _gmem(dtype, shape, stride_order, align):
        """Fake row-major GMEM tensor; `align` is the assumed base alignment in bytes."""
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=stride_order,
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    # Flattened 2D input dims. Their divisibility is also what keeps every row stride a multiple
    # of 16B as TMA requires: 2*N bytes for the bf16 input, N/2 and M/2 for the fp4 outputs.
    sym_M = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)
    sym_N = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)

    # Scale extents are TE's padded ones (NVFP4Quantizer::get_scale_shape):
    #   mS_row: (roundup(M, 128), roundup(ceil(N / 16), 4))
    #   mS_col: (roundup(N, 128), roundup(ceil(M / 16), 4))
    # Neither is expressible in sym_M / sym_N (SymInt has no `//` or `+`), so the scales get
    # fresh syms carrying only the divisibility the padding guarantees.
    scale_row_shape = (
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_OUTER),
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_INNER),
    )
    scale_col_shape = (
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_OUTER),
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_INNER),
    )

    # mX / mO_row / mS_row: (M, N) input, (M, N) rowwise fp4 output and its scales. The fp4
    # extents are logical element counts, so the row stride is N while only N/2 bytes are
    # stored.
    in_fake = _gmem(BFloat16, (sym_M, sym_N), stride_order=(1, 0), align=16)
    out_row_fake = _gmem(Float4E2M1FN, (sym_M, sym_N), stride_order=(1, 0), align=16)
    scale_row_fake = _gmem(Float8E4M3FN, scale_row_shape, stride_order=(1, 0), align=4)

    # mO_col / mS_col: (N, M) transposed fp4 output and its scales, present iff a transpose is
    # produced.
    out_col_fake = (
        _gmem(Float4E2M1FN, (sym_N, sym_M), stride_order=(1, 0), align=16)
        if cfg.RETURN_TRANSPOSE
        else None
    )
    scale_col_fake = (
        _gmem(Float8E4M3FN, scale_col_shape, stride_order=(1, 0), align=4)
        if cfg.RETURN_TRANSPOSE
        else None
    )

    # mAmaxRow: unpadded per-row amax (M,) when row-scaled, else a single global amax (1,).
    amax_row_fake = (
        _gmem(Float32, (sym_M,), stride_order=(0,), align=4)
        if cfg.ROW_SCALED_NVFP4
        else _gmem(Float32, (1,), stride_order=(0,), align=4)
    )
    # mAmaxCol: (1,) global amax of the transposed output, present iff a transpose is produced.
    amax_col_fake = (
        _gmem(Float32, (1,), stride_order=(0,), align=4) if cfg.RETURN_TRANSPOSE else None
    )

    # mNoop: always-present f32 pointer to the single-element cast-noop flag. A pointer rather
    # than a tensor so the runtime value can be null without changing the compiled ABI -- the
    # kernel does the null-check and the noop[0] == 1.0f test on device. The null address here
    # is only a compile-time placeholder for the pointer the dispatcher passes at launch.
    noop_fake = cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4)

    # mRngState: Philox {seed, offset}; present (and consumed) only for stochastic rounding.
    rng_state_fake = (
        _gmem(Int64, (2,), stride_order=(0,), align=8) if cfg.USE_STOCHASTIC_ROUNDING else None
    )

    compiled = cute.compile(
        kernel_obj,
        in_fake,  # mX
        out_row_fake,  # mO_row
        scale_row_fake,  # mS_row
        out_col_fake,  # mO_col
        scale_col_fake,  # mS_col
        amax_row_fake,  # mAmaxRow
        amax_col_fake,  # mAmaxCol
        noop_fake,  # mNoop
        rng_state_fake,  # mRngState
        cute.runtime.make_fake_stream(),  # stream
        options="--enable-tvm-ffi",
    )
    return compiled


def get_nvfp4_quantization_function(
    fn_name: str,
    use_stochastic_rounding: bool,
    use_fast_math: bool,
    row_scaled_nvfp4: bool,
    return_transpose: bool,
) -> bool:
    """Compile the NVFP4 quantize kernel for this config and register it in the TVM-FFI global
    registry under EXACTLY `fn_name` (the key the C++ dispatcher built). Returns True if a kernel
    is registered under `fn_name`, False if the config is unsupported or compilation failed, in
    which case the caller caches the negative result and falls back to the CUDA C++ kernel.
    """

    # Already registered (e.g. by a prior call) -> supported.
    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True

    try:
        cfg = NVFP4QuantizeConfig(
            use_stochastic_rounding=use_stochastic_rounding,
            use_fast_math=use_fast_math,
            row_scaled_nvfp4=row_scaled_nvfp4,
            return_transpose=return_transpose,
        )
    except ValueError as e:
        # The exception message states exactly why the config is unsupported. Surfacing it as a
        # warning lets the C++ dispatcher's CUDA fallback be recognized as expected.
        logger.warning(
            "CuTeDSL NVFP4 backend does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

    logger.debug("Compiling CuTeDSL NVFP4 quantization kernel for %s", cfg)
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "CuTeDSL NVFP4 kernel compilation failed, falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    tvm_ffi.register_global_func(fn_name, compiled, override=True)

    return True
