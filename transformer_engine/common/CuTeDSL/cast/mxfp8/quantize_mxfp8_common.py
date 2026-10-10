# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared definitions, kernel dispatch and TVM-FFI registration for CuTeDSL MXFP8."""

import abc
import logging
import os
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    str_to_cutlass_dtype,
    abs_max_x2_bf16,
    device_compute_capability,
    is_packed16,
)
from transformer_engine.common.CuTeDSL.activations import (
    act_relu,
    act_gelu,
    act_silu,
    act_qgelu,
    act_srelu,
    dact_drelu,
    dact_dsrelu,
    dact_dsilu,
    dact_dqgelu,
    dact_dgelu,
)

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

# Number of elements per MXFP8 scale block. They will share the same E8M0 scale factor
MXFP8_BLOCK_SCALING_SIZE = 32
# How many threads are in one warp
THREADS_PER_WARP = 32

# FP8E4M3 max representable value
FP8E4M3_MAX_NORM = 448.0
FP8E4M3_MAX_NORM_RCP = 1.0 / FP8E4M3_MAX_NORM
FP8E5M2_MAX_NORM = 57344.0
FP8E5M2_MAX_NORM_RCP = 1.0 / FP8E5M2_MAX_NORM

# If N is not divisible by 16, writing quantized FP8 output with N bytes per row will fail because
# TMA requires 16-byte alignment for each row
SYM_N_DIVISIBILITY = 16

SUPPORTED_ACTIVATIONS = {
    "relu": act_relu,
    "gelu": act_gelu,
    "silu": act_silu,
    "qgelu": act_qgelu,
    "srelu": act_srelu,
}

SUPPORTED_DACTIVATIONS = {
    "drelu": dact_drelu,
    "dgelu": dact_dgelu,
    "dsilu": dact_dsilu,
    "dqgelu": dact_dqgelu,
    "dsrelu": dact_dsrelu,
}


@cute.jit
def derive_swizzled_scale_layout(
    M: Int32,
    N: Int32,
    ROWWISE: cutlass.Constexpr[bool],
    COLWISE: cutlass.Constexpr[bool],
    mS_row: Optional[cute.Tensor],
    mS_col: Optional[cute.Tensor],
):
    """Derive the swizzled layout for the rowwise and colwise scale tensors."""
    num_scale_cols = cute.ceil_div(N, MXFP8_BLOCK_SCALING_SIZE)
    num_scale_rows = cute.ceil_div(M, MXFP8_BLOCK_SCALING_SIZE)

    num_tiles_M = cute.ceil_div(M, 128)
    num_tiles_SC = cute.ceil_div(num_scale_cols, 4)
    num_tiles_SR = cute.ceil_div(num_scale_rows, 4)
    num_tiles_N = cute.ceil_div(N, 128)

    if cutlass.const_expr(ROWWISE):
        mS_row = cute.make_tensor(
            mS_row.iterator,
            cute.make_layout(
                ((32, 4, num_tiles_M), (4, num_tiles_SC)),
                stride=((16, 4, num_tiles_SC * 512), (1, 512)),
            ),
        )
    if cutlass.const_expr(COLWISE):
        mS_col = cute.make_tensor(
            mS_col.iterator,
            cute.make_layout(
                ((4, num_tiles_SR), (32, 4, num_tiles_N)),
                stride=((1, 512), (16, 4, num_tiles_SR * 512)),
            ),
        )
    return mS_row, mS_col


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


class MXFP8QuantizeConfig:
    """Configs for the compiled CuTeDSL kernel. These will be fixed once the kernel is compiled and
    they will behave as const expressions.
    """

    def __init__(
        self,
        dtype: str,
        fp8_dtype: str,
        rowwise: bool,
        colwise: bool,
        with_gemm_swizzled_scales: bool,
        with_amax: bool,
        with_dbias: bool = False,
        with_dact: bool = False,
        with_act: bool = False,
        use_2d_quantization: bool = False,
        activation: Optional[str] = None,
    ):
        if use_2d_quantization:
            raise ValueError("2D block scaling is not implemented by the CuTeDSL MXFP8 kernels")
        if dtype is None or dtype not in ("Float32", "Float16", "BFloat16"):
            raise ValueError(f"unknown input dtype {dtype!r}; expected Float32|Float16|BFloat16")
        self.DTYPE = str_to_cutlass_dtype(dtype)
        self.DTYPE_STR = dtype  # readable input-dtype token, for __str__
        if fp8_dtype not in ("Float8E4M3", "Float8E5M2"):
            raise ValueError(
                f"unknown FP8 dtype {fp8_dtype!r}; expected 'Float8E4M3' or 'Float8E5M2'"
            )
        self.FP8_DTYPE = str_to_cutlass_dtype(fp8_dtype)
        self.FP8_DTYPE_STR = fp8_dtype  # readable token, for __str__
        self.ROWWISE = rowwise
        self.COLWISE = colwise
        if not (rowwise or colwise):
            raise ValueError("at least one of rowwise or colwise must be true")
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
        self.WITH_AMAX = with_amax
        if not with_dact and not with_act:
            if activation == "none":
                self.ACTIVATION = None
            else:
                raise ValueError(
                    "activation must be none when with_dact and with_act are both False"
                )
        else:
            if with_dact and with_act:
                raise ValueError(
                    "with_dact and with_act cannot be true at the same time since they are used for"
                    " different paths (bwd vs fwd)"
                )
            if with_dact:
                if activation in SUPPORTED_DACTIVATIONS:
                    self.ACTIVATION = activation
                else:
                    raise ValueError(
                        f"unknown activation {activation!r} for with_dact=True; expected one of"
                        f" {sorted(SUPPORTED_DACTIVATIONS)}"
                    )
            elif with_act:
                if activation in SUPPORTED_ACTIVATIONS:
                    self.ACTIVATION = activation
                else:
                    raise ValueError(
                        f"unknown activation {activation!r} for with_act=True; expected one of"
                        f" {sorted(SUPPORTED_ACTIVATIONS)}"
                    )
        self.WITH_DACT = with_dact
        self.WITH_ACT = with_act
        # dbias is the column reduction of the (post-act/dact) element. With colwise
        # output each thread owns a full column (trivial reduction); rowwise-only
        # uses a cross-thread smem reduction over THREADS_Y. Both mirror the CUDA
        # kernel's COLWISE_SCALING / rowwise dbias branches.
        self.WITH_DBIAS = with_dbias
        self.MAX_NORM_RCP = (
            FP8E4M3_MAX_NORM_RCP if fp8_dtype == "Float8E4M3" else FP8E5M2_MAX_NORM_RCP
        )

    def __str__(self):
        return (
            f"MXFP8QuantizeConfig(dtype={self.DTYPE_STR}, fp8_dtype={self.FP8_DTYPE_STR}, "
            f"rowwise={self.ROWWISE}, colwise={self.COLWISE}, "
            f"swizzled={self.WITH_GEMM_SWIZZLED_SCALES}, with_amax={self.WITH_AMAX}, "
            f"with_dbias={self.WITH_DBIAS}, with_dact={self.WITH_DACT}, "
            f"with_act={self.WITH_ACT}, activation={self.ACTIVATION})"
        )

    __repr__ = __str__


class MXFP8QuantizeKernelBase(abc.ABC):
    """Base class for MXFP8 quantize kernels."""

    @abc.abstractmethod
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        """
        Compiled kernel entrypoint (decorate with @cute.jit).
        All MXFP8 quantize kernels must implement this interface because this is our C++ call site's contract in `quantize_mxfp8_cutedsl.cuh`.
        C++ call site will pass the arguments in this exact order via tvm-ffi and the kernel must accept them even if they are not used.
        """


@cute.jit
def bf16_pair_magnitude(pair: Int32) -> Uint32:
    """Fold both BF16 halves and clear the sign set by max.xorsign.abs."""
    swapped = cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r0}, 0x1032;",
        write_only_types=[Int32],
        read_only_args=[pair],
    )
    return Uint32(abs_max_x2_bf16(pair, swapped)) & Uint32(0x7FFF)


class MXFP8QuantizeEntry(MXFP8QuantizeKernelBase):
    """Select the appropriate MXFP8 quantization kernel based on the configuration and runtime shapes."""

    def __init__(self, cfg: MXFP8QuantizeConfig):
        # Kernels import shared definitions from this module. Resolve their
        # implementations only when constructing the compile-time entry object.
        # pylint: disable=import-outside-toplevel
        from .quantize_mxfp8_general import MXFP8QuantizeKernel
        from .quantize_mxfp8_g2r_rowwise_1lane import MXFP8QuantizeG2RRowwise1LaneKernel
        from .quantize_mxfp8_g2s_bidimensional import MXFP8QuantizeSpecializedBidimensionalKernel
        from .quantize_mxfp8_g2r_rowwise_2lane import MXFP8QuantizeG2RRowwise2LaneKernel
        from .quantize_mxfp8_g2r_bidimensional import MXFP8QuantizeRegisterBidimensionalKernel

        # pylint: enable=import-outside-toplevel

        self.cfg = cfg
        # These activation functions satisfy f(0) = 0 so we don't need to mask with OOB regions
        # (zeros filled by TMA are still zeros without applying activation to them)
        # Note that as the time of writing, all activations supported by MXFP8 satisfy this property, so !ACT_NEED_MASKING is never reachable.
        # This is only a precaution for future use cases if someone wants to add a new activation function.
        self.ACT_NEED_MASKING = cfg.WITH_ACT and cfg.ACTIVATION not in (
            "relu",
            "gelu",
            "silu",
            "qgelu",
            "srelu",
        )
        # Instantiate all possible kernels at compile time,
        # and we will pick the right one at runtime based on the input shape and config.
        self.general_skip_input_masking_kernel = MXFP8QuantizeKernel(cfg, SKIP_INPUT_MASKING=True)
        self.general_skip_input_masking_and_scale_bounds_kernel = MXFP8QuantizeKernel(
            cfg, SKIP_INPUT_MASKING=True, SKIP_SCALE_BOUNDS=True
        )
        # We only need to mask when WITH_ACT is enabled because with activations applied zeros filled by TMA affect block statistics
        self.general_kernel = MXFP8QuantizeKernel(cfg) if self.ACT_NEED_MASKING else None
        self.g2r_rowwise_1lane = MXFP8QuantizeG2RRowwise1LaneKernel(cfg) if cfg.ROWWISE else None
        self.g2s_bidim = (
            MXFP8QuantizeSpecializedBidimensionalKernel(cfg)
            if cfg.ROWWISE and cfg.COLWISE
            else None
        )
        self.g2r_rowwise_2lane = MXFP8QuantizeG2RRowwise2LaneKernel(cfg) if cfg.ROWWISE else None
        self.g2r_bidim = (
            MXFP8QuantizeRegisterBidimensionalKernel(cfg) if cfg.ROWWISE and cfg.COLWISE else None
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        M = mX.shape[0]
        N = mX.shape[1]
        # Select specialized kernels if possible
        plain_cast_only = (
            not self.cfg.WITH_AMAX
            and not self.cfg.WITH_DBIAS
            and not self.cfg.WITH_DACT
            and not self.cfg.WITH_ACT
        )
        dispatched_to_specialized = False
        # The two-lane rowwise and g2r bidimensional kernels use BF16 packed math and 32-byte loads.
        # Keep the existing kernels for FP16, fusions, ragged bidimensional tiles,
        # misaligned input views and compilers predating 256-bit global loads.
        if cutlass.const_expr(
            plain_cast_only
            and self.cfg.DTYPE is cutlass.BFloat16
            and cutlass.target_version(min_version="12.9")
        ):
            input_is_aligned = mX.iterator.toint() % 32 == 0
            if cutlass.const_expr(self.cfg.ROWWISE and not self.cfg.COLWISE):
                scale_stride_matches = Boolean(True)
                if cutlass.const_expr(not self.cfg.WITH_GEMM_SWIZZLED_SCALES):
                    scale_stride_matches = mS_row.shape[1] == N // 32
                # Column tiles use grid.x and row tiles spill from grid.y into grid.z, so
                # no shape exceeds the grid limits.
                if N % 128 == 0 and input_is_aligned and scale_stride_matches:
                    dispatched_to_specialized = True
                    self.g2r_rowwise_2lane(
                        mX,
                        mO_row,
                        mS_row,
                        mO_col,
                        mS_col,
                        mAmax,
                        mNoop,
                        mDActInput,
                        mWorkspace,
                        stream,
                    )
            if cutlass.const_expr(
                self.cfg.ROWWISE and self.cfg.COLWISE and not self.cfg.WITH_GEMM_SWIZZLED_SCALES
            ):
                columnwise_scales_are_aligned = mS_col.iterator.toint() % 16 == 0
                if (
                    M % 32 == 0
                    and N % 256 == 0
                    and M // 32 <= 65535
                    and input_is_aligned
                    and columnwise_scales_are_aligned
                ):
                    dispatched_to_specialized = True
                    self.g2r_bidim(
                        mX,
                        mO_row,
                        mS_row,
                        mO_col,
                        mS_col,
                        mAmax,
                        mNoop,
                        mDActInput,
                        mWorkspace,
                        stream,
                    )
        # The one-lane rowwise and g2s bidimensional kernels support packed16 types (bf16/fp16).
        if cutlass.const_expr(plain_cast_only and is_packed16(self.cfg.DTYPE)):
            if cutlass.const_expr(self.cfg.ROWWISE and not self.cfg.COLWISE):
                g2r_rowwise_1lane_grid_fits = (
                    cute.ceil_div(M, self.g2r_rowwise_1lane._TILE_ROWS) <= 65535
                )
                # The one-lane rowwise kernel requires N divisible by 128 for vectorized stores.
                if not dispatched_to_specialized and N % 128 == 0 and g2r_rowwise_1lane_grid_fits:
                    dispatched_to_specialized = True
                    self.g2r_rowwise_1lane(
                        mX,
                        mO_row,
                        mS_row,
                        mO_col,
                        mS_col,
                        mAmax,
                        mNoop,
                        mDActInput,
                        mWorkspace,
                        stream,
                    )
            # The g2s bidimensional kernel supports swizzled scales but we don't dispatch to it for now
            if cutlass.const_expr(
                self.cfg.ROWWISE and self.cfg.COLWISE and not self.cfg.WITH_GEMM_SWIZZLED_SCALES
            ):
                g2s_bidim_grid_fits = (
                    cute.ceil_div(
                        M,
                        self.g2s_bidim._TILE_ROWS * self.g2s_bidim._NUM_TILES_Y,
                    )
                    <= 65535
                )
                if not dispatched_to_specialized and g2s_bidim_grid_fits:
                    dispatched_to_specialized = True
                    self.g2s_bidim(
                        mX,
                        mO_row,
                        mS_row,
                        mO_col,
                        mS_col,
                        mAmax,
                        mNoop,
                        mDActInput,
                        mWorkspace,
                        stream,
                    )
        # If not using a specialized kernel, fall back to the general kernel
        if not dispatched_to_specialized:
            # If the input shape can be perfectly tiled by the general kernel's tile size, we can skip some boundary check because
            # we know we will not touch any out-of-bounds region.
            shape_is_divisible = (
                mX.shape[0] % self.general_skip_input_masking_kernel._TILE_ROWS == 0
                and mX.shape[1] % self.general_skip_input_masking_kernel._TILE_COLS == 0
            )
            # If shape is divisible, then we won't access OOB regions regardless whatever
            if shape_is_divisible:
                self.general_skip_input_masking_and_scale_bounds_kernel(
                    mX,
                    mO_row,
                    mS_row,
                    mO_col,
                    mS_col,
                    mAmax,
                    mNoop,
                    mDActInput,
                    mWorkspace,
                    stream,
                )
            else:
                # Otherwise we can't skip scale bound check, but we may still skip masking activation inputs with zero
                if cutlass.const_expr(self.ACT_NEED_MASKING):
                    # Masking and the scale guards now share one condition, so this is a plain
                    # two-way choice: either the shape tiles exactly (drop both) or it does not.
                    self.general_kernel(
                        mX,
                        mO_row,
                        mS_row,
                        mO_col,
                        mS_col,
                        mAmax,
                        mNoop,
                        mDActInput,
                        mWorkspace,
                        stream,
                    )
                else:
                    # We still need to check the scale bounds, but we can skip some activations masking because their output is 0
                    # for OOB regions where TMA fills with zeros
                    self.general_skip_input_masking_kernel(
                        mX,
                        mO_row,
                        mS_row,
                        mO_col,
                        mS_col,
                        mAmax,
                        mNoop,
                        mDActInput,
                        mWorkspace,
                        stream,
                    )


def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given MXFP8 quantization config.
    """

    kernel_obj = MXFP8QuantizeEntry(cfg)
    sym_M = cute.sym_int32()
    sym_N = cute.sym_int32(divisibility=SYM_N_DIVISIBILITY)
    in_shape = out_shape = (sym_M, sym_N)
    # TE allocates scale tensors at a padded shape (see
    # MXFP8Quantizer::get_scale_shape in transformer_engine/pytorch/csrc):
    #   rowwise:    (roundup(M, 128),           roundup(ceildiv(N, 32), 4))
    #   columnwise: (roundup(ceildiv(M, 32), 4), roundup(N, 128))
    # These padded extents are NOT M/N (and SymInt has no `//`/`+`), so give the
    # scales their own fresh syms carrying the divisibility the padding
    # guarantees (rowwise: 128 x 4; colwise: 4 x 128).
    scale_rowwise_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
    scale_colwise_shape = (cute.sym_int32(divisibility=4), cute.sym_int32(divisibility=128))
    ws_shape = (cute.sym_int32(), sym_N)  # (blocks_Y, N); N ties to input N
    # Native FP8/E8M0 dtypes at the FFI boundary (matches the DLPack dtype the C++
    # bridge sends).
    out_dtype = cfg.FP8_DTYPE
    scale_dtype = cutlass.Float8E8M0FNU

    in_fake = cute.runtime.make_fake_compact_tensor(
        cfg.DTYPE, in_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16
    )
    out_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            out_dtype,
            out_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.ROWWISE
        else None
    )
    scale_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            scale_dtype,
            scale_rowwise_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        )
        if cfg.ROWWISE
        else None
    )
    out_col_fake = (
        cute.runtime.make_fake_compact_tensor(
            out_dtype,
            out_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.COLWISE
        else None
    )
    scale_col_fake = (
        cute.runtime.make_fake_compact_tensor(
            scale_dtype,
            scale_colwise_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        )
        if cfg.COLWISE
        else None
    )
    amax_fake = (
        cute.runtime.make_fake_compact_tensor(
            Float32, (1,), stride_order=(0,), memspace=cute.AddressSpace.gmem, assumed_align=4
        )
        if cfg.WITH_AMAX
        else None
    )
    # The cast-noop flag is an always-present f32 pointer instead of an optional tensor, so that
    # the same compiled kernel serves both an absent and a present flag (see noop_flag_is_set).
    # The null address here is only a compile-time placeholder for the pointer the C++ dispatcher
    # passes at launch.
    noop_fake = cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4)
    act_input_fake = (
        cute.runtime.make_fake_compact_tensor(
            cfg.DTYPE,
            in_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.WITH_DACT
        else None
    )
    workspace_fake = (
        cute.runtime.make_fake_compact_tensor(
            Float32, ws_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=4
        )
        if cfg.WITH_DBIAS
        else None
    )

    compiled = cute.compile(
        kernel_obj,
        in_fake,  # mX
        out_row_fake,
        scale_row_fake,  # mO_row, mS_row
        out_col_fake,
        scale_col_fake,  # mO_col, mS_col
        amax_fake,  # mAmax
        noop_fake,  # mNoop (pointer to the cast_noop flag, may be null at launch)
        act_input_fake,  # mDActInput (backward slot, unused)
        workspace_fake,  # mWorkspace(backward slot, unused)
        cute.runtime.make_fake_stream(),  # stream (compiled as an explicit tvm-ffi
        # "handle" arg; C++ passes the CUDA stream
        # as void*)
        options="--enable-tvm-ffi",
    )
    return compiled


def get_mxfp8_quantization_function(
    fn_name: str,
    dtype: str,
    fp8_dtype: str,
    rowwise: bool,
    colwise: bool,
    with_gemm_swizzled_scales: bool,
    with_amax: bool,
    with_dbias: bool,
    with_dact: bool,
    with_act: bool,
    use_2d_quantization: bool,
    activation: str,
) -> bool:
    """Compile the MXFP8 quantize kernel for this config and register it in the TVM-FFI global registry
    under EXACTLY `fn_name` (the key the C++ dispatcher built; Python treats it as an opaque name).
    Returns True if a kernel is successfully registered under `fn_name` (the C++ side then fetches it with GetGlobal(fn_name));
    False if the config is unsupported, so the caller caches the negative result and falls back to the CUDA C++ kernel.
    """
    try:
        # Already registered (e.g. by a prior call) -> supported.
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
            cfg = MXFP8QuantizeConfig(
                dtype=dtype,
                fp8_dtype=fp8_dtype,
                rowwise=rowwise,
                colwise=colwise,
                with_gemm_swizzled_scales=with_gemm_swizzled_scales,
                with_amax=with_amax,
                with_dbias=with_dbias,
                with_dact=with_dact,
                with_act=with_act,
                use_2d_quantization=use_2d_quantization,
                activation=activation,
            )
        except ValueError as e:
            # The exception message states exactly why the config is unsupported
            # (unknown dtype/activation, dbias not implemented, ...). Surfacing it as a
            # warning lets the C++ dispatcher's CUDA fallback be recognized as expected.
            logger.warning(
                "CuTeDSL MXFP8 backend does not support this config, "
                "falling back to the CUDA C++ kernel: %s",
                e,
            )
            return False

        logger.debug("Compiling CuTeDSL MXFP8 quantization kernel for %s", cfg)
        compiled = compile_cutedsl_function_from_cfg(cfg)
        # The returned compiled object is not neccessarily the compiled function itself. It could be a Python
        # wrapper that parses arguments and calls the underlying function. This is needed in the general case
        # as TVM-FFI is positional-only, however, for us this is unnecessary launch overhead as we pass the
        # arguments in the correct order on the C++ side.
        # Relevant issue: https://github.com/NVIDIA/cutlass/issues/3527
        # https://github.com/NVIDIA/cutlass/pull/3589 recommends using an explicit positional only marker (/)
        # in the function signature, however it is not merged and it is something easy to miss and silently
        # introduce unnecessarry overhead.
        native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
        tvm_ffi.register_global_func(
            fn_name, native if native is not None else compiled, override=True
        )
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL MXFP8 kernel compilation & registration failed, falling back to the CUDA"
            " C++ kernel: %s",
            e,
        )
        # Unconditionally fallback to CUDA path because we can't tell if this exception is transient or permanent.
        return False
