# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Callable, Iterable, Literal, Optional, Tuple, Union, List
import itertools
import math
import os
import functools
import torch
import transformer_engine_torch as tex
from ..constants import MXFP8_BLOCK_SCALING_SIZE, NVFP4_BLOCK_SCALING_SIZE, TE_DType, DType
from ..utils import ceil_div, get_cached_ones_tensor, get_sm_count, _empty_tensor

from ..quantized_tensor import QuantizedTensorStorage, Quantizer
from ..tensor.float8_blockwise_tensor import Float8BlockQuantizer
from ..tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor.nvfp4_tensor import NVFP4Quantizer
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.float8_tensor_storage import Float8TensorStorage
from ..tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ..tensor.storage.hybrid_tensor_storage import HybridQuantizedTensorStorage
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage
from ..tensor.utils import is_custom
from ..custom_recipes.gemm import custom_gemm
from ...debug.pytorch.debug_quantization import DebugQuantizedTensor, DebugQuantizer

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
    "general_grouped_gemm_for_grouped_tensor",
]


_NUM_MAX_UB_STREAMS = 3


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        # 32 MiB for NVFP4 GEMM, plus additional 1024 B for alignment and misc scales
        return 32 * 1024 * 1024 + 1024
    return 4_194_304


@functools.lru_cache(maxsize=None)
def get_cublas_workspace(device: int, ub: bool, grouped_gemm: bool) -> torch.Tensor:
    """Returns workspace for cublas GEMM."""
    assert not (ub and grouped_gemm), "UB is unsupported for grouped GEMM."

    if ub:
        return torch.empty(
            get_cublas_workspace_size_bytes() * _NUM_MAX_UB_STREAMS,
            dtype=torch.uint8,
            device=device,
        )
    if grouped_gemm:
        _multi_stream_cublas_workspace = []
        for _ in range(tex.get_num_cublas_streams()):
            _multi_stream_cublas_workspace.append(
                torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)
            )
        return _multi_stream_cublas_workspace

    return torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Validate whether a GEMM scaling factor is consistent with its usage"""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def _is_nvfp4_row_scaled_tensor(tensor: torch.Tensor) -> bool:
    """Whether tensor carries row-scaled NVFP4 global amax metadata."""
    return isinstance(tensor, NVFP4TensorStorage) and tensor._row_scaled_nvfp4


def _nvfp4_row_scaled_gemm_inputs(
    A: NVFP4TensorStorage,
    B: NVFP4TensorStorage,
    *,
    transa: bool,
    transb: bool,
) -> Tuple[NVFP4TensorStorage, NVFP4TensorStorage, torch.Tensor, torch.Tensor]:
    """Return per-tensor GEMM aliases and row/column FP32 output scales."""
    A_metadata = A.get_metadata()
    a_amax_key = "amax_rowwise" if transa else "amax_columnwise"
    output_col_scales = A_metadata[a_amax_key]
    assert output_col_scales is not None
    A_metadata[a_amax_key] = output_col_scales.new_ones(1)
    A_metadata["row_scaled_nvfp4"] = False

    B_metadata = B.get_metadata()
    b_amax_key = "amax_columnwise" if transb else "amax_rowwise"
    output_row_scales = B_metadata[b_amax_key]
    assert output_row_scales is not None
    B_metadata[b_amax_key] = output_row_scales.new_ones(1)
    B_metadata["row_scaled_nvfp4"] = False

    assert output_row_scales.dtype == torch.float32 and output_col_scales.dtype == torch.float32
    return (
        NVFP4TensorStorage(**A_metadata),
        NVFP4TensorStorage(**B_metadata),
        output_row_scales.view(-1, 1),
        output_col_scales.view(1, -1),
    )


_NATIVE_GEMM_INPUT_STORAGES = (
    Float8TensorStorage,
    MXFP8TensorStorage,
    Float8BlockwiseQTensorStorage,
    NVFP4TensorStorage,
)


def _unwrap_tensor(
    tensor: Union[torch.Tensor, QuantizedTensorStorage],
    usage: Literal["rowwise", "columnwise"],
) -> Union[torch.Tensor, QuantizedTensorStorage]:
    """Prepare a tensor for native or custom GEMM dispatch."""
    if usage not in ("rowwise", "columnwise"):
        raise ValueError(f"Unsupported GEMM tensor usage ({usage})")

    # Hybrid and debug wrappers may omit a representation that is not needed
    # by their configured GEMMs. Fail here if a caller requests that missing
    # direction instead of passing ``None`` deeper into GEMM preparation.
    if tensor is None:
        raise RuntimeError(
            f"GEMM requested the {usage} representation, but it is unavailable. "
            f"Ensure {usage}_usage is enabled and the representation has not "
            "been dropped by update_usage()."
        )

    # Plain PyTorch tensor
    if not isinstance(tensor, QuantizedTensorStorage):
        return tensor

    # Select the requested representation from a debug wrapper, then process
    # the wrapped tensor normally (it may itself be hybrid or TE-native).
    if isinstance(tensor, DebugQuantizedTensor):
        return _unwrap_tensor(tensor.get_tensor(usage == "columnwise"), usage)

    # Select the direction of a hybrid tensor, then process its sub-storage.
    if isinstance(tensor, HybridQuantizedTensorStorage):
        sub_storage = (
            tensor.rowwise_sub_storage if usage == "rowwise" else tensor.columnwise_sub_storage
        )
        return _unwrap_tensor(sub_storage, usage)

    # Preserve custom tensors for custom_gemm dispatch.
    if is_custom(tensor):
        return tensor

    # Quantized tensor formats with native GEMM support
    if isinstance(tensor, _NATIVE_GEMM_INPUT_STORAGES):
        return tensor

    # Fall back to high-precision GEMM for other quantized tensor formats.
    return tensor.dequantize()


_NATIVE_GEMM_OUTPUT_QUANTIZERS = (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
    Float8BlockQuantizer,
    NVFP4Quantizer,
)


def _validate_native_gemm_output_quantizer(quantization_params):
    """Validate that the native C++ GEMM path can convert an output quantizer."""
    if quantization_params is not None and not isinstance(
        quantization_params, _NATIVE_GEMM_OUTPUT_QUANTIZERS
    ):
        raise NotImplementedError(
            f"{type(quantization_params).__name__} is not supported as a native "
            "GEMM output quantizer. "
            "Return a TE-native quantizer for output/grad_input roles or disable "
            "quantized GEMM output for this boundary."
        )


def validate_or_alloc_output(
    buffer: Optional[torch.Tensor],
    shape: tuple[int, ...] | list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the caller's output buffer, or allocate one if it is None.

    The buffer must be a contiguous tensor matching the required
    shape, dtype, and device.

    """
    shape = tuple(shape)
    if buffer is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if tuple(buffer.shape) != shape:
        raise ValueError(f"Output buffer shape {tuple(buffer.shape)} does not match {shape}.")
    if buffer.dtype != dtype:
        raise ValueError(f"Output buffer dtype {buffer.dtype} does not match {dtype}.")
    if buffer.device != device:
        raise ValueError(f"Output buffer device {buffer.device} does not match {device}.")
    if not buffer.is_contiguous():
        raise ValueError("Output buffer must be contiguous.")
    return buffer


@functools.lru_cache(maxsize=None)
def _cudnn_grouped_gemm_quant_kernel() -> Callable:
    """cuDNN CuTe DSL grouped GEMM kernel for block-scaled inputs.

    This function is a temporary hack until TE supports NVFP4-UE5M3
    GEMMs natively. This should not be used externally and once native
    GEMM support is added then this function (and related helper
    functions) should be removed entirely.

    """
    from cudnn import grouped_gemm_quant_wrapper_sm100  # pylint: disable=no-name-in-module

    return grouped_gemm_quant_wrapper_sm100


@functools.lru_cache(maxsize=None)
def _cudnn_grouped_gemm_wgrad_kernel() -> Callable:
    """cuDNN CuTe DSL grouped wgrad kernel for block-scaled inputs.

    This function is a temporary hack until TE supports NVFP4-UE5M3
    GEMMs natively. This should not be used externally and once native
    GEMM support is added then this function (and related helper
    functions) should be removed entirely.

    """
    from cudnn import grouped_gemm_wgrad_wrapper_sm100  # pylint: disable=no-name-in-module

    return grouped_gemm_wgrad_wrapper_sm100


def _cudnn_wgrad_grouped_gemm_nvfp4_ue5m3(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    amax_a: Optional[torch.Tensor],
    amax_b: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: torch.Tensor,
    accumulate: bool,
    alpha: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
) -> Iterable[Optional[torch.Tensor]]:
    """Compute dw = dy^T @ x for NVFP4-UE5M3 data with cuDNN's grouped wgrad kernel.

    This function is a temporary hack until TE supports NVFP4-UE5M3
    GEMMs natively. This should not be used externally and once native
    GEMM support is added then this function (and related helper
    functions) should be removed entirely.

    """

    # Column-wise NVFP4 buffers are physically (features, tokens), FP4-packed
    # two values per byte along the token dim.
    tokens_packed = a_tensor.shape[-1]
    tokens = tokens_packed * 2
    out_features, in_features = out.size()

    # grouped_gemm_wgrad_wrapper_sm100 wants:
    #     a_tensor  (feature_out, tokens)      K-major, FP4-packed
    #     b_tensor  (tokens, feature_in)
    #     sfa       (round_up(feature_out, 128), scale_cols)
    #     sfb       (round_up(feature_in, 128),  scale_cols)
    fp4 = torch.float4_e2m1fn_x2
    a_tensor = a_tensor.view(dtype=fp4).view(out_features, tokens_packed)
    b_tensor = b_tensor.view(dtype=fp4).view(in_features, tokens_packed).T

    # Create the scale factor tensors with the logical layout cuDNN expects
    # Assume scales have already been swizzled in _cudnn_grouped_gemm_nvfp4_ue5m3
    def _sf(scale_inv, features):
        leading = ceil_div(features, 128) * 128
        return scale_inv.view(leading, -1).view(dtype=torch.float8_e4m3fn)

    # grouped_gemm_wgrad_wrapper_sm100 expects two separate global_scale
    ones = get_cached_ones_tensor(1, dtype=torch.float32, device=a_tensor.device)
    denom = 6.0 * 114688.0  # fp4_max * fp8_max(UE5M3)
    global_scale_a = ones if amax_a is None else amax_a.to(torch.float32).reshape(1) / denom
    global_scale_b = ones if amax_b is None else amax_b.to(torch.float32).reshape(1) / denom
    # Fold alpha into one of them if it's given
    if alpha is not None and alpha != 1.0:
        global_scale_a = global_scale_a * alpha

    out = validate_or_alloc_output(out, (out_features, in_features), out_dtype, a_tensor.device)
    _cudnn_grouped_gemm_wgrad_kernel()(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        sfa_tensor=_sf(sfa, out_features),
        sfb_tensor=_sf(sfb, in_features),
        offsets_tensor=torch.tensor([tokens], dtype=torch.int32, device=a_tensor.device),
        global_scale_a=global_scale_a,
        global_scale_b=global_scale_b,
        acc_dtype=torch.float32,
        wgrad_dtype=out.dtype,
        output_mode="dense",
        wgrad_tensor=out.view(1, out_features, in_features),
        sf_vec_size=NVFP4_BLOCK_SCALING_SIZE,
        sf_fp8_dtype_override="e5m3",
        input_order="tensor_ragged",
        accumulate_on_output=accumulate,
        current_stream=torch.cuda.current_stream().cuda_stream,
    )

    # Apply bias
    if bias is not None:
        out += bias.view(1, in_features)

    # Matches general_gemm's contract: (out, bias_grad, gelu_input, extra_output).
    return out, None, None, None


def _convert_to_cudnn_grouped_gemm_tensor_format(
    data: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    data_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    valid_M_or_N: int,
    k_logical: int,
    L: int = 1,
    sf_swizzled: bool = False,
    use_N_major_for_B: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape a plain buffer into the layout cuDNN's grouped GEMM expects.

    cuDNN requirements:
    A: (valid_m, K, 1), K-major
    B: (N, K, L), K-major (FP8 also supports N-major)

    SFA: (32, 4, ceil(valid_m/128), 4, ceil(ceil(K/sf_vec_size)/4), 1)
    SFB: (32, 4, ceil(N/128),       4, ceil(ceil(K/sf_vec_size)/4), L)

    whereas TE stores flat buffers which can be intepreted as contiguous tensors
    with the following layouts:

    Note: K_packed is K/2 for FP4 (two values per byte) and K for FP8

    A (K-major):    (1, valid_m, K_packed)
    B (K-major):    (L, N,       K_packed)   -- used for FP4 only now
    B (N-major):    (L, K,       N)          -- used for FP8 only now

    SFA (unswizzled): (1, ceil(valid_m/128), 4, 32, ceil(ceil(K/sf_vec_size)/4), 4)
    SFB (unswizzled): (L, ceil(N/128),       4, 32, ceil(ceil(K/sf_vec_size)/4), 4)
    SFA (swizzled): (1, ceil(valid_m/128), ceil(ceil(K/sf_vec_size)/4), 32, 4, 4)
    SFB (swizzled): (L, ceil(N/128),       ceil(ceil(K/sf_vec_size)/4), 32, 4, 4)

    This function is a temporary hack until TE supports NVFP4-UE5M3
    GEMMs natively. This should not be used externally and once native
    GEMM support is added then this function (and related helper
    functions) should be removed entirely.

    """

    if use_N_major_for_B:
        assert data_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ), f"Using N-major layout for B is only supported for FP8, but got {data_dtype}."

    available_scalings = {
        # NVFP4 recipe (UE5M3 rides as E4M3 since torch has no ue5m3 dtype)
        (torch.float4_e2m1fn_x2, torch.float8_e4m3fn): NVFP4_BLOCK_SCALING_SIZE,
        # MXFP8 recipe
        (torch.float8_e4m3fn, torch.float8_e8m0fnu): MXFP8_BLOCK_SCALING_SIZE,
    }
    assert (data_dtype, scale_dtype) in available_scalings, (
        "Unsupported (data_dtype, scale_dtype) pair for a cuDNN block-scaled operand: "
        f"({data_dtype}, {scale_dtype}). Expected NVFP4 (float4_e2m1fn_x2, "
        "float8_e4m3fn) or MXFP8 (float8_e4m3fn, float8_e8m0fnu)."
    )
    sf_vec_size = available_scalings[(data_dtype, scale_dtype)]

    k_sf_tiles = ceil_div(k_logical, 4 * sf_vec_size)

    if data_dtype == torch.float4_e2m1fn_x2:
        k_packed = k_logical // 2  # fp4 packs two values per byte
    else:
        k_packed = k_logical  # fp8 packs one value per byte

    data = data.view(dtype=data_dtype)
    if use_N_major_for_B:
        # B is stored untransposed, i.e. (L, K, N); permuting to (N, K, L) leaves
        # stride 1 on N. Only FP8 accepts this, asserted above.
        data = data.view(L, k_packed, valid_M_or_N)
        data = data.permute(2, 1, 0)
    else:
        # (L, N, K) -> (N, K, L), stride 1 on K.
        data = data.view(L, valid_M_or_N, k_packed)
        data = data.permute(1, 2, 0)

    if sf_swizzled:
        scale_inv = scale_inv.view(dtype=scale_dtype)
        scale_inv = scale_inv.view(
            L,
            ceil_div(valid_M_or_N, 128),
            k_sf_tiles,
            32,
            4,
            4,
        )
        scale_inv = scale_inv.permute(3, 4, 1, 5, 2, 0)
        return data, scale_inv

    scale_inv = scale_inv.view(dtype=scale_dtype)
    scale_inv = scale_inv.view(
        L,
        ceil_div(valid_M_or_N, 128),
        4,
        32,
        k_sf_tiles,
        4,
    )
    scale_inv = scale_inv.permute(3, 2, 1, 5, 4, 0)
    return data, scale_inv


def _cudnn_grouped_gemm_nvfp4_ue5m3(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """Perform GEMM via cuDNN kernels

    The parameters passed are in cuBLAS notation, where
    D = alpha * op(B) @ op(A) + beta * C, where the shape is always
    (N, M) = (N, K) @ (K, M) + (N, M)

    B:
    - "N" is (N, K), which is always TE's rowwise data, and op(B) is B
    - "T" is (K, N), which is always TE's colwise data, and op(B) is B.T
    A
    - "N" is (K, M), which is always TE's colwise data, and op(A) is A
    - "T" is (M, K), which is always TE's rowwise data, and op(A) is A.T

    Note: layout string means layout of "A" and "B" respectively.

    TE stores x (token, feature_in), w (feature_out, feature_in) and dy (token, feature_out) in physical rowwise direction.
    For cuBLAS:
    fprop = x @ wT: token is N, feature_in is K, feature_out is M, so it's TN (x as B, w transposed to wT as A)
    dgrad = dy @ w: token is N, feature_out is K, feature_in is M, so it's NN (dy as B, w as A)
    wgrad = dyT @ x: feature_out is N, token is K, feature_in is M, so it's NT (dy transposed to dyT as B, x as A)

    We use cuDNN-frontend's APIs here which are supposed to be used for grouped GEMM but we set groups = 1
    so it is effectively a single GEMM.

    Naming convention: uppercase letters (A, B) are used for cuBLAS notation, lowercase letters (a, b) are used for cuDNN notation.
                       where cuBLAS's B is cuDNN's a, and cuBLAS's A is cuDNN's b (their notation is inverted).

    This function is a temporary hack until TE supports NVFP4-UE5M3
    GEMMs natively. This should not be used externally and once native
    GEMM support is added then this function (and related helper
    functions) should be removed entirely.

    """
    assert (
        isinstance(A, NVFP4TensorStorage)
        and isinstance(B, NVFP4TensorStorage)
        and A._scale_dtype == DType.kFloat8UE5M3
        and B._scale_dtype == DType.kFloat8UE5M3
    ), "cuDNN MX GEMM is only used for NVFP4 GEMM with e5m3 scale factors for now."

    assert quantization_params is None, "cuDNN GEMM currently does not support output quantization."
    assert gelu is False and gelu_in is None, "cuDNN GEMM currently does not support fused GELU."

    assert ub is None and ub_type is None, "cuDNN GEMM currently does not support CommOverlap."
    assert extra_output is None, "cuDNN GEMM currently does not support extra output."
    assert bulk_overlap is False, "cuDNN GEMM currently does not support bulk overlap."

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    assert out_dtype in (torch.float32, torch.float16, torch.bfloat16), (
        "cuDNN MX GEMM currently only supports float32, float16, and bfloat16 outputs, but got"
        f" {out_dtype}."
    )

    device = A.device

    # cuDNN only accepts GEMM-swizzled scale factors -- an unswizzled buffer is
    # rejected on its strides -- so swizzle first if the quantizer did not
    # (optimize_for_gemm defaults to False). This mirrors what the cuBLAS path
    # does in C++ via swizzle_scales_for_gemm. The call is in-place, swizzles
    # both orientations, and no-ops when the tensor is already swizzled.
    if not A._with_gemm_swizzled_scales:
        tex.swizzle_scales_for_gemm_(A)
    if not B._with_gemm_swizzled_scales:
        tex.swizzle_scales_for_gemm_(B)

    # `grad` only changes behaviour when a bias is supplied: it turns the bias slot
    # into a bias-gradient output, which cuDNN has no epilogue for. Backward GEMMs
    # that pass grad=True without a bias need nothing special.
    assert not (
        grad and bias is not None
    ), "cuDNN GEMM currently does not support fused bias gradient."

    # Pick the buffer whose block scales run along K. In every case the selected
    # buffer is physically (rows, K_packed), so the reshape below is uniform.
    # LHS is always (M, K)
    if transb:
        dataB, sfB, amaxB = B._columnwise_data, B._columnwise_scale_inv, B._amax_columnwise
    else:
        dataB, sfB, amaxB = B._rowwise_data, B._rowwise_scale_inv, B._amax_rowwise
    # RHS is always (K, N)
    if transa:
        dataA, sfA, amaxA = A._rowwise_data, A._rowwise_scale_inv, A._amax_rowwise
    else:
        dataA, sfA, amaxA = A._columnwise_data, A._columnwise_scale_inv, A._amax_columnwise

    # Input tensor dims
    A_shape = list(dataA.size())
    A_shape[-1] *= 2
    B_shape = list(dataB.size())
    B_shape[-1] *= 2

    # GEMM dimensions
    M_full = A_shape[:-1] if transa else [A_shape[0]]
    N_full = [B_shape[0]] if transb else B_shape[:-1]
    K_full = [A_shape[-1]] if transa else A_shape[1:]
    K_full_b = B_shape[1:] if transb else [B_shape[-1]]
    assert (
        K_full == K_full_b
    ), f"Contraction dims disagree: A implies {K_full}, B implies {K_full_b}."
    M = math.prod(M_full)
    N = math.prod(N_full)
    K = math.prod(K_full)

    # Allocate output tensor if needed
    out_shape = N_full + M_full
    out = validate_or_alloc_output(out, out_shape, out_dtype, device)

    # Trivial cases
    if K == 0:
        if bias is not None:
            out_2d = out.view(N, M)
            bias_2d = bias.view(1, M)
            if accumulate:
                out_2d += bias_2d
            else:
                out_2d.copy_(bias_2d)
        elif not accumulate:
            out.zero_()
        return out, None, None, None
    if M == 0 or N == 0:
        return out, None, None, None

    # Route to cuDNN-FE's wgrad API for cases not supported by the
    # grouped GEMM (accumulation to output tensor, insufficient
    # alignment). The wgrad kernel has no bias epilogue, so any bias
    # has to be applied after the GEMM.
    if accumulate or N % 256 != 0:
        alpha = alpha if alpha is not None else 1.0
        # This path uses cuDNN's wgrad (grouped_gemm_wgrad_wrapper_sm100) which supports grad accumulation
        if accumulate:  # Accumulate GEMM's result to the out tensor
            assert beta in (1.0, None), "beta must be one or None if accumulate is True"
        else:  # Overwrite GEMM's result to the out tensor
            assert beta in (0.0, None), "beta must be zero or None if not accumulate"
        _cudnn_wgrad_grouped_gemm_nvfp4_ue5m3(
            a_tensor=dataB.view(N, K // 2),
            b_tensor=dataA.view(M, K // 2),
            sfa=sfB,
            sfb=sfA,
            amax_a=amaxB,
            amax_b=amaxA,
            out_dtype=out_dtype,
            out=out.view(N, M),
            accumulate=accumulate,
            alpha=alpha,
            bias=bias,
        )
        return out, None, None, None

    alpha = alpha if alpha is not None else 1.0
    # cuDNN's general GEMM path (grouped_gemm_quant_wrapper_sm100) doesn't support accumulation
    assert (
        accumulate is False
    ), "cuDNN GEMM currently does not support accumulation for this operation."
    assert beta in (0.0, None), "beta must be zero or None if not accumulate"

    # cuDNN's own operand names are the other way round: its "a" is the (M, K)
    # activation-like operand (TE's B) and its "b" is the (N, K) weight-like one
    # (TE's A).
    cudnn_a, cudnn_sfa = _convert_to_cudnn_grouped_gemm_tensor_format(
        dataB,
        sfB,
        data_dtype=torch.float4_e2m1fn_x2,
        scale_dtype=torch.float8_e4m3fn,  # e5m3 rides as e4m3; torch has no ue5m3
        valid_M_or_N=N,
        k_logical=K,
        L=1,
        sf_swizzled=True,  # ensured above
    )
    cudnn_b, cudnn_sfb = _convert_to_cudnn_grouped_gemm_tensor_format(
        dataA,
        sfA,
        data_dtype=torch.float4_e2m1fn_x2,
        scale_dtype=torch.float8_e4m3fn,  # e5m3 rides as e4m3; torch has no ue5m3
        valid_M_or_N=M,
        k_logical=K,
        L=1,
        sf_swizzled=True,  # ensured above
    )

    # Row-scaled NVFP4 stores one amax per row instead of one per tensor, which
    # this path cannot express; general_gemm handles that mode separately.
    for name, amax in (("A", amaxA), ("B", amaxB)):
        assert amax is None or amax.numel() == 1, (
            f"cuDNN MX GEMM expects a per-tensor amax for {name}, but got {amax.numel()} "
            "values. Row-scaled NVFP4 is not supported on this path."
        )

    # Prepare alpha. cuDNN applies the block scales but not TE's per-tensor global
    # scale, so alpha carries the product of both operands'. A tensor quantized
    # without second-level scaling has no amax and contributes a factor of one.
    nvfp4_global_scale = 6.0 * 114688.0
    ones = get_cached_ones_tensor(1, dtype=torch.float32, device=device)
    scaleA = ones if amaxA is None else amaxA.to(torch.float32).reshape(1) / nvfp4_global_scale
    scaleB = ones if amaxB is None else amaxB.to(torch.float32).reshape(1) / nvfp4_global_scale
    alpha_tensor = (alpha * scaleA * scaleB).to(torch.float32)

    if bias is not None:
        assert (
            bias.dim() == 1 and bias.shape[0] == M
        ), f"cuDNN MX GEMM expects a ({M},) bias, but got {tuple(bias.shape)}."
        # cuDNN checks the stride literally, so (1, N) rather than reshape's (1, 1).
        bias = bias.contiguous().as_strided((M, 1), (1, M))

    d_tensor = out.view(N, M).as_strided((N, M, 1), (M, 1, M * N))

    gemm_kwargs = {
        "a_tensor": cudnn_a,
        "sfa_tensor": cudnn_sfa,
        "b_tensor": cudnn_b,
        "sfb_tensor": cudnn_sfb,
        # One group, so the only padded end offset is the full row count.
        "padded_offsets": torch.tensor([N], dtype=torch.int32, device=device),
        "alpha_tensor": alpha_tensor,
        "bias_tensor": bias,
        "norm_const_tensor": None,  # must be None for FP4 inputs
        "acc_dtype": torch.float32,
        "d_dtype": out_dtype,  # high precision -> no output quantization
        "d_tensor": d_tensor,
        "cd_major": "n",  # only "n" is supported by cuDNN
        "sf_vec_size": NVFP4_BLOCK_SCALING_SIZE,  # Hardcode to NVFP4 for now
        "sf_fp8_dtype_override": "e5m3",  # Hardcode for now
        "current_stream": torch.cuda.current_stream().cuda_stream,
        "discrete_col_sfd": False,
        "use_dynamic_sched": True,
    }
    _cudnn_grouped_gemm_quant_kernel()(**gemm_kwargs)

    # Matches general_gemm's contract: (out, bias_grad, gelu_input, extra_output).
    return out, None, None, None


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer

    A = _unwrap_tensor(A, "rowwise" if transa else "columnwise")
    B = _unwrap_tensor(B, "columnwise" if transb else "rowwise")

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)
    workspace = get_cublas_workspace(A.device.index, ub is not None, False)

    # Temporary hack to route NVFP4 GEMM with UE5M3 scale factors to
    # cuDNN Frontend kernels. UE5M3-specific logic should be removed
    # in its entirety once TE supports NVFP4-UE5M3 GEMMs natively.
    if (
        isinstance(A, NVFP4TensorStorage)
        and isinstance(B, NVFP4TensorStorage)
        and A._scale_dtype == DType.kFloat8UE5M3
        and B._scale_dtype == DType.kFloat8UE5M3
    ):
        return _cudnn_grouped_gemm_nvfp4_ue5m3(
            A,
            B,
            out_dtype,
            quantization_params,
            gelu,
            gelu_in,
            alpha,
            beta,
            accumulate,
            layout,
            out,
            bias,
            grad,
            ub,
            ub_type,
            extra_output,
            bulk_overlap,
        )

    if ub_type is not None:
        assert ub is not None, (
            f"{'AG+GEMM' if ub_type == tex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
            + "a valid `ub` communicator object."
        )

    if ub is not None:
        assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
        if ub_type == tex.CommOverlapType.RS:
            if not (bulk_overlap and not ub.is_fp8_ubuf()):
                assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # If A or B are custom tensors -> dispatch to quantizers's qgemm implementation
    if is_custom(A) or is_custom(B):
        return custom_gemm(
            A,
            B,
            workspace,
            out_dtype,
            quantization_params,
            gelu,
            gelu_in,
            accumulate,
            layout,
            out,
            bias,
            use_split_accumulator,
            grad,
        )

    _validate_native_gemm_output_quantizer(quantization_params)

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    if isinstance(A, Float8BlockwiseQTensorStorage) or isinstance(B, Float8BlockwiseQTensorStorage):
        # FP8 block-scaling requires split accumulator
        use_split_accumulator = True

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": ub,
        "comm_type": ub_type,
        "extra_output": extra_output,
        "bulk_overlap": bulk_overlap,
        "alpha": alpha,
        "beta": beta,
    }

    if not _is_nvfp4_row_scaled_tensor(A) and not _is_nvfp4_row_scaled_tensor(B):
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
    else:
        assert not gelu, "Row-scaled NVFP4 GEMM currently does not support fused GELU."
        assert (
            quantization_params is None
        ), "Row-scaled NVFP4 GEMM currently does not support output quantization."
        assert ub is None, "Row-scaled NVFP4 GEMM currently does not support CommOverlap."
        assert (
            extra_output is None
        ), "Row-scaled NVFP4 GEMM currently does not support extra output."
        assert not bulk_overlap, "Row-scaled NVFP4 GEMM currently does not support bulk overlap."
        assert out is None or (
            isinstance(out, torch.Tensor) and not is_custom(out)
        ), "Row-scaled NVFP4 GEMM currently supports only plain torch.Tensor outputs."
        assert isinstance(
            A, NVFP4TensorStorage
        ), "Row-scaled NVFP4 GEMM currently requires NVFP4 A."
        assert isinstance(
            B, NVFP4TensorStorage
        ), "Row-scaled NVFP4 GEMM currently requires NVFP4 B."
        # Reuse the per-tensor GEMM and apply selected row/column global scales
        # to the FP32 output. This extends #2931 without a dedicated GEMM kernel.
        gemm_A, gemm_B, output_row_scales, output_col_scales = _nvfp4_row_scaled_gemm_inputs(
            A, B, transa=transa, transb=transb
        )

        requested_out, requested_out_dtype = out, out_dtype
        fp32_out = (
            torch.empty_like(requested_out, dtype=torch.float32)
            if requested_out is not None
            else None
        )
        gemm_args = list(args)
        gemm_args[0] = gemm_A  # A
        gemm_args[2] = gemm_B  # B
        gemm_args[4] = fp32_out  # out
        gemm_args[5] = None  # quantization_params
        gemm_args[6] = TE_DType[torch.float32]  # out_dtype
        gemm_args[7] = None  # bias
        gemm_args[14] = False  # accumulate after applying the outer scales
        gemm_kwargs = dict(kwargs)
        gemm_kwargs["beta"] = 0.0
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*gemm_args, **gemm_kwargs)
        out_2d = out.reshape(-1, out.shape[-1])

        assert output_row_scales.numel() in (1, out_2d.shape[0])
        assert output_col_scales.numel() in (1, out_2d.shape[1])
        assert out.dtype == torch.float32
        # When one side is a scalar global amax (e.g. fprop weight), fold both
        # scales into a single factor before the multiply. This reproduces
        # #2931's fused `out * (row_amax * col_amax)` arithmetic bit-for-bit;
        # only the true bilateral case (both per-row and per-col) needs the
        # two-step outer-product scaling.
        if output_col_scales.numel() == 1:
            out_2d.mul_(output_row_scales * output_col_scales)
        elif output_row_scales.numel() == 1:
            out_2d.mul_(output_col_scales * output_row_scales)
        else:
            out_2d.mul_(output_row_scales)
            out_2d.mul_(output_col_scales)
        if bias is not None:
            assert not grad, "Row-scaled NVFP4 backward does not support fused bias gradient."
            out_2d.add_(bias.to(dtype=torch.float32))

        if requested_out is not None:
            if accumulate:
                requested_out.add_(out.to(dtype=requested_out.dtype))
            else:
                requested_out.copy_(out.to(dtype=requested_out.dtype))
            out = requested_out
        elif requested_out_dtype is not None and requested_out_dtype != torch.float32:
            out = out.to(dtype=requested_out_dtype)

    if debug_quantizer is not None:
        out = debug_quantizer.process_gemm_output(out)

    return out, bias_grad, gelu_input, extra_output


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    quantization_params: List[Optional[Quantizer]],
    out_dtype: torch.dtype,
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    A = [_unwrap_tensor(a, "rowwise" if transa else "columnwise") for a in A]
    B = [_unwrap_tensor(b, "columnwise" if transb else "rowwise") for b in B]

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    workspaces = get_cublas_workspace(A[0].device.index, False, True)

    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].size(1), dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    # Determine whether to repeatedly call general_gemm
    use_general_gemm_impl = False
    if isinstance(quantization_params[0], DebugQuantizer):
        use_general_gemm_impl = True
    elif any(_is_nvfp4_row_scaled_tensor(tensor) for tensor in itertools.chain(A, B)):
        use_general_gemm_impl = True
    elif any(
        isinstance(t, NVFP4TensorStorage) and t._scale_dtype == DType.kFloat8UE5M3
        for t in itertools.chain(A, B)
    ):
        use_general_gemm_impl = True

    # Repeatedly call general_gemm if needed
    if use_general_gemm_impl:
        out_views = out
        if single_output:
            start_idx = 0
            out = out[0]
            out_views = [None] * num_gemms
            for i in range(num_gemms):
                size = m_splits[i]
                out_views[i] = out[start_idx : start_idx + size]
                start_idx += size
        for i in range(num_gemms):
            _, bias_or_grad, gelu_input_i, _ = general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out_views[i].dtype,
                layout=layout,
                accumulate=accumulate,
                out=out_views[i],
                gelu=gelu,
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
            if grad and use_bias:
                grad_bias[i] = bias_or_grad
            if gelu:
                gelu_input[i] = gelu_input_i
        return out, grad_bias if grad else bias, gelu_input

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    bias = tex.te_general_grouped_gemm(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype,
        m_splits,
        grad_bias if grad else bias,
        bias_dtype,
        single_output,
        gelu_input,  # this is pre_gelu_out
        grad,  # grad
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, bias, gelu_input


@functools.lru_cache(maxsize=None)
def get_grouped_gemm_setup_workspace_size(num_tensors: int) -> int:
    """Return workspace size for grouped GEMM pointer setup."""
    return tex.get_grouped_gemm_setup_workspace_size(num_tensors)


@functools.lru_cache(maxsize=None)
def _get_fp32_ones_tensor(num_tensors: int, device: torch.device) -> torch.Tensor:
    """Cached ones tensor."""
    return torch.ones(num_tensors, dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=None)
def _get_fp32_zeros_tensor(num_tensors: int, device: torch.device) -> torch.Tensor:
    """Cached zeros tensor."""
    return torch.zeros(num_tensors, dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=None)
def _get_grouped_gemm_setup_workspace(device: int, num_tensors: int) -> torch.Tensor:
    """Persistent setup workspace (per-group pointer/dim arrays) for grouped-tensor GEMM."""
    return torch.empty(
        get_grouped_gemm_setup_workspace_size(num_tensors),
        dtype=torch.uint8,
        device=device,
    )


@functools.lru_cache(maxsize=None)
def _get_grouped_cublas_workspace(device: int, layout: str) -> torch.Tensor:
    """Persistent cuBLAS workspace for the grouped-tensor GEMM path, one per GEMM layout.

    Grouped cuBlasLt GEMM kernels in cuBLAS versions <= 13.7 leave behind stale descriptors in the
    workspace that cause back-to-back GEMM kernels to crash/deadlock on 2nd CUDA-graph replay. As a
    workaround, we allocate a different workspace for each GEMM layout (TN, NN, NT) to avoid
    contamination between subsequent GEMM calls (when there is no other graph node between GEMM
    kernels).
    """
    assert layout in ("TN", "NN", "NT"), f"unexpected grouped GEMM layout {layout}"
    return torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)


def general_grouped_gemm_for_grouped_tensor(
    A,
    B,
    out,
    *,
    layout: str = "TN",
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    bias=None,
    bias_scale: Optional[torch.Tensor] = None,
    grad: bool = False,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Grouped GEMM using GroupedTensor inputs.

    This uses nvte_grouped_gemm and supports different per-matrix shapes.

    The caller must ensure that GroupedTensor metadata is already compatible with the
    underlying GEMM implementation (e.g., aligned offsets and output metadata layout).
    """
    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    if grad:
        raise NotImplementedError("grad is not supported for grouped_tensor GEMM yet.")
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    is_discrete_out = isinstance(out, list)
    is_discrete_in = isinstance(A, list)
    if is_discrete_in and is_discrete_out:
        raise ValueError("Both A and out are discrete. This is not supported yet.")

    if isinstance(A, GroupedTensorStorage) and A.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")
    if isinstance(B, GroupedTensorStorage) and B.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")
    if isinstance(out, GroupedTensorStorage) and out.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")

    def _is_fp8_blockwise(operand) -> bool:
        if isinstance(operand, (list, tuple)):
            return any(isinstance(t, Float8BlockwiseQTensorStorage) for t in operand)
        if isinstance(operand, GroupedTensorStorage):
            return isinstance(operand.quantizer, Float8BlockQuantizer)
        return False

    if _is_fp8_blockwise(A) or _is_fp8_blockwise(B):
        # The fused grouped FP8 block-scaling GEMM only supports split accumulation,
        # so force it on and intentionally override any caller-supplied value. This
        # matches the Float8BlockScaling recipe, which fixes use_split_accumulator=True
        # for all of fprop/dgrad/wgrad, so no user-configurable setting is discarded.
        use_split_accumulator = True

    if is_discrete_out:
        # wgrad case.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_out
    elif is_discrete_in:
        # Use-case: forward pass with list of weights.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_in
    else:
        # Use-case: Single Grouped Parameter for Weight/ Weight Grads.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_grouped_tensor

    if is_discrete_out and bias is not None:
        raise ValueError(
            "Bias is not supported when out is a list (discrete_out mode) yet. "
            "Apply bias manually after the GEMM."
        )

    if bias_scale is not None and bias is None:
        raise ValueError("bias_scale requires bias to be provided.")

    num_tensors = B.num_tensors
    rowwise = B.rowwise_data
    device = rowwise.device if rowwise is not None else B.columnwise_data.device

    # Hopper (SM90) uses a single shared alpha/beta scalar;
    # Blackwell+ (SM100) supports per-group alpha/beta arrays.
    per_group = torch.cuda.get_device_capability() >= (10, 0)
    num_alphabeta = num_tensors if per_group else 1

    if alpha is None:
        alpha = _get_fp32_ones_tensor(num_alphabeta, device)
    if beta is None:
        if accumulate:
            beta = _get_fp32_ones_tensor(num_alphabeta, device)
        else:
            beta = _get_fp32_zeros_tensor(num_alphabeta, device)

    if not alpha.is_cuda or not beta.is_cuda:
        raise ValueError("alpha and beta must be CUDA tensors.")

    workspace_setup = _get_grouped_gemm_setup_workspace(device.index, num_tensors)
    # Each grouped-GEMM layout gets its own persistent cuBLAS workspace: two grouped
    # GEMMs sharing one workspace can deadlock under CUDA-graph replay (see
    # _get_grouped_cublas_workspace). wgrad (NT) is the case seen in TE; fprop (TN) and
    # dgrad (NN) have also been reported to conflict, so all three layouts are isolated.
    workspace_cublas = _get_grouped_cublas_workspace(device.index, layout)

    sm_count = get_sm_count()
    sm_count = sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count)))

    return grouped_gemm_impl(
        A,
        transa,
        B,
        transb,
        out,
        bias,
        bias_scale,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        use_split_accumulator,
        sm_count,
    )
