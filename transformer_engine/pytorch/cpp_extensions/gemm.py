# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import os
import functools
import torch
import transformer_engine_torch as tex
from ..constants import NVFP4_BLOCK_SCALING_SIZE, TE_DType, DType
from ..utils import get_sm_count, _empty_tensor

from ..quantized_tensor import Quantizer
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ..tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage
from ..tensor.utils import is_custom
from ..custom_recipes.gemm import custom_gemm
from ...debug.pytorch.debug_quantization import DebugQuantizer


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
) -> Tuple[NVFP4TensorStorage, NVFP4TensorStorage, torch.Tensor]:
    """Return GEMM aliases and FP32 output scales for row-scaled NVFP4."""
    A_metadata = A.get_metadata()
    weight_amax = A._amax_rowwise if transa else A._amax_columnwise
    assert weight_amax is not None and weight_amax.numel() == 1
    A_metadata["amax_rowwise" if transa else "amax_columnwise"] = weight_amax.new_ones(1)
    A_metadata["row_scaled_nvfp4"] = False

    B_metadata = B.get_metadata()
    rhs_rowwise_amax = B._amax_rowwise
    assert rhs_rowwise_amax is not None
    B_metadata["amax_rowwise"] = rhs_rowwise_amax.new_ones(1)
    B_metadata["row_scaled_nvfp4"] = False

    assert rhs_rowwise_amax.dtype == torch.float32 and weight_amax.dtype == torch.float32
    return (
        NVFP4TensorStorage(**A_metadata),
        NVFP4TensorStorage(**B_metadata),
        (rhs_rowwise_amax * weight_amax).view(-1, 1),
    )


def _cudnn_row_scaled_nvfp4_grouped_gemm(
    weights: List[NVFP4TensorStorage],
    inputs: List[NVFP4TensorStorage],
    outputs: List[torch.Tensor],
    *,
    m_splits: Optional[List[int]],
    bias: Optional[List[torch.Tensor]],
    single_output: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Run tensor-scaled weights and row-scaled inputs with the cuDNN MoE kernel."""
    num_gemms = len(weights)
    if num_gemms == 0 or len(inputs) != num_gemms:
        raise ValueError("Grouped GEMM requires matching non-empty weight and input lists.")
    if not all(isinstance(tensor, NVFP4TensorStorage) for tensor in weights + inputs):
        raise TypeError("cuDNN row-scaled NVFP4 grouped GEMM requires NVFP4 inputs.")
    if any(_is_nvfp4_row_scaled_tensor(tensor) for tensor in weights):
        raise NotImplementedError(
            "cuDNN row-scaled NVFP4 grouped GEMM requires tensor-scaled weights."
        )
    if not all(_is_nvfp4_row_scaled_tensor(tensor) for tensor in inputs):
        raise NotImplementedError("cuDNN row-scaled NVFP4 grouped GEMM requires row-scaled inputs.")
    if any(getattr(tensor, "_nvfp4_use_4over6", False) for tensor in weights + inputs):
        raise NotImplementedError("cuDNN row-scaled NVFP4 grouped GEMM does not support 4over6.")

    m_splits_list = (
        list(m_splits) if m_splits is not None else [int(tensor.size(0)) for tensor in inputs]
    )
    if len(m_splits_list) != num_gemms:
        raise ValueError("m_splits length must match the number of grouped GEMMs.")
    if any(m % 256 != 0 for m in m_splits_list):
        raise NotImplementedError(
            "cuDNN row-scaled NVFP4 grouped GEMM requires M multiples of 256."
        )

    k = int(inputs[0].size(1))
    n = int(weights[0].size(0))
    if k % 128 != 0 or n % 128 != 0:
        raise NotImplementedError(
            "cuDNN row-scaled NVFP4 grouped GEMM requires K and N multiples of 128."
        )
    if any(tuple(tensor.size()) != (n, k) for tensor in weights):
        raise ValueError("All grouped GEMM weights must have the same (N, K) shape.")
    if any(
        int(tensor.size(0)) != m or int(tensor.size(1)) != k
        for tensor, m in zip(inputs, m_splits_list)
    ):
        raise ValueError("Grouped GEMM input shapes must match m_splits and K.")
    expected_output_rows = [sum(m_splits_list)] if single_output else m_splits_list
    if len(outputs) != len(expected_output_rows) or any(
        output.shape[-1] != n or output.numel() != m * n
        for output, m in zip(outputs, expected_output_rows)
    ):
        raise ValueError("Grouped GEMM output shapes do not match m_splits and N.")
    if outputs[0].dtype not in (torch.bfloat16, torch.float16) or any(
        output.dtype != outputs[0].dtype for output in outputs
    ):
        raise NotImplementedError(
            "cuDNN row-scaled NVFP4 grouped GEMM supports uniform BF16/FP16 outputs only."
        )
    if bias is not None and (
        len(bias) != num_gemms
        or any(tensor is None or tuple(tensor.size()) != (n,) for tensor in bias)
    ):
        raise ValueError("Grouped GEMM bias tensors must have shape (N,).")

    from cudnn import (  # pylint: disable=import-outside-toplevel,no-name-in-module
        grouped_gemm_quant_wrapper_sm100,
    )

    device = inputs[0]._rowwise_data.device
    total_m = sum(m_splits_list)

    a_data = torch.cat(
        [tensor._rowwise_data.view(m, k // 2) for tensor, m in zip(inputs, m_splits_list)],
        dim=0,
    )
    a_tensor = a_data.view(torch.float4_e2m1fn_x2).unsqueeze(0).permute(1, 2, 0)

    sfa_compact = torch.cat([tensor._rowwise_scale_inv for tensor in inputs], dim=0)
    sfa_logical = sfa_compact.view(dtype=torch.float8_e4m3fn).view(
        1,
        total_m // 128,
        4,
        32,
        k // (4 * NVFP4_BLOCK_SCALING_SIZE),
        4,
    )
    sfa_logical = sfa_logical.permute(3, 2, 1, 5, 4, 0)
    sfa_storage = torch.empty(
        (1, total_m // 128, k // (4 * NVFP4_BLOCK_SCALING_SIZE), 32, 4, 4),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    sfa_tensor = sfa_storage.permute(3, 4, 1, 5, 2, 0)
    sfa_tensor.copy_(sfa_logical)

    b_ptrs, sfb_ptrs, _sfb_buffer = (
        tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
            [tensor._rowwise_data for tensor in weights],
            [tensor._rowwise_scale_inv for tensor in weights],
            "nvfp4",
            device,
        )
    )

    weight_amaxes = [tensor._amax_rowwise for tensor in weights]
    input_amaxes = [tensor._amax_rowwise for tensor in inputs]
    if any(amax is None or amax.numel() != 1 for amax in weight_amaxes):
        raise ValueError("Row-scaled NVFP4 grouped GEMM requires tensor-scaled weights.")
    if any(amax is None or amax.numel() != m for amax, m in zip(input_amaxes, m_splits_list)):
        raise ValueError("Row-scaled NVFP4 grouped GEMM requires one input scale per row.")
    alpha_tensor = torch.cat(
        [
            amax.view(-1) / (float(tensor._nvfp4_e4m3_max) * 6.0)
            for tensor, amax in zip(weights, weight_amaxes)
        ]
    ).to(dtype=torch.float32)
    row_scale_tensor = torch.cat(
        [
            amax.view(-1) / (float(tensor._nvfp4_e4m3_max) * 6.0)
            for tensor, amax in zip(inputs, input_amaxes)
        ]
    ).to(dtype=torch.float32)

    bias_tensor = None if bias is None else torch.stack(bias, dim=0).transpose(0, 1)

    padded_offsets = torch.tensor(m_splits_list, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    prob_tensor = torch.ones(total_m, 1, 1, dtype=torch.float32, device=device)

    result = grouped_gemm_quant_wrapper_sm100(
        a_tensor=a_tensor,
        b_ptrs=b_ptrs,
        sfa_tensor=sfa_tensor,
        sfb_ptrs=sfb_ptrs,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        bias_tensor=bias_tensor,
        norm_const_tensor=None,
        prob_tensor=prob_tensor,
        row_scale_tensor=row_scale_tensor,
        acc_dtype=torch.float32,
        d_dtype=outputs[0].dtype,
        cd_major="n",
        sf_vec_size=NVFP4_BLOCK_SCALING_SIZE,
        discrete_col_sfd=False,
        b_dtype=torch.float4_e2m1fn_x2,
        b_major="k",
        n=n,
        current_stream=torch.cuda.current_stream().cuda_stream,
        use_dynamic_sched=True,
    )
    d_tensor = result["d_tensor"].squeeze(-1)

    if single_output:
        outputs[0].view(total_m, n).copy_(d_tensor)
        return outputs[0]

    for output, output_data in zip(outputs, d_tensor.split(m_splits_list)):
        output.view(-1, n).copy_(output_data)
    return outputs


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

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)
    workspace = get_cublas_workspace(A.device.index, ub is not None, False)

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

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer
        A = A.get_tensor(not transa)
        B = B.get_tensor(transb)

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
        if _is_nvfp4_row_scaled_tensor(A):
            raise NotImplementedError("Row-scaled NVFP4 GEMM does not support row-scaled A.")
        assert layout[1] == "N", "Row-scaled NVFP4 GEMM currently supports N-layout B only."
        if grad:
            raise RuntimeError(
                "Row-scaled NVFP4 GEMM currently supports fprop only. "
                "Backward NVFP4 gradient quantizers should use scalar global amax."
            )
        assert not gelu, "Row-scaled NVFP4 GEMM currently does not support fused GELU."
        assert not accumulate, "Row-scaled NVFP4 GEMM currently does not support accumulation."
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
        # cuBLAS folds NVFP4 global amax values into GEMM alpha. Keep the row-scaled
        # recipe's global scales out of alpha and apply them in FP32 below.
        gemm_A, gemm_B, rowwise_global_scales = _nvfp4_row_scaled_gemm_inputs(A, B, transa=transa)

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
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*gemm_args, **kwargs)
        out_2d = out.reshape(-1, out.shape[-1])

        assert rowwise_global_scales.dtype == torch.float32 and out.dtype == torch.float32
        assert rowwise_global_scales.numel() == out_2d.shape[0]

        out_2d.mul_(rowwise_global_scales)
        if bias is not None:
            out_2d.add_(bias.to(dtype=torch.float32))

        if requested_out is not None:
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

    if any(_is_nvfp4_row_scaled_tensor(tensor) for tensor in A):
        raise NotImplementedError("Row-scaled NVFP4 grouped GEMM does not support row-scaled A.")
    if any(_is_nvfp4_row_scaled_tensor(tensor) for tensor in B):
        if D_dtype is not None:
            raise NotImplementedError(
                "cuDNN row-scaled NVFP4 grouped GEMM does not support D_dtype."
            )
        if layout != "TN":
            raise NotImplementedError(
                "cuDNN row-scaled NVFP4 grouped GEMM supports TN layout only."
            )
        if grad or gelu or accumulate or use_split_accumulator:
            raise NotImplementedError(
                "cuDNN row-scaled NVFP4 grouped GEMM supports fprop without GELU, "
                "accumulation, or split accumulator only."
            )
        if any(quantizer is not None for quantizer in quantization_params):
            raise NotImplementedError(
                "cuDNN row-scaled NVFP4 grouped GEMM does not support output quantization."
            )
        return (
            _cudnn_row_scaled_nvfp4_grouped_gemm(
                A,
                B,
                out,
                m_splits=m_splits,
                bias=bias if use_bias else None,
                single_output=single_output,
            ),
            grad_bias,
            gelu_input,
        )

    if isinstance(quantization_params[0], DebugQuantizer):
        assert not gelu, "GELU not supported in debug mode"
        if single_output:
            out_init = out[0]
            start_idx = 0
            out = [None] * num_gemms
            for i in range(num_gemms):
                size = m_splits[i]
                out[i] = out_init[start_idx : start_idx + size]
                start_idx += size
        for i in range(num_gemms):
            _, bias_or_grad, _, _ = general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out[0].dtype,
                layout=layout,
                accumulate=accumulate,
                out=out[i],
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
            if grad and use_bias:
                grad_bias[i] = bias_or_grad
        if single_output:
            out = out_init

        return out, grad_bias if grad else bias, None

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

    workspace_setup = torch.empty(
        get_grouped_gemm_setup_workspace_size(num_tensors),
        dtype=torch.uint8,
        device=device,
    )
    workspace_cublas = torch.empty(
        get_cublas_workspace_size_bytes(),
        dtype=torch.uint8,
        device=device,
    )

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
