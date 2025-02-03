# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""
import functools
from typing import Iterable, Optional, Tuple, Union, List
import os
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import assert_dim_for_fp8_exec, get_sm_count

from ..tensor.quantized_tensor import Quantizer
from ..tensor._internal.float8_tensor_base import Float8TensorBase
from ..tensor._internal.mxfp8_tensor_base import MXFP8TensorBase

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
]


@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()


def swizzle_inputs(A: torch.Tensor, B: torch.Tensor, layout: str):
    """Swizzle gemm inputs and return original scaling factor inverses."""
    if not isinstance(A, MXFP8TensorBase) or not isinstance(B, MXFP8TensorBase):
        return None

    original_scale_inverses = (
        A._rowwise_scale_inv,
        A._columnwise_scale_inv,
        B._rowwise_scale_inv,
        B._columnwise_scale_inv,
    )

    if layout[0] == "T":
        A._rowwise_scale_inv = tex.rowwise_swizzle(A._rowwise_data, A._rowwise_scale_inv)
    else:
        A._columnwise_scale_inv = tex.columnwise_swizzle(
            A._columnwise_data, A._columnwise_scale_inv
        )

    if layout[1] == "N":
        B._rowwise_scale_inv = tex.rowwise_swizzle(B._rowwise_data, B._rowwise_scale_inv)
    else:
        B._columnwise_scale_inv = tex.columnwise_swizzle(
            B._columnwise_data, B._columnwise_scale_inv
        )

    return original_scale_inverses


def reset_swizzled_inputs(A, B, scale_inverses):
    """Reset the swizzled scale inverses after GEMM."""
    if scale_inverses is not None:
        (
            A._rowwise_scale_inv,
            A._columnwise_scale_inv,
            B._rowwise_scale_inv,
            B._columnwise_scale_inv,
        ) = scale_inverses


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
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
    # assert quantization_params is None, "FP8 output not supported yet"

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

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

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
    }

    original_scale_inverses = swizzle_inputs(A, B, layout)
    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
    reset_swizzled_inputs(A, B, original_scale_inverses)

    return out, bias_grad, gelu_input, extra_output


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    # assert [a.is_contiguous() for a in A]
    # assert [b.is_contiguous() for b in B]

    if isinstance(A[0], Float8TensorBase):
        for a, b in zip(A, B):
            assert_dim_for_fp8_exec(a._data)
            assert_dim_for_fp8_exec(b._data)

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].shape[1], dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

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
