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
    ub_algo: tex.CommOverlapAlgo = None,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_buffer: Optional[torch.Tensor] = None,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    # assert quantization_params is None, "FP8 output not supported yet"
    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # Use bfloat16 as default bias_dtype
    bias_dtype = torch.bfloat16 if bias is None else bias.dtype
    bias_dtype = TE_DType[bias_dtype]
    if bias is None and not grad:
        bias = _empty_tensor()

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

    fn = tex.generic_gemm
    if ub_algo is not None:
        raise ValueError("Not implemented yet!")
        if ub_algo == tex.CommOverlapAlgo.BULK_OVERLAP_AG:
            fn = ub.bulk_overlap
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(
                args
                + (
                    tex.CommOverlapType.AG,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.CommOverlapAlgo.BULK_OVERLAP_RS:
            fn = ub.bulk_overlap
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(
                args
                + (
                    tex.CommOverlapType.RS,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P:
            fn = ub.split_overlap_ag_p2p
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_AG_P2P:
            assert A_scaling_mode == [-1, -1, 1] and B_scaling_mode == [
                -1,
                -1,
                1,
            ], "Block scaling unsupported for atomic GEMM."
            fn = ub.atomic_gemm_overlap_ag_p2p
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.CommOverlapAlgo.SPLIT_PIPELINED_RS:
            fn = ub.split_overlap_rs
            assert (
                extra_output_tensor is not None
            ), "SPLIT_PIPELINED_RS requires extra output tensor"
            args = tuple(
                args
                + (
                    True,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
            fn = ub.split_overlap_rs_p2p
            assert (
                extra_output_tensor is not None
            ), "SPLIT_PIPELINED_RS_P2P requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_RS:
            assert A_scaling_mode == [-1, -1, 1] and B_scaling_mode == [
                -1,
                -1,
                1,
            ], "Block scaling unsupported for atomic GEMM."
            fn = ub.atomic_gemm_overlap_rs
            assert extra_output_tensor is not None, "ATOMIC_GEMM_RS requires extra output tensor"
            args = tuple(
                args
                + (
                    True,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_RS_P2P:
            assert A_scaling_mode == [-1, -1, 1] and B_scaling_mode == [
                -1,
                -1,
                1,
            ], "Block scaling unsupported for atomic GEMM."
            fn = ub.atomic_gemm_overlap_rs_p2p
            assert (
                extra_output_tensor is not None
            ), "ATOMIC_GEMM_RS_P2P requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
    if ub_algo is not None and ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_AG_P2P:
        out = fn(*args)
        gelu_input = None
        bias_grad = None
    else:
        original_scale_inverses = swizzle_inputs(A, B, layout)
        out, bias_grad, gelu_input = fn(*args)
        reset_swizzled_inputs(A, B, original_scale_inverses)

    return out, bias_grad, gelu_input


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
