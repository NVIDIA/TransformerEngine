# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""
import functools
from typing import Optional, Tuple, Union, List
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import assert_dim_for_fp8_exec


__all__ = [
    "gemm",
    "fp8_gemm",
    "grouped_gemm",
    "fp8_grouped_gemm",
]


@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor()


def fp8_gemm(
    A: torch.Tensor,
    A_scale_inv: torch.Tensor,
    A_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    A_dtype: tex.DType,
    B: torch.Tensor,
    B_scale_inv: torch.Tensor,
    B_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    B_dtype: tex.DType,
    out_dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    accumulate: bool = False,
    out: Optional[torch.Tensor] = None,
    out_index=None,
    fp8_meta_tensor: tex.FP8TensorMeta = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    ub_algo: tex.UbufOverlapAlgo = None,
    ub: Union[tex.UbufCommOverlap, tex.UbufP2PCommOverlap] = None,
    extra_output_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """TN layout GEMM with fp8 inputs."""

    empty_tensor = _empty_tensor()
    if D_dtype is not None and D_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        assert fp8_meta_tensor is not None and out_index is not None
    assert_dim_for_fp8_exec(A)
    assert_dim_for_fp8_exec(B)
    assert A.dtype == torch.uint8
    assert B.dtype == torch.uint8

    if out is None:
        out = torch.empty(
            B.shape[0],
            A.shape[0],
            dtype=out_dtype,
            device="cuda",
        )
    else:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # Use bfloat16 as default bias_dtype
    bias_dtype = torch.bfloat16 if bias is None else bias.dtype
    if gelu:
        gelu_input = torch.empty_like(out, dtype=bias_dtype)
    else:
        gelu_input = empty_tensor
    bias_dtype = TE_DType[bias_dtype]

    out_dtype = TE_DType[out.dtype] if D_dtype is None else D_dtype

    args = (
        A,
        A_scale_inv,
        A_fp8_tensor,
        A_dtype,
        True,  # transa
        B,
        B_scale_inv,
        B_fp8_tensor,
        B_dtype,
        False,  # transb
        out,
        empty_tensor if out_index is None else fp8_meta_tensor.scale[out_index],
        out_dtype,
        empty_tensor if out_index is None else fp8_meta_tensor.amax_history[0][out_index],
        bias if use_bias else empty_tensor,
        bias_dtype,
        gelu_input,  # this is pre_gelu_out
        False,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    fn = torch.ops.tex_ts.te_gemm_ts
    if ub_algo is not None:
        assert ub is not None, "ub object is None!"
        if ub_algo == tex.UbufOverlapAlgo.BULK_OVERLAP_AG:
            fn = ub.bulk_overlap
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(
                args
                + (
                    1,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.UbufOverlapAlgo.BULK_OVERLAP_RS:
            fn = ub.bulk_overlap
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(
                args
                + (
                    0,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P:
            fn = ub.split_overlap_ag_p2p
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.ATOMIC_GEMM_AG_P2P:
            fn = ub.atomic_gemm_overlap_ag_p2p
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS:
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
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
            fn = ub.split_overlap_rs_p2p
            assert (
                extra_output_tensor is not None
            ), "SPLIT_PIPELINED_RS_P2P requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.ATOMIC_GEMM_RS:
            fn = ub.atomic_gemm_overlap_rs
            assert extra_output_tensor is not None, "ATOMIC_GEMM_RS requires extra output tensor"
            args = tuple(
                args
                + (
                    True,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.UbufOverlapAlgo.ATOMIC_GEMM_RS_P2P:
            fn = ub.atomic_gemm_overlap_rs_p2p
            assert (
                extra_output_tensor is not None
            ), "ATOMIC_GEMM_RS_P2P requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
    if ub_algo is not None and ub_algo == tex.UbufOverlapAlgo.ATOMIC_GEMM_AG_P2P:
        out = fn(*args)
    else:
        _ = fn(*args)

    return out, gelu_input


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    gelu_input: Optional[torch.Tensor] = None,
    grad: bool = False,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    ub_algo: tex.UbufOverlapAlgo = None,
    ub: tex.UbufCommOverlap = None,
    extra_output_tensor: torch.Tensor = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Non FP8 GEMM."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    empty_tensor = _empty_tensor()
    fp8_index = -1  # dummy index

    if out is None:
        out = torch.empty(
            B.shape[1] if transb else B.shape[0],
            A.shape[0] if transa else A.shape[1],
            dtype=dtype,
            device="cuda",
        )
    else:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    if gelu and not grad:
        gelu_input = torch.empty_like(out, dtype=dtype)
    elif not gelu:
        gelu_input = empty_tensor

    if grad and use_bias:
        grad_bias = torch.empty(B.shape[1], dtype=out.dtype, device="cuda")
    else:
        grad_bias = empty_tensor

    bias = bias if use_bias else empty_tensor

    assert (
        A.dtype == dtype and B.dtype == dtype
    ), f"Expected dtype={dtype}, but found A.dtype={A.dtype} and B.dtype={B.dtype}"
    input_dtype = TE_DType[dtype]
    output_dtype = TE_DType[out.dtype]
    if use_bias:
        bias_dtype = TE_DType[grad_bias.dtype] if grad else TE_DType[bias.dtype]
    else:
        bias_dtype = output_dtype

    args = (
        A,
        empty_tensor,
        fp8_index,
        input_dtype,
        transa,
        B,
        empty_tensor,
        fp8_index,
        input_dtype,
        transb,
        out,
        empty_tensor,  # out_scale
        output_dtype,
        empty_tensor,  # out_amax
        grad_bias if grad else bias,
        bias_dtype,
        gelu_input,
        grad,
        workspace,
        workspace.shape[0],
        accumulate,
        False,  # use_split_accumulator
    )
    fn = torch.ops.tex_ts.te_gemm_ts
    if ub_algo is not None:
        assert ub is not None, "ub object is None!"
        if ub_algo == tex.UbufOverlapAlgo.BULK_OVERLAP_AG:
            fn = ub.bulk_overlap
            args = tuple(args + (1, empty_tensor))
        elif ub_algo == tex.UbufOverlapAlgo.BULK_OVERLAP_RS:
            fn = ub.bulk_overlap
            args = tuple(args + (0, empty_tensor))
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P:
            fn = ub.split_overlap_ag_p2p
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS:
            fn = ub.split_overlap_rs
            assert (
                extra_output_tensor is not None
            ), "SPLIT_PIPELINED_RS requires extra output tensor"
            args = tuple(
                args
                + (
                    False,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
            fn = ub.split_overlap_rs_p2p
            assert (
                extra_output_tensor is not None
            ), "SPLIT_PIPELINED_RS_P2P requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
    _ = fn(*args)

    return out, grad_bias, gelu_input


def grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    gelu: bool = False,
    gelu_input: Optional[List[torch.Tensor]] = None,
    grad: bool = False,
    accumulate: bool = False,
    layout: str = "TN",
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
) -> Tuple[Union[List[torch.Tensor], None], ...]:
    """Non FP8 Grouped GEMM."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    num_gemms = len(A)
    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    if gelu and not grad:
        gelu_input = [
            torch.empty_like(o, dtype=dtype, memory_format=torch.contiguous_format) for o in out
        ]
    elif not gelu:
        gelu_input = empty_tensors

    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].shape[1], dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors

    bias = bias if use_bias else empty_tensors

    assert (
        A[0].dtype == dtype and B[0].dtype == dtype
    ), f"Expected dtype={dtype}, but found A.dtype={A[0].dtype} and B.dtype={B[0].dtype}"
    input_dtype = TE_DType[dtype]
    output_dtype = TE_DType[out[0].dtype]
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = output_dtype

    torch.ops.tex_ts.te_grouped_gemm_ts(
        A,
        empty_tensor,
        0,  # A_offset
        input_dtype,
        transa,
        B,
        empty_tensor,
        0,  # B_offset
        input_dtype,
        transb,
        out,
        0,  # out_offset
        empty_tensor,  # out_scale
        output_dtype,
        empty_tensor,  # out_amax
        grad_bias if grad else bias,
        bias_dtype,
        gelu_input,  # gelu_input
        grad,
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        False,  # use_split_accumulator
    )

    return out, grad_bias, gelu_input


def fp8_grouped_gemm(
    A: List[torch.Tensor],
    A_scale_inv: Union[torch.Tensor, List[torch.Tensor]],
    A_fp8_tensor_offset: int,
    A_dtype: tex.DType,
    B: List[torch.Tensor],
    B_scale_inv: torch.Tensor,
    B_fp8_tensor_offset: int,
    B_dtype: tex.DType,
    out: Union[torch.Tensor, List[torch.Tensor]],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    m_splits: Optional[List[int]] = None,
    out_offset: Optional[int] = None,
    fp8_meta_tensor: tex.FP8TensorMeta = None,
    gelu: bool = False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
) -> Tuple[Union[List[torch.Tensor], torch.Tensor, None], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    This function accepts two combinations of inputs:
        1. A_scale_inv is a list of tensors, out is a single tensor, and m_splits is not None.
           This is used for the calculation of output (fwd) and dgrad (bwd).
        2. A_scale_inv is a single tensor, out is a list of tensors. This is used for the
           calculation of wgrad.
    """
    if isinstance(A_scale_inv, list):
        assert isinstance(out, torch.Tensor) and m_splits is not None
    elif isinstance(A_scale_inv, torch.Tensor):
        assert isinstance(out, (list, tuple))
    else:
        raise ValueError("A_scale_inv should be a list of tensors or a single tensor.")

    num_gemms = len(A)
    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms
    if D_dtype is not None and D_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        assert fp8_meta_tensor is not None and out_offset is not None
    for a, b in zip(A, B):
        assert_dim_for_fp8_exec(a)
        assert_dim_for_fp8_exec(b)
    assert A[0].dtype == torch.uint8
    assert B[0].dtype == torch.uint8

    # Use bfloat16 as default bias_dtype
    bias_dtype = torch.bfloat16 if bias is None else bias[0].dtype
    bias_dtype = TE_DType[bias_dtype]
    gelu_input = empty_tensors

    if not isinstance(out, torch.Tensor):
        if gelu:
            gelu_input = [
                torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
                for o in out
            ]
        out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

        torch.ops.tex_ts.te_grouped_gemm_ts(
            A,
            A_scale_inv,
            A_fp8_tensor_offset,
            A_dtype,
            True,  # transa
            B,
            B_scale_inv,
            B_fp8_tensor_offset,
            B_dtype,
            False,  # transb
            out,
            0 if out_offset is None else out_offset,
            empty_tensor if out_offset is None else fp8_meta_tensor.scale,
            out_dtype,
            empty_tensor if out_offset is None else fp8_meta_tensor.amax_history,
            bias if use_bias else empty_tensors,
            bias_dtype,
            gelu_input,  # this is pre_gelu_out
            False,  # grad
            workspaces,
            workspaces[0].shape[0],
            accumulate,
            use_split_accumulator,
        )
    else:
        if gelu:
            gelu_input = [torch.empty((m, A[0].size(0)), dtype=bias_dtype) for m in m_splits]
        out_dtype = TE_DType[out.dtype] if D_dtype is None else D_dtype

        torch.ops.tex_ts.te_grouped_gemm_single_output_ts(
            A,
            A_scale_inv,
            A_fp8_tensor_offset,
            A_dtype,
            True,  # transa
            B,
            B_scale_inv,
            B_fp8_tensor_offset,
            B_dtype,
            False,  # transb
            m_splits,
            out,
            0 if out_offset is None else out_offset,
            empty_tensor if out_offset is None else fp8_meta_tensor.scale,
            out_dtype,
            empty_tensor if out_offset is None else fp8_meta_tensor.amax_history,
            bias if use_bias else empty_tensors,
            bias_dtype,
            gelu_input,  # this is pre_gelu_out
            False,  # grad
            workspaces,
            workspaces[0].shape[0],
            accumulate,
            use_split_accumulator,
        )

    return out, gelu_input
