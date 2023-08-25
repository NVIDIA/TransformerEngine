from __future__ import annotations
import subprocess
from ..utils import cache
import torch
from .. import cpp_extensions as _nvte
from .empty import empty
from . import execution_state


@cache
def _is_hopper():
    gpu_name = (
        subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        .decode("utf-8")
        .strip()
    )
    return "H100" in gpu_name


@cache
def _cublas_workspace():
    workspace_size = 33_554_432 if _is_hopper() else 4_194_304
    data = torch.empty(workspace_size, dtype=torch.int8, device="cuda")
    return _nvte.Tensor(
        _nvte.DType.Byte,
        data,
        torch.Tensor(device="cuda"),
        torch.Tensor(device="cuda"),
        torch.Tensor(device="cuda"),
    )


def _to_cublas_args(A: _nvte.Tensor, B: _nvte.Tensor, transA: bool, transB: bool):
    return B, A, not transA, not transB


def matmul_transpose(mat: _nvte.Tensor, mul: _nvte.Tensor, out_dtype: _nvte.DType):
    "returns mat @ mul^T"
    return matmul_transpose_add(mat, mul, empty(), out_dtype)


def matmul_transpose_gelu(mat: _nvte.Tensor, mul: _nvte.Tensor, out_dtype: _nvte.DType):
    "returns mat @ mul^T, GELU(mat @ mul^T)"
    return matmul_transpose_add_gelu(mat, mul, empty(), out_dtype)


def matmul_transpose_gelu_add(mat: _nvte.Tensor, mul: _nvte.Tensor, add: _nvte.Tensor):
    "returns mat @ mul^T, GELU(mat @ mul^T) + add"
    return matmul_transpose_add_gelu_add(mat, mul, empty(), add)


def matmul_transpose_add(
    mat: _nvte.Tensor, mul: _nvte.Tensor, add: _nvte.Tensor, out_dtype: _nvte.DType
):
    "returns mat @ mul^T + add"
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    out = empty((b.shape[0], a.shape[0]), out_dtype)
    _nvte.cublas_gemm(
        a,
        b,
        out,
        add,
        empty(),
        trans_a,
        trans_b,
        execution_state.pass_ == "backward",
        _cublas_workspace(),
        False,
        execution_state.pass_ == "backward",
        0,
    )
    return out


def matmul_transpose_add_gelu(
    mat: _nvte.Tensor, mul: _nvte.Tensor, add: _nvte.Tensor, out_dtype: _nvte.DType
):
    "returns mat @ mul^T + add, GELU(mat @ mul^T + add)"
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    out = empty((b.shape[0], a.shape[0]), out_dtype)
    pre_gelu = empty(out.shape, add.dtype)
    _nvte.cublas_gemm(
        a,
        b,
        out,
        add,
        pre_gelu,
        trans_a,
        trans_b,
        execution_state.pass_ == "backward",
        _cublas_workspace(),
        False,
        execution_state.pass_ == "backward",
        0,
    )
    return pre_gelu, out


def matmul_transpose_add_add(
    mat: _nvte.Tensor, mul: _nvte.Tensor, add1: _nvte.Tensor, add2: _nvte.Tensor
):
    "returns mat @ mul^T + add1 + add2"
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    _nvte.cublas_gemm(
        a,
        b,
        add2,
        add1,
        empty(),
        trans_a,
        trans_b,
        execution_state.pass_ == "backward",
        _cublas_workspace(),
        True,
        execution_state.pass_ == "backward",
        0,
    )
    return add2


def matmul_transpose_add_gelu_add(
    mat: _nvte.Tensor, mul: _nvte.Tensor, add1: _nvte.Tensor, add2: _nvte.Tensor
):
    "returns mat @ mul^T + add1, GELU(mat @ mul^T + add1) + add2"
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    pre_gelu = empty(add2.shape, add1.dtype)
    _nvte.cublas_gemm(
        a,
        b,
        add2,
        add1,
        pre_gelu,
        trans_a,
        trans_b,
        execution_state.pass_ == "backward",
        _cublas_workspace(),
        True,
        execution_state.pass_ == "backward",
        0,
    )
    return pre_gelu, add2
