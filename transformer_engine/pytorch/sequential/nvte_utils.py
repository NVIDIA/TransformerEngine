from functools import cache
import subprocess
from typing import Sequence
import torch
import transformer_engine_cuda as nvte


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
    return nvte.Tensor(
        nvte.DType.Byte, data, torch.Tensor(), torch.Tensor(), torch.Tensor()
    )


def _to_cublas_args(A: nvte.Tensor, B: nvte.Tensor, transA: bool, transB: bool):
    return B, A, not transA, not transB


def _is_during_backward() -> bool:
    raise NotImplementedError()  # TODO


def make_nvte_tensor(t: torch.Tensor):
    return nvte.Tensor(
        torch_to_te_dtype(t.dtype),
        t.data,
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
    )


# Wrappers around functions needing workspace
def _cast_transpose_dbias(
    input: nvte.Tensor,
    cast_output: nvte.Tensor,
    transposed_output: nvte.Tensor,
    dbias: nvte.Tensor,
):
    workspace_query = empty()
    nvte.cast_transpose_dbias(
        input, cast_output, transposed_output, dbias, workspace_query
    )
    workspace = empty_like(workspace_query)
    nvte.cast_transpose_dbias(input, cast_output, transposed_output, dbias, workspace)


# DTYPES
def te_to_torch_dtype(dtype: nvte.DType):
    match dtype:
        case nvte.DType.Byte:
            return torch.int8
        case nvte.DType.Int32:
            return torch.int32
        case nvte.DType.Int64:
            return torch.int64
        case nvte.DType.Float32:
            return torch.float32
        case nvte.DType.Float16:
            return torch.float16
        case nvte.DType.BFloat16:
            return torch.bfloat16
        case nvte.DType.Float8E4M3:
            return torch.int8
        case nvte.DType.Float8E5M2:
            return torch.int8


def torch_to_te_dtype(dtype: torch.dtype):
    match dtype:
        case torch.int:
            return nvte.DType.Int32
        case torch.int32:
            return nvte.DType.Int32
        case torch.int64:
            return nvte.DType.Int64
        case torch.float:
            return nvte.DType.Float32
        case torch.float32:
            return nvte.DType.Float32
        case torch.half:
            return nvte.DType.Float16
        case torch.float16:
            return nvte.DType.Float16
        case torch.bfloat16:
            return nvte.DType.BFloat16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def bit_width(dtype: nvte.DType):
    match dtype:
        case nvte.DType.Byte:
            return 8
        case nvte.DType.Int32:
            return 32
        case nvte.DType.Int64:
            return 64
        case nvte.DType.Float32:
            return 32
        case nvte.DType.Float16:
            return 16
        case nvte.DType.BFloat16:
            return 16
        case nvte.DType.Float8E4M3:
            return 8
        case nvte.DType.Float8E5M2:
            return 8


def is_fp8(t: nvte.Tensor | nvte.DType):
    if isinstance(t, nvte.Tensor):
        dtype = t.dtype
    else:
        dtype = t
    return dtype == nvte.DType.Float8E4M3 or dtype == nvte.DType.Float8E5M2


# ADD
def add(A: nvte.Tensor, B: nvte.Tensor, out_dtype: nvte.DType):
    if is_fp8(A) or is_fp8(B):
        raise NotImplementedError()
    else:
        output = torch.empty(A.shape, dtype=te_to_torch_dtype(out_dtype), device="cuda")
        torch.add(A.data, B.data, out=output)
        return make_nvte_tensor(output)


def dbias(t: nvte.Tensor, out_dtype: nvte.DType):
    if is_fp8(t):
        raise NotImplementedError()
    else:
        output = torch.sum(t.data, dtype=te_to_torch_dtype(out_dtype), dim=0)
        return make_nvte_tensor(output)


# CREATE
_AMAX_HISTORY_LEN = 512


def empty(shape: Sequence[int] = (), dtype: nvte.DType = nvte.DType.Float32):
    if is_fp8(dtype):
        return nvte.Tensor(
            dtype,
            torch.empty(
                _AMAX_HISTORY_LEN, dtype=te_to_torch_dtype(dtype), device="cuda"
            ),
            torch.empty(_AMAX_HISTORY_LEN, dtype=torch.float32, device="cuda"),
            torch.empty(1, dtype=torch.float32, device="cuda"),
            torch.empty(1, dtype=torch.float32, device="cuda"),
        )
    else:
        return nvte.Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )


def empty_like(t: nvte.Tensor):
    return empty(t.shape, t.dtype)


def multi_empty_share_metadata(*shapes_dtypes: tuple[Sequence[int], nvte.DType]):
    amax = torch.empty(_AMAX_HISTORY_LEN, dtype=torch.float32, device="cuda")
    scale = torch.empty(1, dtype=torch.float32, device="cuda")
    scale_inv = torch.empty(1, dtype=torch.float32, device="cuda")

    return tuple(
        nvte.Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            amax,
            scale,
            scale_inv,
        )
        if is_fp8(dtype)
        else nvte.Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
        for shape, dtype in shapes_dtypes
    )


# CAST + TRANPOSE
def cast(t: nvte.Tensor, dtype: nvte.DType):
    assert t.dtype != dtype
    assert is_fp8(t) != is_fp8(dtype)

    output = empty(t.shape, dtype)
    if is_fp8(dtype):
        nvte.fp8_quantize(t, output)
    elif is_fp8(t):
        nvte.fp8_dequantize(t, output)
    else:
        output.data.copy_(t.data)

    return output


def cast_checked(t: nvte.Tensor, dtype: nvte.DType | None):
    if dtype is None or t.dtype == dtype:
        return t
    else:
        return cast(t, dtype)


def transpose(t: nvte.Tensor):
    output = empty(t.shape[::-1], t.dtype)
    nvte.transpose(t, output)
    return output


def cast_transpose(t: nvte.Tensor, dtype: nvte.DType):
    assert t.dtype != dtype
    assert is_fp8(t) != is_fp8(dtype)

    out_cast, out_transpose = multi_empty_share_metadata(
        (t.shape, dtype), (t.shape[::-1], dtype)
    )

    nvte.cast_transpose(t, out_cast, out_transpose)
    return out_cast, out_transpose


def cast_transpose_checked(t: nvte.Tensor, dtype: nvte.DType | None):
    if dtype is None or t.dtype == dtype:
        return t, transpose(t)
    else:
        return cast_transpose(t, dtype)


def multi_cast_transpose(*desc: tuple[nvte.Tensor, nvte.DType]):
    outs = [
        multi_empty_share_metadata((t.shape, dtype), (t.shape[::-1], dtype))
        for t, dtype in desc
    ]
    out_cast_list, out_transpose_list = zip(*outs)
    input_list, _ = zip(*desc)
    nvte.multi_cast_transpose(input_list, out_cast_list, out_transpose_list)  # type: ignore
    return outs


def multi_cast_transpose_checked(*desc: tuple[nvte.Tensor, nvte.DType | None]):
    transpose_results = list[tuple[nvte.Tensor, nvte.Tensor] | None]()
    to_cast_transpose = list[tuple[nvte.Tensor, nvte.DType]]()
    for t, dtype in desc:
        if dtype is None or t.dtype == dtype:
            transpose_results.append((t, transpose(t)))
        else:
            to_cast_transpose.append((t, dtype))
            transpose_results.append(None)
    cast_transpose_results = multi_cast_transpose(*to_cast_transpose)
    results = list[tuple[nvte.Tensor, nvte.Tensor]]()
    i = 0
    for result in transpose_results:
        if result is None:
            results.append(cast_transpose_results[i])
            i += 1
        else:
            results.append(result)
    return results


def cast_transpose_dbias_checked(
    t: nvte.Tensor, cast_dtype: nvte.DType | None, dbias_dtype: nvte.DType
):
    if dbias_dtype == t.dtype and cast_dtype is not None and cast_dtype != t.dtype:
        out_cast, out_transpose = multi_empty_share_metadata(
            (t.shape, cast_dtype), (t.shape[::-1], cast_dtype)
        )
        out_dbias = empty((t.shape[1],), dbias_dtype)
        _cast_transpose_dbias(t, out_cast, out_transpose, out_dbias)
        return out_cast, out_transpose, out_dbias
    else:
        out_cast, out_transpose = cast_transpose_checked(t, cast_dtype)
        out_dbias = dbias(t, dbias_dtype)
        return out_cast, out_transpose, out_dbias


# MATMUL TRANSPOSE
def matmul_transpose(mat: nvte.Tensor, mul: nvte.Tensor, out_dtype: nvte.DType):
    "returns mat @ mul^T"
    return matmul_transpose_add(mat, mul, empty(), out_dtype)


def matmul_transpose_gelu(mat: nvte.Tensor, mul: nvte.Tensor, out_dtype: nvte.DType):
    "returns mat @ mul^T, GELU(mat @ mul^T)"
    return matmul_transpose_add_gelu(mat, mul, empty(), out_dtype)


def matmul_transpose_add(
    mat: nvte.Tensor, mul: nvte.Tensor, add: nvte.Tensor, out_dtype: nvte.DType
):
    "returns mat @ mul^T + add"
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    out = empty((b.shape[0], a.shape[0]), out_dtype)
    nvte.cublas_gemm(
        a,
        b,
        out,
        add,
        empty(),
        trans_a,
        trans_b,
        _is_during_backward(),
        _cublas_workspace(),
        False,
        _is_during_backward(),
        0,
    )
    return out


def matmul_transpose_add_gelu(
    mat: nvte.Tensor, mul: nvte.Tensor, add: nvte.Tensor, out_dtype: nvte.DType
):
    "returns mat @ mul^T + add, GELU(mat @ mul^T + add)"
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    out = empty((b.shape[0], a.shape[0]), out_dtype)
    pre_gelu = empty(out.shape, add.dtype)
    nvte.cublas_gemm(
        a,
        b,
        out,
        add,
        pre_gelu,
        trans_a,
        trans_b,
        _is_during_backward(),
        _cublas_workspace(),
        False,
        _is_during_backward(),
        0,
    )
    return pre_gelu, out


def matmul_transpose_add_add(
    mat: nvte.Tensor, mul: nvte.Tensor, add1: nvte.Tensor, add2: nvte.Tensor
):
    "returns mat @ mul^T + add1 + add2"
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    nvte.cublas_gemm(
        a,
        b,
        add2,
        add1,
        empty(),
        trans_a,
        trans_b,
        _is_during_backward(),
        _cublas_workspace(),
        True,
        _is_during_backward(),
        0,
    )
    return add2


def matmul_transpose_add_gelu_add(
    mat: nvte.Tensor, mul: nvte.Tensor, add1: nvte.Tensor, add2: nvte.Tensor
):
    "returns mat @ mul^T + add1, GELU(mat @ mul^T + add1) + add2"
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    pre_gelu = empty(add2.shape, add1.dtype)
    nvte.cublas_gemm(
        a,
        b,
        add2,
        add1,
        pre_gelu,
        trans_a,
        trans_b,
        _is_during_backward(),
        _cublas_workspace(),
        True,
        _is_during_backward(),
        0,
    )
    return pre_gelu, add2
