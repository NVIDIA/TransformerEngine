from functools import cache
import os
import subprocess
from typing import Literal, Sequence
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


@cache
def _fwd_ln_sm_margin():
    return int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))


@cache
def _bwd_ln_sm_margin():
    return int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))


@cache
def _sm_total_count() -> int:
    return torch.cuda.get_device_properties(  # type: ignore
        torch.cuda.current_device()
    ).multi_processor_count


def _sm_margin():
    if _pass == "backward":
        return _bwd_ln_sm_margin()
    elif _pass == "forward":
        return _fwd_ln_sm_margin()
    else:
        return 0


def _to_cublas_args(A: nvte.Tensor, B: nvte.Tensor, transA: bool, transB: bool):
    return B, A, not transA, not transB


def set_current_pass(pass_: Literal["forward", "backward", "inference"]):
    global _pass
    _pass = pass_


def make_nvte_tensor(t: torch.Tensor):
    return nvte.Tensor(
        torch_to_te_dtype(t.dtype),
        t.data,
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
    )


# DTYPES
def te_to_torch_dtype(dtype: nvte.DType):
    match dtype:
        case nvte.DType.Byte:
            return torch.uint8
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


def dbias(grad: nvte.Tensor, out_dtype: nvte.DType):
    if is_fp8(grad):
        raise NotImplementedError()
    else:
        output = torch.sum(grad.data, dtype=te_to_torch_dtype(out_dtype), dim=0)
        return make_nvte_tensor(output)


# CREATE
_AMAX_HISTORY_LEN = 512


def empty(shape: Sequence[int] = (), dtype: nvte.DType = nvte.DType.Float32):
    if shape == ():
        return nvte.Tensor(
            dtype,
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
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
    if is_fp8(t):
        assert not is_fp8(dtype)

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
    grad: nvte.Tensor, cast_dtype: nvte.DType | None, dbias_dtype: nvte.DType
):
    if (
        dbias_dtype == grad.dtype
        and cast_dtype is not None
        and cast_dtype != grad.dtype
    ):
        out_cast, out_transpose = multi_empty_share_metadata(
            (grad.shape, cast_dtype), (grad.shape[::-1], cast_dtype)
        )
        out_dbias = empty((grad.shape[1],), dbias_dtype)
        workspace = empty()
        for _ in range(2):
            nvte.cast_transpose_dbias(
                grad, out_cast, out_transpose, out_dbias, workspace
            )
            workspace = empty_like(workspace)
        return out_cast, out_transpose, out_dbias
    else:
        out_cast, out_transpose = cast_transpose_checked(grad, cast_dtype)
        out_dbias = dbias(grad, dbias_dtype)
        return out_cast, out_transpose, out_dbias


# MATMUL TRANSPOSE
def matmul_transpose(mat: nvte.Tensor, mul: nvte.Tensor, out_dtype: nvte.DType):
    "returns mat @ mul^T"
    # TODO: this should be allowed, though cublaslt_gemm cannot be used in this case
    assert mat.dtype == mul.dtype
    return matmul_transpose_add(mat, mul, empty(), out_dtype)


def matmul_transpose_gelu(mat: nvte.Tensor, mul: nvte.Tensor, out_dtype: nvte.DType):
    "returns mat @ mul^T, GELU(mat @ mul^T)"
    assert mat.dtype == mul.dtype
    return matmul_transpose_add_gelu(mat, mul, empty(), out_dtype)


def matmul_transpose_add(
    mat: nvte.Tensor, mul: nvte.Tensor, add: nvte.Tensor, out_dtype: nvte.DType
):
    "returns mat @ mul^T + add"
    assert mat.dtype == mul.dtype
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
        _pass == "backward",
        _cublas_workspace(),
        False,
        _pass == "backward",
        0,
    )
    return out


def matmul_transpose_add_gelu(
    mat: nvte.Tensor, mul: nvte.Tensor, add: nvte.Tensor, out_dtype: nvte.DType
):
    "returns mat @ mul^T + add, GELU(mat @ mul^T + add)"
    assert mat.dtype == mul.dtype
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
        _pass == "backward",
        _cublas_workspace(),
        False,
        _pass == "backward",
        0,
    )
    return pre_gelu, out


def matmul_transpose_add_add(
    mat: nvte.Tensor, mul: nvte.Tensor, add1: nvte.Tensor, add2: nvte.Tensor
):
    "returns mat @ mul^T + add1 + add2"
    assert mat.dtype == mul.dtype
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    nvte.cublas_gemm(
        a,
        b,
        add2,
        add1,
        empty(),
        trans_a,
        trans_b,
        _pass == "backward",
        _cublas_workspace(),
        True,
        _pass == "backward",
        0,
    )
    return add2


def matmul_transpose_add_gelu_add(
    mat: nvte.Tensor, mul: nvte.Tensor, add1: nvte.Tensor, add2: nvte.Tensor
):
    "returns mat @ mul^T + add1, GELU(mat @ mul^T + add1) + add2"
    assert mat.dtype == mul.dtype
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
        _pass == "backward",
        _cublas_workspace(),
        True,
        _pass == "backward",
        0,
    )
    return pre_gelu, add2


# LAYERNORM
def layernorm(
    x: nvte.Tensor,
    eps: float,
    zero_centered_gamma: bool,
    gamma: nvte.Tensor,
    beta: nvte.Tensor,
    out_dtype: nvte.DType,
):
    "returns (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta, mu (for bwd), rsigma (for bwd)"

    assert len(x.shape) == 2
    n = x.shape[0]
    mu = empty((n,), nvte.DType.Float32)
    rsigma = empty((n,), nvte.DType.Float32)
    out = empty(x.shape, out_dtype)

    if zero_centered_gamma:
        func = nvte.layernorm1p_fwd
    else:
        func = nvte.layernorm_fwd

    workspace = empty()
    barrier = empty()
    for _ in range(2):
        func(
            x,
            gamma,
            beta,
            eps,
            out,
            mu,
            rsigma,
            _sm_total_count() - _sm_margin(),
            workspace,
            barrier,
        )
        workspace = empty_like(workspace)
        barrier = empty_like(barrier)

    return out, mu, rsigma


def dlayernorm(
    grad: nvte.Tensor,
    zero_centered_gamma: bool,
    x: nvte.Tensor,
    gamma: nvte.Tensor,
    mu: nvte.Tensor,
    rsigma: nvte.Tensor,
    dx_dtype: nvte.DType,
    dgamma_dtype: nvte.DType,
    dbeta_dtype: nvte.DType,
):
    "returns dx, dgamma, dbeta"

    dx = empty(x.shape, dx_dtype)
    dgamma = empty(gamma.shape, dgamma_dtype)
    dbeta = empty(gamma.shape, dbeta_dtype)

    if zero_centered_gamma:
        func = nvte.layernorm1p_bwd
    else:
        func = nvte.layernorm_bwd

    workspace = empty()
    barrier = empty()
    dgamma_part = empty()
    dbeta_part = empty()
    for _ in range(2):
        func(
            grad,
            x,
            mu,
            rsigma,
            gamma,
            dx,
            dgamma,
            dbeta,
            dgamma_part,
            dbeta_part,
            _sm_total_count() - _sm_margin(),
            workspace,
            barrier,
        )
        workspace = empty_like(workspace)
        barrier = empty_like(barrier)
        dgamma_part = empty_like(dgamma_part)
        dbeta_part = empty_like(dbeta_part)

    return dx, dgamma, dbeta
