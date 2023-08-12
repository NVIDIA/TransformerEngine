from contextlib import contextmanager
from functools import cache
import os
import subprocess
from typing import Literal, Sequence
import torch
import transformer_engine_cuda as _nvte


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
        _nvte.DType.Byte, data, torch.Tensor(), torch.Tensor(), torch.Tensor()
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


def _to_cublas_args(A: _nvte.Tensor, B: _nvte.Tensor, transA: bool, transB: bool):
    return B, A, not transA, not transB


def set_current_pass(pass_: Literal["forward", "backward", "inference"]):
    global _pass
    _pass = pass_


def make_nvte_tensor(t: torch.Tensor):
    return _nvte.Tensor(
        torch_to_te_dtype(t.dtype),
        t.data,
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
    )


# DTYPES
def te_to_torch_dtype(dtype: _nvte.DType):
    match dtype:
        case _nvte.DType.Byte:
            return torch.uint8
        case _nvte.DType.Int32:
            return torch.int32
        case _nvte.DType.Int64:
            return torch.int64
        case _nvte.DType.Float32:
            return torch.float32
        case _nvte.DType.Float16:
            return torch.float16
        case _nvte.DType.BFloat16:
            return torch.bfloat16
        case _nvte.DType.Float8E4M3:
            return torch.int8
        case _nvte.DType.Float8E5M2:
            return torch.int8


def torch_to_te_dtype(dtype: torch.dtype):
    match dtype:
        case torch.int:
            return _nvte.DType.Int32
        case torch.int32:
            return _nvte.DType.Int32
        case torch.int64:
            return _nvte.DType.Int64
        case torch.float:
            return _nvte.DType.Float32
        case torch.float32:
            return _nvte.DType.Float32
        case torch.half:
            return _nvte.DType.Float16
        case torch.float16:
            return _nvte.DType.Float16
        case torch.bfloat16:
            return _nvte.DType.BFloat16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def bit_width(dtype: _nvte.DType):
    match dtype:
        case _nvte.DType.Byte:
            return 8
        case _nvte.DType.Int32:
            return 32
        case _nvte.DType.Int64:
            return 64
        case _nvte.DType.Float32:
            return 32
        case _nvte.DType.Float16:
            return 16
        case _nvte.DType.BFloat16:
            return 16
        case _nvte.DType.Float8E4M3:
            return 8
        case _nvte.DType.Float8E5M2:
            return 8


def _type_name(dtype: _nvte.DType):
    match dtype:
        case _nvte.DType.Byte:
            return "byte"
        case _nvte.DType.Int32:
            return "int32"
        case _nvte.DType.Int64:
            return "int64"
        case _nvte.DType.Float32:
            return "fp32"
        case _nvte.DType.Float16:
            return "fp16"
        case _nvte.DType.BFloat16:
            return "bf16"
        case _nvte.DType.Float8E4M3:
            return "fp8e4m3"
        case _nvte.DType.Float8E5M2:
            return "fp8e5m2"


def is_fp8(t: _nvte.Tensor | _nvte.DType):
    if isinstance(t, _nvte.Tensor):
        dtype = t.dtype
    else:
        dtype = t
    return dtype == _nvte.DType.Float8E4M3 or dtype == _nvte.DType.Float8E5M2


# ADD
def add(A: _nvte.Tensor, B: _nvte.Tensor, out_dtype: _nvte.DType):
    if is_fp8(A) or is_fp8(B):
        raise NotImplementedError()
    else:
        output = torch.empty(A.shape, dtype=te_to_torch_dtype(out_dtype), device="cuda")
        torch.add(A.data, B.data, out=output)
        return make_nvte_tensor(output)


def dbias(grad: _nvte.Tensor, out_dtype: _nvte.DType):
    if is_fp8(grad):
        raise NotImplementedError()
    else:
        output = torch.sum(grad.data, dtype=te_to_torch_dtype(out_dtype), dim=0)
        return make_nvte_tensor(output)


# CREATE
_AMAX_HISTORY_LEN = 512


def empty(shape: Sequence[int] = (), dtype: _nvte.DType = _nvte.DType.Float32):
    if shape == ():
        return _nvte.Tensor(
            dtype,
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
    if is_fp8(dtype):
        return _nvte.Tensor(
            dtype,
            torch.empty(
                _AMAX_HISTORY_LEN, dtype=te_to_torch_dtype(dtype), device="cuda"
            ),
            torch.empty(_AMAX_HISTORY_LEN, dtype=torch.float32, device="cuda"),
            torch.empty(1, dtype=torch.float32, device="cuda"),
            torch.empty(1, dtype=torch.float32, device="cuda"),
        )
    else:
        return _nvte.Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )


def empty_like(t: _nvte.Tensor):
    return empty(t.shape, t.dtype)


def multi_empty_share_metadata(*shapes_dtypes: tuple[Sequence[int], _nvte.DType]):
    amax = torch.empty(_AMAX_HISTORY_LEN, dtype=torch.float32, device="cuda")
    scale = torch.empty(1, dtype=torch.float32, device="cuda")
    scale_inv = torch.empty(1, dtype=torch.float32, device="cuda")

    return tuple(
        _nvte.Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            amax,
            scale,
            scale_inv,
        )
        if is_fp8(dtype)
        else _nvte.Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
        for shape, dtype in shapes_dtypes
    )


# CAST + TRANPOSE
def cast(t: _nvte.Tensor, dtype: _nvte.DType):
    assert t.dtype != dtype
    if is_fp8(t):
        assert not is_fp8(dtype)

    output = empty(t.shape, dtype)
    if is_fp8(dtype):
        _nvte.fp8_quantize(t, output)
    elif is_fp8(t):
        _nvte.fp8_dequantize(t, output)
    else:
        output.data.copy_(t.data)

    return output


def cast_checked(t: _nvte.Tensor, dtype: _nvte.DType | None):
    if dtype is None or t.dtype == dtype:
        return t
    else:
        return cast(t, dtype)


def transpose(t: _nvte.Tensor):
    output = empty(t.shape[::-1], t.dtype)
    _nvte.transpose(t, output)
    return output


def cast_transpose(t: _nvte.Tensor, dtype: _nvte.DType):
    assert t.dtype != dtype
    assert is_fp8(t) != is_fp8(dtype)

    out_cast, out_transpose = multi_empty_share_metadata(
        (t.shape, dtype), (t.shape[::-1], dtype)
    )

    _nvte.cast_transpose(t, out_cast, out_transpose)
    return out_cast, out_transpose


def cast_transpose_checked(t: _nvte.Tensor, dtype: _nvte.DType | None):
    if dtype is None or t.dtype == dtype:
        return t, transpose(t)
    else:
        return cast_transpose(t, dtype)


def multi_cast_transpose(*desc: tuple[_nvte.Tensor, _nvte.DType]):
    outs = [
        multi_empty_share_metadata((t.shape, dtype), (t.shape[::-1], dtype))
        for t, dtype in desc
    ]
    out_cast_list, out_transpose_list = zip(*outs)
    input_list, _ = zip(*desc)
    _nvte.multi_cast_transpose(input_list, out_cast_list, out_transpose_list)  # type: ignore
    return outs


def multi_cast_transpose_checked(*desc: tuple[_nvte.Tensor, _nvte.DType | None]):
    transpose_results = list[tuple[_nvte.Tensor, _nvte.Tensor] | None]()
    to_cast_transpose = list[tuple[_nvte.Tensor, _nvte.DType]]()
    for t, dtype in desc:
        if dtype is None or t.dtype == dtype:
            transpose_results.append((t, transpose(t)))
        else:
            to_cast_transpose.append((t, dtype))
            transpose_results.append(None)
    cast_transpose_results = multi_cast_transpose(*to_cast_transpose)
    results = list[tuple[_nvte.Tensor, _nvte.Tensor]]()
    i = 0
    for result in transpose_results:
        if result is None:
            results.append(cast_transpose_results[i])
            i += 1
        else:
            results.append(result)
    return results


def cast_transpose_dbias_checked(
    grad: _nvte.Tensor, cast_dtype: _nvte.DType | None, dbias_dtype: _nvte.DType
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
            _nvte.cast_transpose_dbias(
                grad, out_cast, out_transpose, out_dbias, workspace
            )
            workspace = empty_like(workspace)
        return out_cast, out_transpose, out_dbias
    else:
        out_cast, out_transpose = cast_transpose_checked(grad, cast_dtype)
        out_dbias = dbias(grad, dbias_dtype)
        return out_cast, out_transpose, out_dbias


# MATMUL TRANSPOSE
def matmul_transpose(mat: _nvte.Tensor, mul: _nvte.Tensor, out_dtype: _nvte.DType):
    "returns mat @ mul^T"
    # TODO: this should be allowed, though cublaslt_gemm cannot be used in this case
    assert mat.dtype == mul.dtype
    return matmul_transpose_add(mat, mul, empty(), out_dtype)


def matmul_transpose_gelu(mat: _nvte.Tensor, mul: _nvte.Tensor, out_dtype: _nvte.DType):
    "returns mat @ mul^T, GELU(mat @ mul^T)"
    assert mat.dtype == mul.dtype
    return matmul_transpose_add_gelu(mat, mul, empty(), out_dtype)


def matmul_transpose_add(
    mat: _nvte.Tensor, mul: _nvte.Tensor, add: _nvte.Tensor, out_dtype: _nvte.DType
):
    "returns mat @ mul^T + add"
    assert mat.dtype == mul.dtype
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
        _pass == "backward",
        _cublas_workspace(),
        False,
        _pass == "backward",
        0,
    )
    return out


def matmul_transpose_add_gelu(
    mat: _nvte.Tensor, mul: _nvte.Tensor, add: _nvte.Tensor, out_dtype: _nvte.DType
):
    "returns mat @ mul^T + add, GELU(mat @ mul^T + add)"
    assert mat.dtype == mul.dtype
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
        _pass == "backward",
        _cublas_workspace(),
        False,
        _pass == "backward",
        0,
    )
    return pre_gelu, out


def matmul_transpose_add_add(
    mat: _nvte.Tensor, mul: _nvte.Tensor, add1: _nvte.Tensor, add2: _nvte.Tensor
):
    "returns mat @ mul^T + add1 + add2"
    assert mat.dtype == mul.dtype
    a, b, trans_a, trans_b = _to_cublas_args(mat, mul, False, True)
    _nvte.cublas_gemm(
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
    mat: _nvte.Tensor, mul: _nvte.Tensor, add1: _nvte.Tensor, add2: _nvte.Tensor
):
    "returns mat @ mul^T + add1, GELU(mat @ mul^T + add1) + add2"
    assert mat.dtype == mul.dtype
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
        _pass == "backward",
        _cublas_workspace(),
        True,
        _pass == "backward",
        0,
    )
    return pre_gelu, add2


# LAYERNORM
class _LayerNormConfig:
    def __init__(
        self, hidden_size: int, gamma: _nvte.Tensor, x: _nvte.Tensor, out: _nvte.Tensor
    ):
        self.hidden_size = hidden_size
        self.gamma_dtype_name = _type_name(gamma.dtype)
        self.x_dtype_name = _type_name(x.dtype)
        self.out_dtype_name = _type_name(out.dtype)

    def __str__(self):
        return str(
            (
                self.hidden_size,
                self.gamma_dtype_name,
                self.x_dtype_name,
                self.out_dtype_name,
            )
        )


@contextmanager
def _handle_unsupported_layernorm_config(
    hidden_size: int, gamma: _nvte.Tensor, x: _nvte.Tensor, out: _nvte.Tensor
):
    try:
        yield
    except RuntimeError as error:
        config = _LayerNormConfig(hidden_size, gamma, x, out)
        if "in function get_fwd_launcher: FWD: Unsupported types." in str(error):
            raise ValueError(
                "This configuration for layernorm is not supported. "
                "(Regex) Search for REGISTER_FWD_(TUNED|GENERAL)_LAUNCHER to see possible options. "
                f"Used configuration: {config}"
            ) from error
        elif "in function get_bwd_launcher: BWD: Unsupported types." in str(error):
            raise ValueError(
                "This configuration for layernorm is not supported. "
                "(Regex) Search for REGISTER_BWD_(TUNED|GENERAL)_LAUNCHER to see possible options. "
                f"Used configuration: {config}"
            ) from error
        else:
            raise


def layernorm(
    x: _nvte.Tensor,
    eps: float,
    zero_centered_gamma: bool,
    gamma: _nvte.Tensor,
    beta: _nvte.Tensor,
    out_dtype: _nvte.DType,
):
    "returns (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta, mu (for bwd), rsigma (for bwd)"

    assert len(x.shape) == 2
    n, hidden_size = x.shape
    mu = empty((n,), _nvte.DType.Float32)
    rsigma = empty((n,), _nvte.DType.Float32)
    out = empty(x.shape, out_dtype)

    if zero_centered_gamma:
        func = _nvte.layernorm1p_fwd
    else:
        func = _nvte.layernorm_fwd

    with _handle_unsupported_layernorm_config(hidden_size, gamma, x, out):
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
    grad: _nvte.Tensor,
    zero_centered_gamma: bool,
    x: _nvte.Tensor,
    gamma: _nvte.Tensor,
    mu: _nvte.Tensor,
    rsigma: _nvte.Tensor,
    dx_dtype: _nvte.DType,
    dgamma_dtype: _nvte.DType,
    dbeta_dtype: _nvte.DType,
):
    "returns dx, dgamma, dbeta"

    dx = empty(x.shape, dx_dtype)
    dgamma = empty(gamma.shape, dgamma_dtype)
    dbeta = empty(gamma.shape, dbeta_dtype)

    if zero_centered_gamma:
        func = _nvte.layernorm1p_bwd
    else:
        func = _nvte.layernorm_bwd

    with _handle_unsupported_layernorm_config(x.shape[1], gamma, x, dx):
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
