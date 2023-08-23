import os
import torch
from ..utils import contextmanager, cache
from .. import cpp_extensions as _nvte
from . import execution_state
from .dtype import dtype_name
from .empty import empty, empty_like


@cache
def _fwd_sm_margin():
    return int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))


@cache
def _bwd_sm_margin():
    return int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))


@cache
def _sm_total_count() -> int:
    return torch.cuda.get_device_properties(  # type: ignore
        torch.cuda.current_device()
    ).multi_processor_count


def _sm_margin():
    if execution_state.pass_ == "backward":
        return _bwd_sm_margin()
    elif execution_state.pass_ == "forward":
        return _fwd_sm_margin()
    else:
        return 0


class _NormConfig:
    def __init__(
        self, hidden_size: int, gamma: _nvte.Tensor, x: _nvte.Tensor, out: _nvte.Tensor
    ):
        self.hidden_size = hidden_size
        self.gamma_dtype_name = dtype_name(gamma.dtype)
        self.x_dtype_name = dtype_name(x.dtype)
        self.out_dtype_name = dtype_name(out.dtype)

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
def _handle_unsupported_config(
    func_name: str,
    hidden_size: int,
    gamma: _nvte.Tensor,
    x: _nvte.Tensor,
    out: _nvte.Tensor,
):
    try:
        yield
    except RuntimeError as error:
        config = _NormConfig(hidden_size, gamma, x, out)
        if "Unsupported types." in str(error):
            raise ValueError(
                f"This configuration for {func_name} is not supported. "
                "(Regex) Search for REGISTER_FWD_(TUNED|GENERAL)_LAUNCHER to see possible options. "
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

    with _handle_unsupported_config("layernorm", hidden_size, gamma, x, out):
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

    with _handle_unsupported_config("dlayernorm", x.shape[1], gamma, x, dx):
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


def rmsnorm(
    x: _nvte.Tensor,
    eps: float,
    zero_centered_gamma: bool,
    gamma: _nvte.Tensor,
    out_dtype: _nvte.DType,
):
    "returns x / sqrt(var(x) + eps) * gamma, rsigma (for bwd)"

    assert len(x.shape) == 2

    n, hidden_size = x.shape
    rsigma = empty((n,), _nvte.DType.Float32)
    out = empty(x.shape, out_dtype)

    if zero_centered_gamma:
        raise NotImplementedError()  # TODO
    else:
        func = _nvte.rmsnorm_fwd

    with _handle_unsupported_config("rmsnorm", hidden_size, gamma, x, out):
        workspace = empty()
        barrier = empty()
        for _ in range(2):
            func(
                x,
                gamma,
                eps,
                out,
                rsigma,
                _sm_total_count() - _sm_margin(),
                workspace,
                barrier,
            )
            workspace = empty_like(workspace)
            barrier = empty_like(barrier)

    return out, rsigma


def drmsnorm(
    grad: _nvte.Tensor,
    zero_centered_gamma: bool,
    x: _nvte.Tensor,
    gamma: _nvte.Tensor,
    rsigma: _nvte.Tensor,
    dx_dtype: _nvte.DType,
    dgamma_dtype: _nvte.DType,
):
    "returns dx, dgamma"

    dx = empty(x.shape, dx_dtype)
    dgamma = empty(gamma.shape, dgamma_dtype)

    if zero_centered_gamma:
        raise NotImplementedError()  # TODO
    else:
        func = _nvte.rmsnorm_bwd

    with _handle_unsupported_config("drmsnorm", x.shape[1], gamma, x, dx):
        workspace = empty()
        barrier = empty()
        dgamma_part = empty()
        for _ in range(2):
            func(
                grad,
                x,
                rsigma,
                gamma,
                dx,
                dgamma,
                dgamma_part,
                _sm_total_count() - _sm_margin(),
                workspace,
                barrier,
            )
            workspace = empty_like(workspace)
            barrier = empty_like(barrier)
            dgamma_part = empty_like(dgamma_part)

    return dx, dgamma
