from abc import ABC
from functools import cache
from typing import Protocol, runtime_checkable

from .environment import PytorchDistributedGroup
from ..common_back.generic_tensor import (
    GenericTensor,
    TransformerEngineExtensionsFP8TensorMeta,
    FP8Tensor,
    NativeTensor,
)
from ..common_back.enums import DType
import torch
from dataclasses import dataclass
from ..multiple_dispatch import multiple_dispatch
from ... import cpp_extensions
import subprocess
import os
import torch.distributed as dist


@runtime_checkable
class PytorchTransformerEngineExtensionsFP8TensorMeta(
    TransformerEngineExtensionsFP8TensorMeta, Protocol
):
    scale: torch.Tensor
    scale_inv: torch.Tensor
    amax_history: torch.Tensor


@cache
def is_hopper():
    gpu_name = (
        subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        .decode("utf-8")
        .strip()
    )
    return "H100" in gpu_name


@cache
def cublas_workspace():
    workspace_size = 33_554_432 if is_hopper() else 4_194_304
    return torch.empty(workspace_size, dtype=torch.int8, device="cuda")


@cache
def fwd_ln_sm_margin():
    return int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))


@cache
def bwd_ln_sm_margin():
    return int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))


# Tesnsor types
class PytorchTensor(GenericTensor, ABC):
    dtype: DType
    tensor: torch.Tensor


@dataclass
class PytorchNativeTensor(PytorchTensor, NativeTensor):
    dtype: DType
    tensor: torch.Tensor


@dataclass
class PytorchFP8Tensor(PytorchTensor, FP8Tensor):
    dtype: DType
    tensor: torch.Tensor
    meta_ref: PytorchTransformerEngineExtensionsFP8TensorMeta
    index_in_meta: int


# Converters
@multiple_dispatch
def as_native_tensor(dtype: DType, tensor: torch.Tensor):
    return PytorchNativeTensor(dtype, tensor)


@multiple_dispatch
def as_fp8_tensor(
    dtype: DType,
    tensor: torch.Tensor,
    meta: PytorchTransformerEngineExtensionsFP8TensorMeta,
    index: int,
):
    return PytorchFP8Tensor(dtype, tensor, meta, index)


# Allocation
@multiple_dispatch
def empty(shape: tuple[int, ...], dtype: DType):
    tensor = torch.empty(shape, dtype=dtype.torch_dtype(), device="cuda")
    return PytorchNativeTensor(dtype, tensor)


# Initialization
@multiple_dispatch
def zeros(out: PytorchTensor):
    out.tensor.zero_()


@multiple_dispatch
def ones(out: PytorchTensor):
    out.tensor.fill_(1)


@multiple_dispatch
def normal_dist(mean: float, std: float, out: PytorchTensor):
    out.tensor.normal_(mean, std)


@multiple_dispatch
def uniform_dist(low: float, high: float, out: PytorchTensor):
    out.tensor.uniform_(low, high)


# Transpose
@multiple_dispatch
def transpose(x: PytorchNativeTensor, out: PytorchNativeTensor):
    out.tensor.copy_(x.tensor.mT)


# LayerNorm
@multiple_dispatch
def layer_norm(
    x: PytorchNativeTensor,
    weight: PytorchNativeTensor,
    bias: PytorchNativeTensor,
    eps: float,
    zero_centered_gamma: bool,
    out: PytorchFP8Tensor,
    out_mu: PytorchNativeTensor,
    out_rsigma: PytorchNativeTensor,
):
    cpp_extensions.layernorm_fwd_fp8(  # type: ignore
        x.tensor,
        weight.tensor,
        bias.tensor,
        eps,
        out.meta_ref,
        out.index_in_meta,
        out.dtype.tex_dtype(),
        fwd_ln_sm_margin(),
        zero_centered_gamma,
        out.tensor,
        out_mu.tensor,
        out_rsigma.tensor,
    )


@multiple_dispatch
def dlayer_norm(
    grad: PytorchNativeTensor,
    x: PytorchNativeTensor,
    weight: PytorchNativeTensor,
    zero_centered_gamma: bool,
    mu: PytorchNativeTensor,
    rsigma: PytorchNativeTensor,
    out_dgrad: PytorchNativeTensor,
    out_wgrad: PytorchNativeTensor,
    out_bgrad: PytorchNativeTensor,
) -> None:
    cpp_extensions.layernorm_bwd(
        grad.tensor,
        x.tensor,
        mu.tensor,
        rsigma.tensor,
        weight.tensor,
        bwd_ln_sm_margin(),
        zero_centered_gamma,
        out_dgrad.tensor,
        out_wgrad.tensor,
        out_bgrad.tensor,
    )


@multiple_dispatch
def layer_norm_inf(
    x: PytorchNativeTensor,
    weight: PytorchNativeTensor,
    bias: PytorchNativeTensor,
    eps: float,
    zero_centered_gamma: bool,
    out: PytorchFP8Tensor,
):
    cpp_extensions.layernorm_fwd_fp8_inf(  # type: ignore
        x.tensor,
        weight.tensor,
        bias.tensor,
        eps,
        out.meta_ref,
        out.index_in_meta,
        out.dtype.tex_dtype(),
        zero_centered_gamma,
    )


# Gemm
@multiple_dispatch
def gemm(
    a: PytorchFP8Tensor,
    b: PytorchFP8Tensor,
    out: PytorchTensor,
):
    cpp_extensions.fp8_gemm(  # type: ignore
        a.tensor,
        a.meta_ref.scale_inv,
        a.index_in_meta,
        a.dtype.tex_dtype(),
        b.tensor,
        b.meta_ref.scale_inv,
        b.index_in_meta,
        b.dtype.tex_dtype(),
        out.dtype.torch_dtype(),
        cublas_workspace(),
        out=out.tensor,
        fp8_meta_tensor=out.meta_ref if isinstance(out, PytorchFP8Tensor) else None,
        out_index=out.index_in_meta if isinstance(out, PytorchFP8Tensor) else None,
        D_dtype=out.dtype.tex_dtype(),
    )


@multiple_dispatch
def gemm(a: PytorchNativeTensor, b: PytorchNativeTensor, out: PytorchNativeTensor):
    cpp_extensions.gemm(  # type: ignore
        a.tensor,
        b.tensor,
        a.dtype.torch_dtype(),
        cublas_workspace(),
        out=out.tensor,
    )


# Cast
@multiple_dispatch
def cast(
    x: PytorchNativeTensor,
    out: PytorchNativeTensor,
):
    if x is not out:
        x.tensor.copy_(out.tensor)


@multiple_dispatch
def cast(
    x: PytorchNativeTensor,
    out: PytorchFP8Tensor,
):
    cpp_extensions.cast_to_fp8(  # type: ignore
        x.tensor,
        out.meta_ref,
        out.index_in_meta,
        out.dtype.tex_dtype(),
        out.tensor,
    )


@multiple_dispatch
def cast(
    x: PytorchFP8Tensor,
    out: PytorchNativeTensor,
):
    cpp_extensions.cast_from_fp8(  # type: ignore
        x.tensor,
        x.meta_ref,
        x.index_in_meta,
        x.dtype.tex_dtype(),
        out.dtype.tex_dtype(),
        out.tensor,
    )


# Copy
@multiple_dispatch
def copy(
    x: PytorchNativeTensor,
    out: PytorchNativeTensor,
):
    if x is not out:
        out.tensor.copy_(x.tensor)


# Pointwise
@multiple_dispatch
def add(x: PytorchNativeTensor, y: PytorchNativeTensor, out: PytorchNativeTensor):
    if x is not out and y is not out:
        torch.add(x.tensor, y.tensor, out=out.tensor)
    elif x is out and y is not out:
        x.tensor.add_(y.tensor)
    elif x is not out and y is out:
        y.tensor.add_(x.tensor)
    else:  # x is out and y is out
        x.tensor.add_(x.tensor)


# Communication


@multiple_dispatch
def gather(
    x: PytorchTensor, out: PytorchTensor, group: PytorchDistributedGroup
) -> None:
    raise NotImplementedError()


@multiple_dispatch
def scatter(
    x: PytorchTensor, out: PytorchTensor, group: PytorchDistributedGroup
) -> None:
    raise NotImplementedError()


@multiple_dispatch
def all_gather(
    x: PytorchTensor, out: PytorchTensor, group: PytorchDistributedGroup
) -> None:
    raise NotImplementedError()


@multiple_dispatch
def reduce_scatter(
    x: PytorchTensor, out: PytorchTensor, group: PytorchDistributedGroup
) -> None:
    raise NotImplementedError()


@multiple_dispatch
def all_reduce(
    x: PytorchTensor, out: PytorchTensor, group: PytorchDistributedGroup
) -> None:
    raise NotImplementedError()
