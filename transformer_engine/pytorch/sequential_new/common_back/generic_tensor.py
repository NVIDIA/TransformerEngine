from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar, overload, Callable
from .enums import DType
from ..multiple_dispatch import multiple_dispatch


TensorType = TypeVar("TensorType", bound="FrameworkTensor")


class GenericTensor(Protocol):
    dtype: DType


ParamInitializer = Callable[[GenericTensor], None]


@dataclass
class TensorDescriptor:
    shape: tuple[int, ...]
    constructor: ParamInitializer | None
    dtype: DType


class FrameworkTensor(Protocol):
    @overload
    def view(self: TensorType, size: Sequence[int], /) -> TensorType:
        ...

    @overload
    def view(self: TensorType, *size: int) -> TensorType:
        ...

    def view(self: TensorType, *size: int | Sequence[int]) -> TensorType:
        raise NotImplementedError()

    def __getitem__(
        self: TensorType, indices: int | slice | tuple[int | slice]
    ) -> TensorType:
        raise NotImplementedError()

    def is_contiguous(self) -> bool:
        raise NotImplementedError()


@dataclass
class NativeTensor(GenericTensor):
    dtype: DType
    tensor: FrameworkTensor


class TransformerEngineExtensionsFP8TensorMeta:
    scale: FrameworkTensor
    scale_inv: FrameworkTensor
    amax_history: FrameworkTensor


@dataclass
class FP8Tensor(GenericTensor):
    dtype: DType
    tensor: FrameworkTensor
    meta: TransformerEngineExtensionsFP8TensorMeta
    index: int


# Allocation
@multiple_dispatch
def empty(shape: tuple[int, ...], dtype: DType) -> FrameworkTensor:
    raise NotImplementedError()


# Initialization
@multiple_dispatch
def zeros(out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def ones(out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def normal_dist(mean: float, std: float, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def uniform_dist(low: float, high: float, out: GenericTensor) -> None:
    raise NotImplementedError()


# Transpose
@multiple_dispatch
def transpose(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# LayerNorm
@multiple_dispatch
def layer_norm(
    x: GenericTensor,
    weight: GenericTensor,
    bias: GenericTensor,
    eps: float,
    zero_centered_gamma: bool,
    out_act: GenericTensor,
    out_mu: GenericTensor,
    out_rsigma: GenericTensor,
) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dlayer_norm(
    grad: GenericTensor,
    x: GenericTensor,
    weight: GenericTensor,
    zero_centered_gamma: bool,
    mu: GenericTensor,
    rsigma: GenericTensor,
    out_dgrad: GenericTensor,
    out_wgrad: GenericTensor,
    out_bgrad: GenericTensor,
) -> None:
    raise NotImplementedError()


@multiple_dispatch
def layer_norm_inf(
    x: GenericTensor,
    weight: GenericTensor,
    bias: GenericTensor,
    eps: float,
    zero_centered_gamma: bool,
    out_act: GenericTensor,
) -> None:
    raise NotImplementedError()


# Gemm
@multiple_dispatch
def gemm(a: GenericTensor, b: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Cast
@multiple_dispatch
def cast(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Copy
@multiple_dispatch
def copy(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Pointwise
@multiple_dispatch
def add(x: GenericTensor, y: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Dropout
@multiple_dispatch
def dropout(x: GenericTensor, p: float, out: GenericTensor) -> None:
    raise NotImplementedError()


# Activation
@multiple_dispatch
def relu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def gelu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def geglu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def reglu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def swiglu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def drelu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dgelu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dgeglu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dreglu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dswiglu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()
