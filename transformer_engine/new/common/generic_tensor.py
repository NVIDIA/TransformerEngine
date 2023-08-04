from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar, overload, Callable, runtime_checkable

from .generic_environment import DistributedGroup
from .enums import DType
from ..multiple_dispatch import multiple_dispatch


TensorType = TypeVar("TensorType", bound="FrameworkTensor")


class GenericTensor(ABC):
    dtype: DType
    tensor: FrameworkTensor


ParamInitializer = Callable[[GenericTensor], None]


@dataclass
class TensorDescriptor:
    shape: tuple[int, ...]
    initializer: ParamInitializer | None
    dtype: DType
    is_parameter: bool


@runtime_checkable
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


class NativeTensor(GenericTensor, ABC):
    dtype: DType
    tensor: FrameworkTensor


@runtime_checkable
class TransformerEngineExtensionsFP8TensorMeta(Protocol):
    scale: FrameworkTensor
    scale_inv: FrameworkTensor
    amax_history: FrameworkTensor


class FP8Tensor(GenericTensor, ABC):
    dtype: DType
    tensor: FrameworkTensor
    meta: TransformerEngineExtensionsFP8TensorMeta
    index: int


# Converters
@multiple_dispatch(False)
def as_native_tensor(dtype: DType, tensor: FrameworkTensor) -> NativeTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def as_fp8_tensor(
    dtype: DType,
    tensor: FrameworkTensor,
    meta: TransformerEngineExtensionsFP8TensorMeta,
    index: int,
) -> FP8Tensor:
    raise NotImplementedError()


# Allocation
@multiple_dispatch(False)
def empty(shape: tuple[int, ...], dtype: DType) -> NativeTensor:
    raise NotImplementedError()


# Initialization
@multiple_dispatch(False)
def zeros(out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def ones(out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def normal_dist(mean: float, std: float, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def uniform_dist(low: float, high: float, out: GenericTensor) -> None:
    raise NotImplementedError()


# Transpose
@multiple_dispatch(False)
def transpose(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


# LayerNorm
@multiple_dispatch(False)
def layer_norm(
    x: GenericTensor,
    weight: GenericTensor,
    bias: GenericTensor,
    eps: float,
    zero_centered_gamma: bool,
) -> tuple[GenericTensor, GenericTensor, GenericTensor]:
    raise NotImplementedError()


@multiple_dispatch(False)
def dlayer_norm(
    grad: GenericTensor,
    x: GenericTensor,
    weight: GenericTensor,
    zero_centered_gamma: bool,
    mu: GenericTensor,
    rsigma: GenericTensor,
) -> tuple[GenericTensor, GenericTensor, GenericTensor]:
    raise NotImplementedError()


@multiple_dispatch(False)
def layer_norm_inf(
    x: GenericTensor,
    weight: GenericTensor,
    bias: GenericTensor,
    eps: float,
    zero_centered_gamma: bool,
) -> GenericTensor:
    raise NotImplementedError()


# Gemm
@multiple_dispatch(False)
def gemm(a: GenericTensor, b: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


# Cast
@multiple_dispatch(False)
def cast(x: GenericTensor, dtype: DType) -> GenericTensor:
    raise NotImplementedError()


# Copy
@multiple_dispatch(False)
def copy(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


# Pointwise
@multiple_dispatch(False)
def add(x: GenericTensor, y: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


# Dropout
@multiple_dispatch(False)
def dropout(x: GenericTensor, p: float) -> GenericTensor:
    raise NotImplementedError()


# Activation
@multiple_dispatch(False)
def relu(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def gelu(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def geglu(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def reglu(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def swiglu(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def drelu(grad: GenericTensor, x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def dgelu(grad: GenericTensor, x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def dgeglu(grad: GenericTensor, x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def dreglu(grad: GenericTensor, x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def dswiglu(grad: GenericTensor, x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


# Communication
@multiple_dispatch(False)
def gather(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def scatter(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def all_gather(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def reduce_scatter(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()


@multiple_dispatch(False)
def all_reduce(x: GenericTensor) -> GenericTensor:
    raise NotImplementedError()
