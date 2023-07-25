from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar, overload, Callable
from .enums import DType
from ..multiple_dispatch import multiple_dispatch


TensorType = TypeVar("TensorType", bound="NativeTensor")


class GenericTensor(Protocol):
    dtype: DType


ParamInitializer = Callable[[tuple[int, ...], DType, GenericTensor], None]


@dataclass
class TensorDescriptor:
    shape: tuple[int, ...]
    constructor: ParamInitializer | None
    dtype: DType


class NativeTensor(GenericTensor, Protocol):
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


class TransformerEngineExtensionsFP8TensorMeta:
    scale: NativeTensor
    scale_inv: NativeTensor
    amax_history: NativeTensor


@dataclass
class FP8Tensor(GenericTensor):
    dtype: DType
    tensor: NativeTensor
    meta: TransformerEngineExtensionsFP8TensorMeta
    index: int


# Allocation
@multiple_dispatch
def empty(shape: tuple[int, ...], dtype: DType) -> NativeTensor:
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


# Gemm
@multiple_dispatch
def gemm(a: GenericTensor, b: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Special
@multiple_dispatch
def cast(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


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
def drelu(grad: GenericTensor, x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dgelu(grad: GenericTensor, x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dgeglu(grad: GenericTensor, x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dreglu(grad: GenericTensor, x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def dswiglu(grad: GenericTensor, x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()
