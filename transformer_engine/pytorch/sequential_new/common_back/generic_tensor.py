from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar, overload, Callable, runtime_checkable
from .enums import DType
from ..multiple_dispatch import multiple_dispatch


TensorType = TypeVar("TensorType", bound="FrameworkTensor")


class GenericTensor(ABC):
    dtype: DType


ParamInitializer = Callable[[GenericTensor], None]


@dataclass
class TensorDescriptor:
    shape: tuple[int, ...]
    constructor: ParamInitializer | None
    dtype: DType


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
def transpose(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# LayerNorm
@multiple_dispatch(False)
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


@multiple_dispatch(False)
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


@multiple_dispatch(False)
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
@multiple_dispatch(False)
def gemm(a: GenericTensor, b: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Cast
@multiple_dispatch(False)
def cast(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Copy
@multiple_dispatch(False)
def copy(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Pointwise
@multiple_dispatch(False)
def add(x: GenericTensor, y: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


# Dropout
@multiple_dispatch(False)
def dropout(x: GenericTensor, p: float, out: GenericTensor) -> None:
    raise NotImplementedError()


# Activation
@multiple_dispatch(False)
def relu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def gelu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def geglu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def reglu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def swiglu(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def drelu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def dgelu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def dgeglu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def dreglu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch(False)
def dswiglu(grad: GenericTensor, x: GenericTensor, out_dgrad: GenericTensor) -> None:
    raise NotImplementedError()
