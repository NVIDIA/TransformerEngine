import torch
from .enums import DType
from ..multiple_dispatch import multiple_dispatch


class GenericTensor:
    tensor: torch.Tensor
    dtype: DType


@multiple_dispatch
def dropout(x: GenericTensor, p: float, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def gemm(a: GenericTensor, b: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def cast(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def copy(x: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


@multiple_dispatch
def add(x: GenericTensor, y: GenericTensor, out: GenericTensor) -> None:
    raise NotImplementedError()


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
