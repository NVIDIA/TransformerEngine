from __future__ import annotations
from typing import Protocol, Sequence, TypeVar, overload
from .enums import DType


class TensorTypeBase(Protocol):
    @overload
    def view(self, size: Sequence[int], /) -> TensorTypeBase:
        ...

    @overload
    def view(self, *size: int) -> TensorTypeBase:
        ...

    def view(self, *size: int | Sequence[int]) -> TensorTypeBase:
        raise NotImplementedError()

    def __getitem__(self, indices: int | slice | tuple[int | slice]) -> TensorTypeBase:
        raise NotImplementedError()

    def is_contiguous(self) -> bool:
        raise NotImplementedError()


TensorType = TypeVar(
    "TensorType",
    bound=TensorTypeBase,
)


class FrameworkInterface(Protocol[TensorType]):
    Tensor: type[TensorType]

    @staticmethod
    def fi_empty(shape: tuple[int, ...], dtype: DType) -> "Tensor":
        raise NotImplementedError()

    @staticmethod
    def fi_zeros(
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: "Tensor" | None,
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_ones(
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: "Tensor" | None,
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_normal(
        mean: float,
        std: float,
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: "Tensor" | None,
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_uniform(
        min: float,
        max: float,
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: "Tensor" | None,
    ) -> "Tensor" | None:
        raise NotImplementedError()

    def fi_register_buffer(self, name: str, tensor: "Tensor") -> None:
        raise NotImplementedError()


def empty(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
):
    return fi.fi_empty(shape, dtype)


@overload
def zeros(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def zeros(
    fi: type[FrameworkInterface[TensorType]],
    /,
    *,
    out: TensorType,
) -> None:
    ...


def zeros(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    /,
    *,
    out: TensorType | None = None,
):
    return fi.fi_zeros(shape, dtype, out)


@overload
def ones(
    fi: type[FrameworkInterface[TensorType]], shape: tuple[int, ...], dtype: DType, /
) -> TensorType:
    ...


@overload
def ones(
    fi: type[FrameworkInterface[TensorType]],
    /,
    *,
    out: TensorType,
) -> None:
    ...


def ones(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    out: TensorType | None = None,
):
    return fi.fi_ones(shape, dtype, out)


@overload
def normal(
    mean: float,
    std: float,
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def normal(
    mean: float,
    std: float,
    fi: type[FrameworkInterface[TensorType]],
    /,
    *,
    out: TensorType,
) -> None:
    ...


def normal(
    mean: float,
    std: float,
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    out: TensorType | None = None,
):
    return fi.fi_normal(mean, std, shape, dtype, out)


@overload
def uniform(
    min: float,
    max: float,
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def uniform(
    min: float,
    max: float,
    fi: type[FrameworkInterface[TensorType]],
    /,
    *,
    out: TensorType,
) -> None:
    ...


def uniform(
    min: float,
    max: float,
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    out: TensorType | None = None,
):
    return fi.fi_uniform(min, max, shape, dtype, out)


class ParamConstructor(Protocol):
    @staticmethod
    def __call__(
        fi: type[FrameworkInterface[TensorType]],
        shape: tuple[int, ...],
        dtype: DType,
        /,
    ) -> TensorType:
        raise NotImplementedError()
