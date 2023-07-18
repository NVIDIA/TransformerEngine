from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar, overload
from .enums import DType


class TensorTypeBase:
    @overload
    @abstractmethod
    def view(self, size: Sequence[int], /) -> TensorTypeBase:
        ...

    @overload
    @abstractmethod
    def view(self, *size: int) -> TensorTypeBase:
        ...

    @abstractmethod
    def view(self, *size: int | Sequence[int]) -> TensorTypeBase:
        ...

    @abstractmethod
    def __getitem__(self, indices: int | slice | tuple[int | slice]) -> TensorTypeBase:
        ...

    @abstractmethod
    def is_contiguous(self) -> bool:
        ...


TensorType = TypeVar("TensorType", bound=TensorTypeBase)


class FrameworkInterface(ABC, Generic[TensorType]):
    @staticmethod
    @abstractmethod
    def fi_empty(shape: tuple[int, ...], dtype: DType) -> TensorType:
        ...

    @staticmethod
    @abstractmethod
    def fi_zeros(
        shape: tuple[int, ...] | None, dtype: DType | None, out: TensorType | None
    ) -> TensorType | None:
        ...

    @staticmethod
    @abstractmethod
    def fi_ones(
        shape: tuple[int, ...] | None, dtype: DType | None, out: TensorType | None
    ) -> TensorType | None:
        ...

    @staticmethod
    @abstractmethod
    def fi_normal(
        mean: float,
        std: float,
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: TensorType | None,
    ) -> TensorType | None:
        ...

    @staticmethod
    @abstractmethod
    def fi_uniform(
        min: float,
        max: float,
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: TensorType | None,
    ) -> TensorType | None:
        ...

    @abstractmethod
    def fi_register_buffer(self, name: str, tensor: TensorType) -> None:
        ...


def empty(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
) -> TensorType:
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
) -> TensorType | None:
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
) -> TensorType | None:
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
) -> TensorType | None:
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
) -> TensorType | None:
    return fi.fi_uniform(min, max, shape, dtype, out)
