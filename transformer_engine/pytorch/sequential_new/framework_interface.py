from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

TensorType = TypeVar("TensorType")


class FrameworkInterface(ABC, Generic[TensorType]):
    @staticmethod
    @abstractmethod
    def fi_empty(shape: tuple[int, ...]) -> TensorType:
        ...

    @staticmethod
    @abstractmethod
    def fi_zeros(shape: tuple[int, ...]) -> TensorType:
        ...

    @staticmethod
    @abstractmethod
    def fi_ones(shape: tuple[int, ...]) -> TensorType:
        ...

    @staticmethod
    @abstractmethod
    def fi_normal(mean: float, std: float, shape: tuple[int, ...]) -> TensorType:
        ...

    @staticmethod
    @abstractmethod
    def fi_uniform(min: float, max: float, shape: tuple[int, ...]) -> TensorType:
        ...

    @abstractmethod
    def fi_register_buffer(self, name: str, tensor: TensorType) -> None:
        ...

    @abstractmethod
    def __getattr__(self, name: str) -> TensorType | FrameworkInterface[TensorType]:
        ...

    @abstractmethod
    def __setattr__(
        self, name: str, value: TensorType | FrameworkInterface[TensorType]
    ) -> None:
        ...

    @abstractmethod
    def __delattr__(self, name: str):
        ...


def empty(fi: FrameworkInterface[TensorType], shape: tuple[int, ...]) -> TensorType:
    return fi.fi_empty(shape)


def zeros(fi: FrameworkInterface[TensorType], shape: tuple[int, ...]) -> TensorType:
    return fi.fi_zeros(shape)


def ones(fi: FrameworkInterface[TensorType], shape: tuple[int, ...]) -> TensorType:
    return fi.fi_ones(shape)


def normal(
    mean: float, std: float, fi: FrameworkInterface[TensorType], shape: tuple[int, ...]
) -> TensorType:
    return fi.fi_normal(mean, std, shape)


def uniform(
    min: float, max: float, fi: FrameworkInterface[TensorType], shape: tuple[int, ...]
) -> TensorType:
    return fi.fi_uniform(min, max, shape)
