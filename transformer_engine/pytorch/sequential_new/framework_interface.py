from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TensorType = TypeVar("TensorType")


class FrameworkInterface(ABC, Generic[TensorType]):
    @staticmethod
    @abstractmethod
    def fi_empty(shape: tuple[int, ...]) -> TensorType:
        ...

    @abstractmethod
    def fi_register_buffer(self, name: str, tensor: TensorType) -> None:
        ...
