from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .iteration_info import IterationAware


T = TypeVar("T")


class Persistent(Generic[T], ABC, IterationAware):
    """
    Storage for data that is to be persisted between iterations.
    Examples include fp8 metatensors (during training)
    and KV cache (during inference).
    """

    @abstractmethod
    def __call__(self) -> T:
        ...
