from __future__ import annotations
from typing import Callable, TypeVar
from types import TracebackType
from dataclasses import dataclass
from .cpp_extensions import DType
import torch

T = TypeVar("T")


@dataclass
class Recipe:
    amax_history_len: int
    amax_reduction_period: int
    amax_reduction_method: Callable[[torch.Tensor], torch.Tensor]
    scaling_factor_compute_method: Callable[[torch.Tensor, torch.Tensor], None]
    lowp: DType
    world_size: int

    def __enter__(self):
        __recipe_stack.append(self)

    def __exit__(self, exc_type: type[T], exc_value: T, exc_traceback: TracebackType):
        assert __recipe_stack[-1] is self
        __recipe_stack.pop()

    @staticmethod
    def current() -> Recipe:
        return __recipe_stack[-1]


def _default_amax_reduction_method(
    per_tensor_amax_histories: torch.Tensor,
) -> torch.Tensor:
    return per_tensor_amax_histories.max(dim=1).values  # type: ignore


def _default_scaling_factor_compute_method(
    per_tensor_amaxes: torch.Tensor, out: torch.Tensor
):
    out.fill_(1.0)  # TODO


__recipe_stack = [
    Recipe(
        1024,
        10,
        _default_amax_reduction_method,
        _default_scaling_factor_compute_method,
        DType.Float32,
        1,
    )
]
