from __future__ import annotations
from typing import Callable, TypeVar, ClassVar, NamedTuple
from types import TracebackType
from .cpp_extensions import DType
import torch

T = TypeVar("T")


def _default_amax_reduction_method(
    per_tensor_amax_histories: torch.Tensor,
) -> torch.Tensor:
    return per_tensor_amax_histories.max(dim=1).values  # type: ignore


def _default_scaling_factor_compute_method(
    per_tensor_amaxes: torch.Tensor, out: torch.Tensor
):
    out.fill_(1.0)  # TODO


class Recipe(NamedTuple):
    amax_history_len: int = 1024
    amax_reduction_period: int = 10
    amax_reduction_method: Callable[
        [torch.Tensor], torch.Tensor
    ] = _default_amax_reduction_method
    scaling_factor_compute_method: Callable[
        [torch.Tensor, torch.Tensor], None
    ] = _default_scaling_factor_compute_method
    lowp: DType = DType.Float32
    world_size: int = 1

    recipe_stack: ClassVar[list[Recipe]] = []  # static

    def __enter__(self):
        Recipe.recipe_stack.append(self)

    def __exit__(self, exc_type: type[T], exc_value: T, exc_traceback: TracebackType):
        assert Recipe.recipe_stack[-1] is self
        Recipe.recipe_stack.pop()

    @staticmethod
    def current() -> Recipe:
        return Recipe.recipe_stack[-1]


Recipe.recipe_stack.append(Recipe())
