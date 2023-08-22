from typing import Callable
import torch


class Recipe:
    amax_history_len: int
    amax_reduction_period: int
    amax_reduction_method: Callable[[torch.Tensor], torch.Tensor]
    scaling_factor_compute_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def current() -> Recipe:
    raise NotImplementedError()
