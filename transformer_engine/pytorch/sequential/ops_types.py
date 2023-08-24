from __future__ import annotations
from typing import Callable
from typing_extensions import Unpack
from . import nvte
from .ops import Context, Grads

Forward = Callable[[nvte.Tensor], tuple[nvte.Tensor, Context]]
ForwardFused = Callable[[nvte.Tensor], tuple[nvte.Tensor, tuple[Context, ...]]]
Backward = Callable[[Context, nvte.Tensor], tuple[nvte.Tensor, Grads]]
BackwardFused = Callable[
    [Unpack[tuple[Context, ...]], nvte.Tensor], tuple[nvte.Tensor, tuple[Grads, ...]]
]
Inference = Callable[[nvte.Tensor], nvte.Tensor]

__all__ = [
    "Forward",
    "ForwardFused",
    "Backward",
    "BackwardFused",
    "Inference",
    "Context",
    "Grads",
]
