from typing import Callable
import transformer_engine_cuda as _nvte  # pylint: disable=import-error
from typing_extensions import Unpack
from .ops import Context, Grads

Forward = Callable[[_nvte.Tensor], tuple[_nvte.Tensor, Context]]
ForwardFused = Callable[[_nvte.Tensor], tuple[_nvte.Tensor, tuple[Context, ...]]]
Backward = Callable[[Context, _nvte.Tensor], tuple[_nvte.Tensor, Grads]]
BackwardFused = Callable[
    [Unpack[tuple[Context, ...]], _nvte.Tensor], tuple[_nvte.Tensor, tuple[Grads, ...]]
]
Inference = Callable[[_nvte.Tensor], _nvte.Tensor]

__all__ = [
    "Forward",
    "ForwardFused",
    "Backward",
    "BackwardFused",
    "Inference",
    "Context",
    "Grads",
]
