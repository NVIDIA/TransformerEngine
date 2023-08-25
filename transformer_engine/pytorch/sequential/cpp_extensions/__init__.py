# type: ignore
from __future__ import annotations
from typing import Sequence
from .real import *

from . import destroy_tensor, printing


# Quacks like a Tensor. </joke>
# Note: cannot inherit from _Tensor as
# it is a torch.ScriptClass, and those,
# for some reason, do not support being
# inherited from.
# Also, having to use free functions
# as ScriptClass methods are not
# torch.compile friendly.
class Tensor:
    handle: object
    data: torch.Tensor
    amax: torch.Tensor
    scale: torch.Tensor
    scale_inv: torch.Tensor

    def __init__(
        self,
        dtype: Enum,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
    ):
        self.handle = create_tensor(dtype.value, data, amax, scale, scale_inv)
        self.data = data
        self.amax = amax
        self.scale = scale
        self.scale_inv = scale_inv

    @property
    def dtype(self) -> DType:
        return get_tensor_dtype(self.handle)

    @property
    def shape(self) -> Sequence[int]:
        return get_tensor_shape(self.handle)

    def __repr__(self) -> str:
        return printing.tensor_repr(self.handle)

    def __del__(self):
        try:
            destroy_tensor(self.handle)
        except AttributeError:
            pass
