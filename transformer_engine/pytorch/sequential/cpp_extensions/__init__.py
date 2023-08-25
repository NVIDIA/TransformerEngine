# type: ignore
from typing import Any
from .real import *

from . import printing

_Tensor = globals().pop("Tensor")


# Quacks like a Tensor. </joke>
class Tensor:
    __raw: object

    def __init__(
        self,
        dtype: Enum,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
    ):
        self.__raw = _Tensor(dtype.value, data, amax, scale, scale_inv)

    def __repr__(self) -> str:
        return printing.tensor_repr(self.__raw)

    # Note: cannot inherit from _Tensor as
    # it is a torch.ScriptClass, and those,
    # for some reason, do not support being
    # inherited from. Using __getattr__ to
    # work around this limitation.
    def __getattr__(self, __name: str) -> Any:
        return getattr(self.__raw, __name)

    @property
    def dtype(self):
        return DType(self.__raw.dtype)
