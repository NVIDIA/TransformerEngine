# type: ignore
from typing import Any
from .real import *

from . import printing

globals().pop("Tensor")


# Quacks like a Tensor. </joke>
# Note: cannot inherit from _Tensor as
# it is a torch.ScriptClass, and those,
# for some reason, do not support being
# inherited from.
# Also, having to use free functions
# as ScriptClass methods are not
# torch.compile friendly.
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
        self.__raw = _make_tensor(dtype.value, data, amax, scale, scale_inv)

    @property
    def dtype(self) -> DType:
        return DType(_get_tensor_dtype(self.__raw))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(_get_tensor_shape(self.__raw))

    @property
    def data(self) -> torch.Tensor:
        return _get_tensor_data(self.__raw)

    @property
    def amax(self) -> torch.Tensor:
        return _get_tensor_amax(self.__raw)

    @property
    def scale(self) -> torch.Tensor:
        return _get_tensor_scale(self.__raw)

    @property
    def scale_inv(self) -> torch.Tensor:
        return _get_tensor_scale_inv(self.__raw)

    def __repr__(self) -> str:
        return printing.tensor_repr(self.__raw)
