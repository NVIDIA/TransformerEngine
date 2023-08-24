from typing import Any
from .real import *

from . import printing

raw_tensor = globals().pop("Tensor")


class _TensorImpostor:
    __raw: object

    def __init__(self, __raw: object):
        self.__raw = __raw

    def __repr__(self) -> str:
        return printing.tensor_repr(self.__raw)  # type: ignore

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.__raw, __name)


class _TensorTypeImpostor:
    def __call__(
        self,
        dtype: Enum,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
    ):
        return _TensorImpostor(raw_tensor(dtype.value, data, amax, scale, scale_inv))  # type: ignore


Tensor = _TensorTypeImpostor()
