from typing import Any
from .real import *

from . import printing

raw_tensor = globals().pop("Tensor")


class __TensorImpostor:
    def __getattribute__(self, __name: str) -> Any:
        if __name == "__repr__":
            return printing.tensor_repr  # type: ignore
        else:
            return getattr(raw_tensor, __name)

    def __call__(
        self,
        dtype: Any,
        data: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
    ):
        return raw_tensor(self, dtype.value, data, scale, scale_inv)  # type: ignore


Tensor = __TensorImpostor()
