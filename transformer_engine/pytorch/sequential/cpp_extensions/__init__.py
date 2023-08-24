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


Tensor = __TensorImpostor()
