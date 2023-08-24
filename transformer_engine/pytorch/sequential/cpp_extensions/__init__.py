from typing import Any
from .real import *
from . import printing  # only for side effects

# Make tensors printable

raw_type: type = globals().pop("Tensor")


class __TensorImpostor:
    def __getattribute__(self, __name: str) -> Any:
        if __name != "__repr__":
            return getattr(raw_type, __name)
        else:
            return printing.tensor_repr  # type: ignore


Tensor = __TensorImpostor()
