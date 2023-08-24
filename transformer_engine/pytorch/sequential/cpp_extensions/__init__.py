from .real import *
from . import printing  # only for side effects

# Make tensors printable

raw_type: type = globals().pop("Tensor")
class Tensor(raw_type): # type: ignore
    def __repr__(self):
        return printing.tensor_repr(self) # type: ignore
