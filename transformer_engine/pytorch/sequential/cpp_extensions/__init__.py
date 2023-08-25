# type: ignore
from __future__ import annotations
from .real import *

from . import printing

_TensorHandle = globals().pop("Tensor")

# Use n object pool, as torch compile
# does not like creating ScriptClass
# objects on the fly.
tensor_handles = set()


def allocate_handles():
    HANDLE_COUNT = 1024
    for _ in range(HANDLE_COUNT):
        tensor_handles.add(_TensorHandle())


# Preallocate some tensors
allocate_handles()


def make_tensor(
    dtype: DType,
    data: torch.Tensor,
    amax: torch.Tensor,
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
):
    if not tensor_handles:
        allocate_handles()
    handle = tensor_handles.pop()
    reset_tensor(handle, dtype, data, amax, scale, scale_inv)
    return handle


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
        self.__raw = make_tensor(dtype.value, data, amax, scale, scale_inv)

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

    def __del__(self):
        try:
            global tensor_handles
            tensor_handles.append(self.__raw)
        except AttributeError:
            pass
