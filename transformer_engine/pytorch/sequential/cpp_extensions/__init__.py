# type: ignore
from __future__ import annotations
from typing import Sequence
from .real import *

from .all_fp8_values import ALL_FP8E4M3_VALUES, ALL_FP8E5M2_VALUES


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
        self.handle = create_tensor(
            dtype.value, data.shape, data, amax, scale, scale_inv
        )
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
        if self.dtype == DType.Float8E4M3 or DType.Float8E5M2:
            conv_table = (
                torch.tensor(ALL_FP8E4M3_VALUES, device="cpu")
                if self.dtype == DType.Float8E4M3
                else torch.tensor(ALL_FP8E5M2_VALUES, device="cpu")
            )
            fp32_values = conv_table[self.data.cpu().int()]
            data_repr = repr(fp32_values)
        else:
            data_repr = repr(self.data)
        data_repr = data_repr[::-1][data_repr[::-1].find("]") :][::-1]
        data_repr = "T" + data_repr[1:]
        return f"""\
{data_repr},
    dtype={self.dtype.name},\
amax={self.amax[0].item() if self.amax.numel() else None},\
scale={self.scale.item() if self.scale.numel() else None},\
scale_inv={self.scale_inv.item() if self.scale_inv.numel() else None}\
)"""

    def __del__(self):
        try:
            destroy_tensor(self.handle)
        except AttributeError:
            pass
