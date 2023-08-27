from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from .dynamic_load import inject_real

inject_real(globals())

from .all_fp8_values import ALL_FP8E4M3_VALUES, ALL_FP8E5M2_VALUES

if TYPE_CHECKING:
    from . import *  # type: ignore


class Tensor:
    __raw: RawTensor
    data: torch.Tensor
    amax: torch.Tensor
    scale: torch.Tensor
    scale_inv: torch.Tensor

    def __init__(
        self,
        dtype: DType,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
    ):
        self.__raw = RawTensor(
            data.data_ptr(),
            data.shape,
            dtype,
            amax.data_ptr(),
            scale.data_ptr(),
            scale_inv.data_ptr(),
        )
        self.data = data
        self.amax = amax
        self.scale = scale
        self.scale_inv = scale_inv

    def data_ptr(self):
        return self.data.data_ptr()

    @property
    def dtype(self):
        return self.__raw.dtype

    @property
    def shape(self):
        return self.__raw.shape

    def __repr__(self):
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
