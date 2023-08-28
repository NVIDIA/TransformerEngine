from __future__ import annotations
from typing import TYPE_CHECKING, overload
import torch
from .dynamic_load import inject_real

inject_real(globals())

from .all_fp8_values import ALL_FP8E4M3_VALUES, ALL_FP8E5M2_VALUES

if TYPE_CHECKING:
    from . import *  # type: ignore


class Tensor:
    _raw: RawTensor
    dtype: DType
    shape: list[int]
    data: torch.Tensor
    amax: torch.Tensor
    scale: torch.Tensor
    scale_inv: torch.Tensor

    @overload
    def __init__(
        self,
        _raw: RawTensor,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
        /,
        *,
        dtype_override: DType | None = None,
    ) -> None:
        ...

    def __init__(
        self,
        arg0: torch.Tensor | RawTensor,
        arg1: torch.Tensor,
        arg2: torch.Tensor,
        arg3: torch.Tensor,
        arg4: torch.Tensor = torch.Tensor(),
        *,
        dtype_override: DType | None = None,
    ):
        if isinstance(arg0, RawTensor):
            self._raw = arg0
            self.dtype = self._raw.dtype
            self.shape = list(self._raw.shape)
            self.data = arg1
            self.amax = arg2
            self.scale = arg3
            self.scale_inv = arg4
            return
        data, amax, scale, scale_inv = arg0, arg1, arg2, arg3

        if dtype_override is not None:
            self.dtype = dtype_override
        else:
            self.dtype = torch_to_te_dtype(data.dtype)
        self.shape = list(data.shape)
        self._raw = RawTensor(
            data.data_ptr(),
            self.shape,
            getattr(DType, "__orig_type__")(self.dtype.value),
            amax.data_ptr(),
            scale.data_ptr(),
            scale_inv.data_ptr(),
        )
        self.data = data
        self.amax = amax
        self.scale = scale
        self.scale_inv = scale_inv

    def query_shape_dtype(self):
        self.dtype = getattr(DType, "__orig_type__")(self._raw.dtype.value)
        self.shape = list(self._raw.shape)
        return self

    def data_ptr(self):
        return self.data.data_ptr()

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
    dtype={dtype_name(self.dtype)},\
amax={self.amax[0].item() if self.amax.numel() else None},\
scale={self.scale.item() if self.scale.numel() else None},\
scale_inv={self.scale_inv.item() if self.scale_inv.numel() else None}\
)"""


def te_to_torch_dtype(dtype: DType):
    match dtype:
        case DType.Byte:
            return torch.int8
        case DType.Int32:
            return torch.int32
        case DType.Int64:
            return torch.int64
        case DType.Float32:
            return torch.float32
        case DType.Float16:
            return torch.float16
        case DType.BFloat16:
            return torch.bfloat16
        # Using different types for fp8e4m3 and fp8e5m2
        # allows for a type conversion in the other way
        case DType.Float8E4M3:
            return torch.int8
        case DType.Float8E5M2:
            return torch.uint8


def torch_to_te_dtype(dtype: torch.dtype):
    match dtype:
        case torch.int32:
            return DType.Int32
        case torch.int64:
            return DType.Int64
        case torch.float32:
            return DType.Float32
        case torch.float16:
            return DType.Float16
        case torch.bfloat16:
            return DType.BFloat16
        case torch.int8:
            # We assume that this is not a workspace (Byte)
            # tensor, as these shouldn't be exposed outside
            # of basic operations.
            return DType.Float8E4M3
        case torch.uint8:
            return DType.Float8E5M2
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def bit_width(dtype: DType):
    match dtype:
        case DType.Byte:
            return 8
        case DType.Int32:
            return 32
        case DType.Int64:
            return 64
        case DType.Float32:
            return 32
        case DType.Float16:
            return 16
        case DType.BFloat16:
            return 16
        case DType.Float8E4M3:
            return 8
        case DType.Float8E5M2:
            return 8


def dtype_name(dtype: DType):
    match dtype:
        case DType.Byte:
            return "byte"
        case DType.Int32:
            return "int32"
        case DType.Int64:
            return "int64"
        case DType.Float32:
            return "fp32"
        case DType.Float16:
            return "fp16"
        case DType.BFloat16:
            return "bf16"
        case DType.Float8E4M3:
            return "fp8e4m3"
        case DType.Float8E5M2:
            return "fp8e5m2"
