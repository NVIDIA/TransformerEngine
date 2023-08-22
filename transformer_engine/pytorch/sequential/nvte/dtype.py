import torch
from .. import cpp_extensions as _nvte


def te_to_torch_dtype(dtype: _nvte.DType):
    match dtype:
        case _nvte.DType.Byte:
            return torch.uint8
        case _nvte.DType.Int32:
            return torch.int32
        case _nvte.DType.Int64:
            return torch.int64
        case _nvte.DType.Float32:
            return torch.float32
        case _nvte.DType.Float16:
            return torch.float16
        case _nvte.DType.BFloat16:
            return torch.bfloat16
        case _nvte.DType.Float8E4M3:
            return torch.int8
        case _nvte.DType.Float8E5M2:
            return torch.int8


def torch_to_te_dtype(dtype: torch.dtype):
    match dtype:
        case torch.int:
            return _nvte.DType.Int32
        case torch.int32:
            return _nvte.DType.Int32
        case torch.int64:
            return _nvte.DType.Int64
        case torch.float:
            return _nvte.DType.Float32
        case torch.float32:
            return _nvte.DType.Float32
        case torch.half:
            return _nvte.DType.Float16
        case torch.float16:
            return _nvte.DType.Float16
        case torch.bfloat16:
            return _nvte.DType.BFloat16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def bit_width(dtype: _nvte.DType):
    match dtype:
        case _nvte.DType.Byte:
            return 8
        case _nvte.DType.Int32:
            return 32
        case _nvte.DType.Int64:
            return 64
        case _nvte.DType.Float32:
            return 32
        case _nvte.DType.Float16:
            return 16
        case _nvte.DType.BFloat16:
            return 16
        case _nvte.DType.Float8E4M3:
            return 8
        case _nvte.DType.Float8E5M2:
            return 8


def dtype_name(dtype: _nvte.DType):
    match dtype:
        case _nvte.DType.Byte:
            return "byte"
        case _nvte.DType.Int32:
            return "int32"
        case _nvte.DType.Int64:
            return "int64"
        case _nvte.DType.Float32:
            return "fp32"
        case _nvte.DType.Float16:
            return "fp16"
        case _nvte.DType.BFloat16:
            return "bf16"
        case _nvte.DType.Float8E4M3:
            return "fp8e4m3"
        case _nvte.DType.Float8E5M2:
            return "fp8e5m2"


def is_fp8(t: _nvte.Tensor | _nvte.DType):
    if isinstance(t, _nvte.Tensor):
        dtype = t.dtype
    else:
        dtype = t
    return dtype == _nvte.DType.Float8E4M3 or dtype == _nvte.DType.Float8E5M2
