import torch
import cutlass

import transformer_engine
import transformer_engine_torch as tex

_torch_to_cutlass_dtype = {
    torch.uint8: cutlass.Uint8,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_str_to_cutlass_dtype = {
    "e4m3": cutlass.Float8E4M3,
    "e5m2": cutlass.Float8E5M2,
    "none": None,
}

_str_to_te_dtype = {
    "e4m3": tex.DType.kFloat8E4M3,
    "e5m2": tex.DType.kFloat8E5M2,
    "none": None,
}


def torch_to_cutlass_dtype(torch_dtype):
    if torch_dtype not in _torch_to_cutlass_dtype:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return _torch_to_cutlass_dtype[torch_dtype]


def str_to_te_dtype(str_dtype):
    if str_dtype not in _str_to_te_dtype:
        raise ValueError(f"Unsupported string dtype: {str_dtype}")
    return _str_to_te_dtype[str_dtype]
