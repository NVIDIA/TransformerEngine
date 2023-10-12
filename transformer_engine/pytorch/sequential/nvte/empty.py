from __future__ import annotations
from typing import Sequence
import torch
from . import cpp_extensions as _nvte
from .dtype import te_to_torch_dtype, is_fp8
from . import execution_state


def empty(shape: Sequence[int] = (), out_dtype: _nvte.DType = _nvte.DType.Float32):
    return multi_empty_share_metadata((shape, out_dtype))[0]


def empty_like(t: _nvte.Tensor):
    return empty(t.shape, t.dtype)


def multi_empty_share_metadata(*shapes_dtypes: tuple[Sequence[int], _nvte.DType]):
    if any(is_fp8(dtype) for _, dtype in shapes_dtypes):
        if len({dtype for _, dtype in shapes_dtypes if is_fp8(dtype)}) != 1:
            raise ValueError(
                "All FP8 tensors that share the same metatensors must have the same dtype."
            )
        fp8_dtype = next(dtype for _, dtype in shapes_dtypes if is_fp8(dtype))
        amax, scale, scale_inv = execution_state.meta_tensor_provider(fp8_dtype)
    return tuple(
        _nvte.Tensor(
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda")
            if shape != ()
            else torch.Tensor().cuda(),
            amax if is_fp8(dtype) else torch.Tensor().cuda(),  # type: ignore[possibly-unbound]
            scale if is_fp8(dtype) else torch.Tensor().cuda(),  # type: ignore[possibly-unbound]
            scale_inv if is_fp8(dtype) else torch.Tensor().cuda(),  # type: ignore[possibly-unbound]
        )
        for shape, dtype in shapes_dtypes
    )
