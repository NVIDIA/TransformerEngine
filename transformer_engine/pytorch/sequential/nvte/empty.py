from typing import Sequence
import torch
from .. import cpp_extensions as _nvte
from .dtype import te_to_torch_dtype, is_fp8
from .execution_state import meta_tensor_provider


def empty(shape: Sequence[int] = (), dtype: _nvte.DType = _nvte.DType.Float32):
    return multi_empty_share_metadata((shape, dtype))[0]


def empty_like(t: _nvte.Tensor):
    return empty(t.shape, t.dtype)


def multi_empty_share_metadata(*shapes_dtypes: tuple[Sequence[int], _nvte.DType]):
    if any(is_fp8(dtype) for _, dtype in shapes_dtypes):
        amax, scale, scale_inv = meta_tensor_provider()
    return tuple(
        _nvte.Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            amax,  # type: ignore[possibly-unbound]
            scale,  # type: ignore[possibly-unbound]
            scale_inv,  # type: ignore[possibly-unbound]
        )
        if is_fp8(dtype)
        else (
            _nvte.Tensor(
                dtype,
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
            )
            if shape == ()
            else _nvte.Tensor(
                dtype,
                torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
            )
        )
        for shape, dtype in shapes_dtypes
    )
