from typing import Sequence
import torch
from . import _nvte, meta_tensor_context
from .tensor import Tensor
from .dtype import te_to_torch_dtype, is_fp8

_AMAX_HISTORY_LEN = 512


def empty(shape: Sequence[int] = (), dtype: _nvte.DType = _nvte.DType.Float32):
    return multi_empty_share_metadata((shape, dtype))[0]


def empty_like(t: Tensor):
    return empty(t.shape, t.dtype)


def multi_empty_share_metadata(*shapes_dtypes: tuple[Sequence[int], _nvte.DType]):
    if any(is_fp8(dtype) for _, dtype in shapes_dtypes):
        amax, scale, scale_inv = _create_metatensors()
    return tuple(
        Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            amax,  # type:ignore[possibly-unbound]
            scale,  # type:ignore[possibly-unbound]
            scale_inv,  # type:ignore[possibly-unbound]
        )
        if is_fp8(dtype)
        else (
            Tensor(
                dtype,
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
            )
            if shape == ()
            else Tensor(
                dtype,
                torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
            )
        )
        for shape, dtype in shapes_dtypes
    )


def _create_metatensors():
    meta_tensor_context.current().next_tensor()
    if meta_tensor_context.current().has_metatensors():
        amax, scale, scale_inv = meta_tensor_context.current().get_metatensors()
    else:
        amax = torch.zeros(_AMAX_HISTORY_LEN, dtype=torch.float32, device="cuda")
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        scale_inv = torch.ones(1, dtype=torch.float32, device="cuda")
        meta_tensor_context.current().set_metatensors((amax, scale, scale_inv))
    return amax, scale, scale_inv
