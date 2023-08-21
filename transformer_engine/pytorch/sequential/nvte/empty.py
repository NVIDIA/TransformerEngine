from typing import Sequence
import torch
from . import _nvte
from .tensor import Tensor
from .dtype import te_to_torch_dtype, is_fp8

_AMAX_HISTORY_LEN = 512


def empty(shape: Sequence[int] = (), dtype: _nvte.DType = _nvte.DType.Float32):
    if shape == ():
        return Tensor(
            dtype,
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
    if is_fp8(dtype):
        return Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            torch.zeros(_AMAX_HISTORY_LEN, dtype=torch.float32, device="cuda"),
            torch.ones(1, dtype=torch.float32, device="cuda"),
            torch.ones(1, dtype=torch.float32, device="cuda"),
        )
    else:
        return Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )


def empty_like(t: Tensor):
    return empty(t.shape, t.dtype)


def multi_empty_share_metadata(*shapes_dtypes: tuple[Sequence[int], _nvte.DType]):
    amax = torch.zeros(_AMAX_HISTORY_LEN, dtype=torch.float32, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    scale_inv = torch.ones(1, dtype=torch.float32, device="cuda")

    return tuple(
        Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            amax,
            scale,
            scale_inv,
        )
        if is_fp8(dtype)
        else Tensor(
            dtype,
            torch.empty(shape, dtype=te_to_torch_dtype(dtype), device="cuda"),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
        for shape, dtype in shapes_dtypes
    )
