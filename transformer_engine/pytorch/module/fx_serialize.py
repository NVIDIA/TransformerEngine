# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utilities for converting runtime values into FX-evaluable expressions."""

from __future__ import annotations

import enum
import pickle
from functools import singledispatch
from typing import Dict, Tuple

import torch
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage


class _SerializeContext:
    def __init__(self) -> None:
        self.globals: Dict[str, object] = {}

    def add_global(self, name: str, value: object) -> None:
        existing = self.globals.get(name)
        if existing is not None and existing is not value:
            raise RuntimeError(f"FX serializer global name collision for '{name}'")
        self.globals[name] = value

@singledispatch
def _convert(value: object, ctx: _SerializeContext) -> str:
    try:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL).hex()
    except Exception as err:
        raise TypeError(
            f"Unsupported value for FX serialization: {type(value)!r}. "
            "Register a dedicated converter for this subtype."
        ) from err
    ctx.add_global("pickle", pickle)
    return f"pickle.loads(bytes.fromhex({payload!r}))"


@_convert.register(type(None))
def _(value: None, ctx: _SerializeContext) -> str:
    del value, ctx
    return "None"


@_convert.register(bool)
def _(value: bool, ctx: _SerializeContext) -> str:
    del ctx
    return "True" if value else "False"


@_convert.register(int)
def _(value: int, ctx: _SerializeContext) -> str:
    del ctx
    return repr(value)


@_convert.register(float)
def _(value: float, ctx: _SerializeContext) -> str:
    del ctx
    return repr(value)


@_convert.register(str)
def _(value: str, ctx: _SerializeContext) -> str:
    del ctx
    return repr(value)


@_convert.register(tuple)
def _(value: tuple, ctx: _SerializeContext) -> str:
    items = [_convert(item, ctx) for item in value]
    if len(items) == 1:
        return f"({items[0]},)"
    return f"({', '.join(items)})"


@_convert.register(list)
def _(value: list, ctx: _SerializeContext) -> str:
    items = [_convert(item, ctx) for item in value]
    return f"[{', '.join(items)}]"


@_convert.register(dict)
def _(value: dict, ctx: _SerializeContext) -> str:
    items = [f"{_convert(k, ctx)}: {_convert(v, ctx)}" for k, v in value.items()]
    return f"{{{', '.join(items)}}}"


@_convert.register(torch.dtype)
def _(value: torch.dtype, ctx: _SerializeContext) -> str:
    ctx.add_global("torch", torch)
    return str(value)


@_convert.register(torch.device)
def _(value: torch.device, ctx: _SerializeContext) -> str:
    ctx.add_global("torch", torch)
    return f"torch.device({str(value)!r})"


@_convert.register(torch.Size)
def _(value: torch.Size, ctx: _SerializeContext) -> str:
    ctx.add_global("torch", torch)
    items = [_convert(v, ctx) for v in tuple(value)]
    return f"torch.Size([{', '.join(items)}])"


@_convert.register(enum.Enum)
def _(value: enum.Enum, ctx: _SerializeContext) -> str:
    enum_cls = type(value)
    ctx.add_global(enum_cls.__name__, enum_cls)
    return f"{enum_cls.__name__}.{value.name}"


def _convert_or_none(value: object, ctx: _SerializeContext) -> str:
    try:
        return _convert(value, ctx)
    except TypeError:
        return "None"


@_convert.register(QuantizedTensorStorage)
def _(value: QuantizedTensorStorage, ctx: _SerializeContext) -> str:
    cls = type(value)
    cls_name = cls.__name__
    ctx.add_global(cls_name, cls)

    if cls_name == "Float8TensorStorage":
        return (
            f"{cls_name}("
            f"data={_convert(getattr(value, '_data'), ctx)}, "
            f"fp8_scale_inv={_convert(getattr(value, '_scale_inv'), ctx)}, "
            f"fp8_dtype={_convert(getattr(value, '_fp8_dtype'), ctx)}, "
            f"data_transpose={_convert(getattr(value, '_transpose'), ctx)}, "
            f"quantizer={_convert_or_none(getattr(value, '_quantizer'), ctx)}"
            f")"
        )

    if cls_name == "MXFP8TensorStorage":
        return (
            f"{cls_name}("
            f"rowwise_data={_convert(getattr(value, '_rowwise_data'), ctx)}, "
            f"rowwise_scale_inv={_convert(getattr(value, '_rowwise_scale_inv'), ctx)}, "
            f"columnwise_data={_convert(getattr(value, '_columnwise_data'), ctx)}, "
            f"columnwise_scale_inv={_convert(getattr(value, '_columnwise_scale_inv'), ctx)}, "
            f"fp8_dtype={_convert(getattr(value, '_fp8_dtype'), ctx)}, "
            f"quantizer={_convert_or_none(getattr(value, '_quantizer'), ctx)}, "
            f"with_gemm_swizzled_scales={_convert(getattr(value, '_with_gemm_swizzled_scales'), ctx)}"
            f")"
        )

    if cls_name == "Float8BlockwiseQTensorStorage":
        return (
            f"{cls_name}("
            f"rowwise_data={_convert(getattr(value, '_rowwise_data'), ctx)}, "
            f"rowwise_scale_inv={_convert(getattr(value, '_rowwise_scale_inv'), ctx)}, "
            f"columnwise_data={_convert(getattr(value, '_columnwise_data'), ctx)}, "
            f"columnwise_scale_inv={_convert(getattr(value, '_columnwise_scale_inv'), ctx)}, "
            f"fp8_dtype={_convert(getattr(value, '_fp8_dtype'), ctx)}, "
            f"quantizer={_convert_or_none(getattr(value, '_quantizer'), ctx)}, "
            f"is_2D_scaled={_convert(getattr(value, '_is_2D_scaled'), ctx)}"
            f")"
        )

    if cls_name == "NVFP4TensorStorage":
        return (
            f"{cls_name}("
            f"rowwise_data={_convert(getattr(value, '_rowwise_data'), ctx)}, "
            f"rowwise_scale_inv={_convert(getattr(value, '_rowwise_scale_inv'), ctx)}, "
            f"columnwise_data={_convert(getattr(value, '_columnwise_data'), ctx)}, "
            f"columnwise_scale_inv={_convert(getattr(value, '_columnwise_scale_inv'), ctx)}, "
            f"amax_rowwise={_convert(getattr(value, '_amax_rowwise'), ctx)}, "
            f"amax_columnwise={_convert(getattr(value, '_amax_columnwise'), ctx)}, "
            f"fp4_dtype={_convert(getattr(value, '_fp4_dtype'), ctx)}, "
            f"quantizer={_convert_or_none(getattr(value, '_quantizer'), ctx)}, "
            f"with_gemm_swizzled_scales={_convert(getattr(value, '_with_gemm_swizzled_scales'), ctx)}"
            f")"
        )

    # Fall back to generic object serializer for unknown storage subclasses.
    return _convert.dispatch(object)(value, ctx)


def convert_to_fx(value: object) -> Tuple[str, Dict[str, object]]:
    """Build FX expression + globals that reconstruct value from scratch."""
    ctx = _SerializeContext()
    expr = _convert(value, ctx)
    return expr, ctx.globals
