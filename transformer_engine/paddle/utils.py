# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utility functions for Transformer Engine modules"""

from typing import Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from .cpp_extensions import swiglu_pd


def cast_if_needed(
    tensor: Union[paddle.Tensor, None], dtype: paddle.dtype
) -> Union[paddle.Tensor, None]:
    """Cast tensor to dtype"""
    return tensor if tensor is None or tensor.dtype == dtype else paddle.cast(tensor, dtype)


def cast_if_needed_inplace(
    tensor: Union[paddle.Tensor, None], dtype: paddle.dtype
) -> Union[paddle.Tensor, None]:
    """Cast tensor to dtype (inplace), not to be used on layer inputs"""
    return tensor if tensor is None or tensor.dtype == dtype else tensor._to(dtype=dtype)


def check_dim_for_fp8_forward_exec(tensor: paddle.Tensor) -> bool:
    """For fp8 fprop (TN layout), inputs and weights must be such
    that dim0 is divisible by 8 and dim1 is divisible by 16.
    """
    return not tensor.shape[0] % 8 and not tensor.shape[1] % 16


def assert_dim_for_fp8_forward_exec(tensor: paddle.Tensor) -> None:
    """For fp8 fprop (TN layout), inputs and weights must be such
    that dim0 is divisible by 8 and dim1 is divisible by 16.
    """
    # single tensor check so it's clear which tensor is triggering the assertion
    assert check_dim_for_fp8_forward_exec(tensor), (
        "Tensor dimensions are not compatible for FP8 execution: "
        f"({tensor.shape[0]} % 8 != 0, {tensor.shape[1]} % 16 != 0)"
    )


def get_bias_dtype(activation_dtype: paddle.dtype):
    """Get bias dtype given activation_dtype"""
    return paddle.bfloat16 if activation_dtype == paddle.float32 else activation_dtype


def get_paddle_act_func(activation):
    """Get paddle activation function"""
    funcs = {
        "gelu": F.gelu,
        "relu": F.relu,
        "silu": F.silu,
        "swiglu": swiglu_pd,
    }
    if activation not in funcs:
        raise "Activation type " + activation + " is not supported."
    return funcs[activation]


def attention_mask_func(
    attention_scores: paddle.Tensor, attention_mask: paddle.Tensor
) -> paddle.Tensor:
    """Get attention mask"""

    def _masked_fill(x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, y, x)

    attention_scores = _masked_fill(attention_scores, attention_mask, -10000.0)
    return attention_scores


def mask_to_cu_seqlens(mask: paddle.Tensor, need_kv: bool = False) -> paddle.Tensor:
    """Convert mask to cu_seqlens"""
    assert "bool" in str(mask.dtype), "mask must be bool dtype"
    assert len(mask.shape) == 4 and mask.shape[1] == 1, "mask must be [b, 1, s_q, s_kv]"
    q_actual_seqlens = paddle.sum(mask[:, :, :, 0].logical_not(), axis=(-1, -2), dtype="int32")
    q_cu_seqlens = paddle.cumsum(q_actual_seqlens)
    q_cu_seqlens = paddle.concat([paddle.zeros([1], dtype=paddle.int32), q_cu_seqlens], axis=0)
    if not need_kv:
        return q_cu_seqlens, None
    kv_actual_seqlens = paddle.sum(mask[:, :, 0, :].logical_not(), axis=(-1, -2), dtype="int32")
    kv_cu_seqlens = paddle.cumsum(kv_actual_seqlens)
    kv_cu_seqlens = paddle.concat([paddle.zeros([1], dtype=paddle.int32), kv_cu_seqlens], axis=0)
    return q_cu_seqlens, kv_cu_seqlens


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


def save_for_backward_allow_none(ctx, *args) -> None:
    """Save tensors for backward. Args could be None"""
    indices_mapping = []
    tensors_to_save = []
    for x in args:
        if isinstance(x, paddle.Tensor):
            indices_mapping.append(len(tensors_to_save))
            tensors_to_save.append(x)
        elif x is None:
            indices_mapping.append(-1)
        else:
            raise ValueError(f"Type {type(x)} is not allowed.")

    ctx._indices_mapping = indices_mapping
    ctx.save_for_backward(*tensors_to_save)


def saved_tensor_allow_none(ctx) -> Tuple[Optional[paddle.Tensor]]:
    """Used with `save_for_backward_allow_none` in pair. Get saved tensors from ctx."""
    assert hasattr(
        ctx, "_indices_mapping"
    ), "`saved_tensor_allow_none` must be used with `save_for_backward_allow_none` in pair."

    indices_mapping = ctx._indices_mapping
    outputs = []
    saved_tensors = ctx.saved_tensor()

    for index in indices_mapping:
        if index < 0:
            outputs.append(None)
        else:
            outputs.append(saved_tensors[index])

    return tuple(outputs)


def clear_tensor_data(*tensors: Tuple[Optional[paddle.Tensor], ...]) -> None:
    """
    Free tensor buffer
    """

    def can_free(t):
        return (
            t is not None
            and isinstance(t, paddle.Tensor)
            and t._is_initialized()
            and t.inplace_version == 0
        )

    for t in tensors:
        if can_free(t):
            t._clear_dataptr()
