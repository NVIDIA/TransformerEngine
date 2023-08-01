# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utility functions for Transformer Engine modules"""

from typing import Union

import paddle
import paddle.nn.functional as F


def cast_if_needed(tensor: Union[paddle.Tensor, None],
                   dtype: paddle.dtype) -> Union[paddle.Tensor, None]:
    """Cast tensor to dtype"""
    return tensor if tensor is None or tensor.dtype == dtype else paddle.cast(tensor, dtype)


def cast_if_needed_inplace(tensor: Union[paddle.Tensor, None],
                           dtype: paddle.dtype) -> Union[paddle.Tensor, None]:
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
        f"({tensor.shape[0]} % 8 != 0, {tensor.shape[1]} % 16 != 0)")


def get_bias_dtype(activation_dtype: paddle.dtype):
    """Get bias dtype given activation_dtype"""
    return paddle.bfloat16 if activation_dtype == paddle.float32 else activation_dtype


def get_paddle_act_func(activation):
    """Get paddle activation function"""
    funcs = {
        'gelu': F.gelu,
        'relu': F.relu,
    }
    if activation not in funcs:
        raise "Activation type " + activation + " is not supported."
    return funcs[activation]
