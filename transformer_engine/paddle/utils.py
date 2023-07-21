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


def get_paddle_act_func(activation):
    """Get paddle activation function"""
    funcs = {
        'gelu': F.gelu,
        'relu': F.relu,
    }
    if activation not in funcs:
        raise "Activation type " + activation + " is not supported."
    return funcs[activation]
