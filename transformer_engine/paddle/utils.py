# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utility functions for Transformer Engine modules"""
import paddle


def cast_if_needed(tensor: paddle.Tensor, dtype: paddle.dtype) -> paddle.Tensor:
    """Cast tensor to dtype"""
    with paddle.set_grad_enabled(True):
        return tensor if tensor is None or tensor.dtype == dtype else tensor._to(dtype=dtype)
