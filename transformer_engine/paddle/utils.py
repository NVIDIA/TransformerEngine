# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utility functions for Transformer Engine modules"""

from typing import Union

import paddle


def cast_if_needed(tensor: Union[paddle.Tensor, None],
                   dtype: paddle.dtype) -> Union[paddle.Tensor, None]:
    """Cast tensor to dtype"""
    return tensor if tensor is None or tensor.dtype == dtype else paddle.cast(tensor, dtype)
