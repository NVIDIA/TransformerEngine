# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utility functions for Transformer Engine modules"""
import paddle


def get_device_compute_capability() -> float:
    """Returns the cuda compute capability of current GPU"""
    prop = paddle.device.cuda.get_device_capability()
    return prop[0] + prop[1] / 10
