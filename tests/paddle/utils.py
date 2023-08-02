# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utils for testing"""

import numpy as np

import paddle

import transformer_engine    # pylint: disable=unused-import

from transformer_engine.paddle.fp8 import FP8TensorMeta


def create_fp8_meta(num_gemms=1, amax_history_len=10):
    """
    Create and initialize FP8TensorMeta
    """
    fp8_meta = FP8TensorMeta(is_forward=True)
    fp8_meta.prepare(num_gemms, amax_history_len)
    return fp8_meta


def assert_allclose(actual,
                    desired,
                    rtol=1e-05,
                    atol=1e-08,
                    equal_nan=True,
                    err_msg='',
                    verbose=True):
    """Compare two input paddle tensors"""
    if isinstance(actual, paddle.Tensor):
        actual = paddle.cast(actual, 'float32').numpy()
    if isinstance(desired, paddle.Tensor):
        desired = paddle.cast(desired, 'float32').numpy()
    np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)
