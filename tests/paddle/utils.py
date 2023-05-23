# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utils for testing"""
import paddle
import numpy as np
import transformer_engine_paddle as tex


def create_fp8_meta(num_fp8_tensors, amax_history_len):
    """
    Create and initialize FP8TensorMeta
    """
    fp8_meta = tex.FP8TensorMeta()
    fp8_meta.scale = paddle.ones(num_fp8_tensors, dtype='float32')
    fp8_meta.scale_inv = paddle.ones(num_fp8_tensors, dtype='float32')
    fp8_meta.amax_history = paddle.zeros((amax_history_len, num_fp8_tensors), dtype='float32')
    return fp8_meta


def assert_allclose(actual,
                    desired,
                    rtol=1e-05,
                    atol=1e-08,
                    equal_nan=True,
                    err_msg='',
                    verbose=True):
    """Compare two input numpy arrays"""
    if not isinstance(actual, float):
        actual = actual.astype(np.float32)
    if not isinstance(desired, float):
        desired = desired.astype(np.float32)
    np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)
