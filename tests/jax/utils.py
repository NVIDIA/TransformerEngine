# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=c-extension-no-member

from typing import Any, Callable, Tuple, Union

from cuda import cudart

import numpy as np

import jax.numpy as jnp

from jax import lax

PRNGKey = Any
Shape = Tuple[int, ...]
DType = jnp.dtype
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision,
                                                                       lax.Precision]]
Initializer = Callable[[PRNGKey, Shape, DType], Array]


def is_fp8_supported():
    """
    Thus JAX doesn't have API to query capability
    Use cuda-python for get the compute capability
    """
    cudaSuccess = cudart.cudaError_t.cudaSuccess
    ret, gpu_id = cudart.cudaGetDevice()
    assert ret == cudaSuccess
    flag = cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor
    ret, sm_major = cudart.cudaDeviceGetAttribute(flag, gpu_id)
    assert ret == cudaSuccess
    return sm_major >= 9


def assert_allclose(actual,
                    desired,
                    rtol=1e-05,
                    atol=1e-08,
                    equal_nan=True,
                    err_msg='',
                    verbose=True):
    if not isinstance(actual, float):
        actual = actual.astype(jnp.float32)
    if not isinstance(desired, float):
        desired = desired.astype(jnp.float32)
    np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)
