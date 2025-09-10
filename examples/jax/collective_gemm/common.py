# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared functions for the comm_overlap tests"""

import jax.numpy as jnp
import numpy as np


# Add this after your existing imports
def dtype_tols(dtype, rtol=None, atol=None):
    """Expected numerical tolerance for a data type."""
    # Return immediately if tolerances are fully specified
    if rtol is not None and atol is not None:
        return {"rtol": rtol, "atol": atol}

    # Default tolerances for common dtypes
    if dtype in [jnp.float32, "float32"]:
        return {"rtol": 1e-5, "atol": 1e-8}
    elif dtype in [jnp.float16, "float16"]:
        return {"rtol": 1e-3, "atol": 1e-6}
    elif dtype in [jnp.bfloat16, "bfloat16"]:
        return {"rtol": 1e-2, "atol": 1e-5}
    else:
        return {"rtol": 1e-5, "atol": 1e-8}


def assert_allclose(
    actual,
    desired,
    rtol=None,
    atol=None,
    dtype=None,
    **kwargs,
):
    """Check if two tensors are close."""
    # Infer data type if needed
    if dtype is None:
        if isinstance(actual, float):
            dtype = "float32"
        else:
            dtype = actual.dtype

    # Determine tolerances
    tols = {}
    if rtol is None or atol is None:
        tols = dtype_tols(dtype)
    if rtol is not None:
        tols["rtol"] = rtol
    if atol is not None:
        tols["atol"] = atol

    # Cast tensors to fp32
    if not isinstance(actual, float):
        actual = actual.astype(jnp.float32)
    if not isinstance(desired, float):
        desired = desired.astype(jnp.float32)

    # Check if tensors are close
    np.testing.assert_allclose(actual, desired, **tols, **kwargs)


def assert_allclose_print_index(ref_output, gathered_output, rtol=1e-5, atol=1e-8):
    if not jnp.allclose(ref_output, gathered_output, rtol=rtol, atol=atol):
        diff = jnp.abs(ref_output - gathered_output)
        mask = diff > (atol + rtol * jnp.abs(gathered_output))
        print(mask.astype(int))
        print(jnp.where(mask, diff, 0))
