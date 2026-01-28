# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""CUB custom ops"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import dtypes, ffi

from .base import BasePrimitive, register_primitive

__all__ = ["CubTopkPrimitive"]

def get_cub_topk_workspace_bytes() -> int:
    """
    Get the workspace size for CUB Topk
    The safe way is calling the CUB kernel to query the workspace size.
    For convenience, we use a heuristic value based on experiments.
    4 MiB is enough for N up to 5,000,000 and K up to 100,000.
    """
    return 4 * 1024 * 1024


class CubTopkPrimitive(BasePrimitive):
    """
    CUB Topk Primitive
    """

    name = "te_cub_topk_ffi"
    multiple_results = True
    impl_static_args = (2,)  # k_value
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        in_keys_aval,
        in_values_aval,
        *,
        k_value,
    ):
        keys_dtype = dtypes.canonicalize_dtype(in_keys_aval.dtype)
        values_dtype = dtypes.canonicalize_dtype(in_values_aval.dtype)
        assert keys_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert values_dtype == jnp.int32

        workspace_bytes = get_cub_topk_workspace_bytes()
        out_keys_aval = jax.core.ShapedArray(shape=(k_value,), dtype=keys_dtype)
        out_values_aval = jax.core.ShapedArray(shape=(k_value,), dtype=jnp.int32)
        workspace_aval = jax.core.ShapedArray(shape=(workspace_bytes,), dtype=jnp.uint8)
        return (out_keys_aval, out_values_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        out_keys_aval, out_values_aval, _workspace_aval = CubTopkPrimitive.abstract(*args, **kwargs)
        return (out_keys_aval, out_values_aval)

    @staticmethod
    def lowering(
        ctx,
        in_keys,
        in_values,
        k_value,
    ):
        workspace_bytes = get_cub_topk_workspace_bytes()
        return ffi.ffi_lowering(
            CubTopkPrimitive.name,
        )(
            ctx,
            in_keys,
            in_values,
            k_value=k_value,
            workbuf_bytes=workspace_bytes,
        )

    @staticmethod
    def impl(
        in_keys,
        in_values,
        k_value,
    ):
        assert CubTopkPrimitive.inner_primitive is not None
        out_keys, out_values, _workspace = CubTopkPrimitive.inner_primitive.bind(
            in_keys,
            in_values,
            k_value=k_value,
        )
        return (out_keys, out_values)


register_primitive(CubTopkPrimitive)

def cub_topk(
    x: jnp.ndarray,
    k_value: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    CUB Topk max pairs
    """
    keys = x
    values = jnp.arange(x.shape[0], dtype=jnp.int32)
    out_keys, out_values = CubTopkPrimitive.outer_primitive.bind(
        keys,
        values,
        k_value=k_value,
    )
    return out_keys, out_values
