# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""CUB custom ops"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import dtypes, ffi

from .base import BasePrimitive, register_primitive

__all__ = ["topk"]


def get_cub_topk_workspace_bytes() -> int:
    """
    Get the workspace size for CUB Topk
    The safe way is calling the CUB kernel to query the workspace size.
    However, JAX JIT compiling needs a fixed tensor size. Using 4MB as
    a WAR since it is large enough for N up to 5,000,000 and K up to 100,000.
    """
    return 4 * 1024 * 1024


class TopKPrimitive(BasePrimitive):
    """
    Topk Primitive
    """

    name = "te_topk_ffi"
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
        assert in_keys_aval.ndim in (1, 2), "topk input must be 1D or 2D"

        workspace_bytes = get_cub_topk_workspace_bytes()
        if in_keys_aval.ndim == 2:
            batch_size = in_keys_aval.shape[0]
            out_shape = (batch_size, k_value)
        else:
            out_shape = (k_value,)
        out_keys_aval = jax.core.ShapedArray(shape=out_shape, dtype=keys_dtype)
        out_values_aval = jax.core.ShapedArray(shape=out_shape, dtype=jnp.int32)
        workspace_aval = jax.core.ShapedArray(shape=(workspace_bytes,), dtype=jnp.uint8)
        return (out_keys_aval, out_values_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        out_keys_aval, out_values_aval, _workspace_aval = TopKPrimitive.abstract(*args, **kwargs)
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
            TopKPrimitive.name,
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
        assert TopKPrimitive.inner_primitive is not None
        out_keys, out_values, _workspace = TopKPrimitive.inner_primitive.bind(
            in_keys,
            in_values,
            k_value=k_value,
        )
        return (out_keys, out_values)


register_primitive(TopKPrimitive)


def topk(
    x: jnp.ndarray,
    k_value: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Topk max pairs.  x may be 1D (N,) or 2D (batch, N).
    For 2D input the operation is applied independently to each row and the
    outputs have shape (batch, k).
    """
    keys = x
    # Build an index array with the same shape as x: 0..N-1 along the last axis.
    values = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape).copy()
    out_keys, out_values = TopKPrimitive.outer_primitive.bind(
        keys,
        values,
        k_value=k_value,
    )
    return out_keys, out_values
