# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""TopK custom op"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import dtypes, ffi

from .base import BasePrimitive, register_primitive
from .misc import te_dtype_to_jax_dtype

__all__ = ["topk"]


@functools.lru_cache(maxsize=512)
def get_topk_workspace_sizes(batch_size: int, seq_len: int, k: int):
    """Query the workspace shape and dtype required for TopK.

    The result is memoised per (batch_size, seq_len, k) tuple so that repeated
    JIT compilations with the same shapes incur only one host-side CUDA call.
    """
    import transformer_engine_jax as _te_jax

    (wkspace_info,) = _te_jax.get_topk_workspace_sizes(batch_size, seq_len, k)
    return wkspace_info


class TopKPrimitive(BasePrimitive):
    """
    TopK Primitive

    Selects the top-k entries (by value) from each row of a 2-D input using the
    AIR radix-selection algorithm.  Returns both the top-k key values and their
    column indices within each row.
    """

    name = "te_topk_ffi"
    multiple_results = True
    impl_static_args = (2,)  # k_value
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        in_keys_aval,
        in_lengths_aval,
        *,
        k_value,
    ):
        keys_dtype = dtypes.canonicalize_dtype(in_keys_aval.dtype)
        assert keys_dtype in [
            jnp.float32,
            jnp.bfloat16,
        ], f"topk: unsupported key dtype {keys_dtype}; supported: float32, bfloat16"
        assert in_keys_aval.ndim == 2, "topk: keys input must be 2D (batch_size, seq_len)"
        assert dtypes.canonicalize_dtype(in_lengths_aval.dtype) == jnp.int32

        batch_size, seq_len = in_keys_aval.shape
        wkspace_info = get_topk_workspace_sizes(batch_size, seq_len, k_value)

        out_shape = (batch_size, k_value)
        out_keys_aval = jax.core.ShapedArray(shape=out_shape, dtype=keys_dtype)
        out_indices_aval = jax.core.ShapedArray(shape=out_shape, dtype=jnp.int32)
        workspace_aval = jax.core.ShapedArray(
            shape=wkspace_info[0], dtype=te_dtype_to_jax_dtype(wkspace_info[1])
        )
        return (out_keys_aval, out_indices_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        out_keys_aval, out_indices_aval, _workspace_aval = TopKPrimitive.abstract(*args, **kwargs)
        return (out_keys_aval, out_indices_aval)

    @staticmethod
    def lowering(ctx, in_keys, in_lengths, k_value):
        return ffi.ffi_lowering(TopKPrimitive.name)(
            ctx,
            in_keys,
            in_lengths,
            k_value=k_value,
        )

    @staticmethod
    def impl(in_keys, in_lengths, k_value):
        assert TopKPrimitive.inner_primitive is not None
        out_keys, out_indices, _workspace = TopKPrimitive.inner_primitive.bind(
            in_keys,
            in_lengths,
            k_value=k_value,
        )
        return (out_keys, out_indices)


register_primitive(TopKPrimitive)


def topk(
    x: jnp.ndarray,
    k_value: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Select the top-k largest entries from each row using the AIR radix algorithm.

    Args:
        x:       Input array of shape ``(batch_size, seq_len)`` or ``(seq_len,)``.
                 Supported dtypes: ``float32``, ``bfloat16``.
        k_value: Number of top entries to select per row.

    Returns:
        A tuple ``(values, indices)`` where both arrays have shape
        ``(batch_size, k_value)`` (or ``(k_value,)`` for 1-D input).  The
        outputs are *unordered*: use ``jax.lax.sort_key_val`` if a sorted result
        is required.  ``indices`` are the column positions within the original row.
    """
    squeezed = x.ndim == 1
    if squeezed:
        x = x[jnp.newaxis, :]  # (1, seq_len)

    assert x.ndim == 2, f"topk expected 2D input tensor 'x' but {x.shape=}"
    batch_size, seq_len = x.shape
    lengths = jnp.full((batch_size,), seq_len, dtype=jnp.int32)

    out_keys, out_indices = TopKPrimitive.outer_primitive.bind(
        x,
        lengths,
        k_value=k_value,
    )

    if squeezed:
        out_keys = out_keys[0]  # (k_value,)
        out_indices = out_indices[0]  # (k_value,)

    return out_keys, out_indices
