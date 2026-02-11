# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Experimental JAX array inspection utilities."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import ffi

from transformer_engine.jax.cpp_extensions.base import BasePrimitive, register_primitive

__all__ = ["inspect_array", "load_array_dump"]


class InspectPrimitive(BasePrimitive):
    """
    No-op used for inspect array values.
    """

    name = "te_inspect_ffi"
    multiple_results = False
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        x_min_aval,
        x_max_aval,
        x_mean_aval,
        x_std_aval,
    ):
        """
        inspect abstract
        """
        assert (
            x_min_aval.shape == () and x_min_aval.dtype == jnp.float32
        ), "x_min must be a scalar with dtype float32"
        assert (
            x_max_aval.shape == () and x_max_aval.dtype == jnp.float32
        ), "x_max must be a scalar with dtype float32"
        assert (
            x_mean_aval.shape == () and x_mean_aval.dtype == jnp.float32
        ), "x_mean must be a scalar with dtype float32"
        assert (
            x_std_aval.shape == () and x_std_aval.dtype == jnp.float32
        ), "x_std must be a scalar with dtype float32"
        return x_aval

    @staticmethod
    def lowering(
        ctx,
        x,
        x_min,
        x_max,
        x_mean,
        x_std,
    ):
        """
        inspect lowering rules
        """

        return ffi.ffi_lowering(
            InspectPrimitive.name,
            operand_output_aliases={0: 0},  # donate input buffer to output buffer
        )(
            ctx,
            x,
            x_min,
            x_max,
            x_mean,
            x_std,
        )

    @staticmethod
    def impl(
        x,
        x_min,
        x_max,
        x_mean,
        x_std,
    ):
        """
        inspect implementation
        """
        assert InspectPrimitive.inner_primitive is not None
        (x) = InspectPrimitive.inner_primitive.bind(
            x,
            x_min,
            x_max,
            x_mean,
            x_std,
        )
        return x


register_primitive(InspectPrimitive)


def _inspect_array_inner(x: jnp.ndarray) -> jnp.ndarray:
    return InspectPrimitive.outer_primitive.bind(
        x,
        jnp.min(x).astype(jnp.float32),
        jnp.max(x).astype(jnp.float32),
        jnp.mean(x.astype(jnp.float32)),
        jnp.std(x.astype(jnp.float32)),
    )


@partial(jax.custom_vjp, nondiff_argnums=())
def _inspect(
    x,
):
    """ """
    output, _ = _inspect_fwd_rule(
        x,
    )
    return output


def _inspect_fwd_rule(
    x,
):
    """"""
    ctx = ()
    x = _inspect_array_inner(x)
    return x, ctx


def _inspect_bwd_rule(
    ctx,
    grad,
):
    """"""
    del ctx
    return (grad,)


_inspect.defvjp(_inspect_fwd_rule, _inspect_bwd_rule)


def inspect_array(x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Utility function to inspect JAX arrays by printing their name, shape, dtype, and statistics.

    Args:
        x (jnp.ndarray): The JAX array to inspect.
        name (str): The name of the array for identification in the output.
    """
    # TODO: Handle the name of the tensor in the primitive and output files
    return _inspect(x)


def compare(a: jnp.ndarray, b: jnp.ndarray, name: str) -> jnp.ndarray:
    """Utility function to compare two JAX arrays and print their differences.

    Args:
        a (jnp.ndarray): The first JAX array to compare.
        b (jnp.ndarray): The second JAX array to compare.
        name (str): The name of the comparison for identification in the output.

    Returns:
        jnp.ndarray: The first input array `a`, returned unchanged.
    """
    # a, b = b, a

    diff = a - b
    jax.debug.print(
        "Comparing arrays {name}: min={min}, max={max}, mean={mean}, std={std}",
        name=name,
        min=jnp.min(diff),
        max=jnp.max(diff),
        mean=jnp.mean(diff),
        std=jnp.std(diff),
    )

    return a

    out_f32 = inspect_array(a.astype(jnp.float32) - b.astype(jnp.float32), name) + b.astype(
        jnp.float32
    )
    return out_f32.astype(a.dtype)


def load_array_dump(filename: str, shape: tuple, dtype: jnp.dtype) -> jnp.ndarray:
    """Utility function to load a JAX array from a dumped binary file.

    Args:
        filename (str): The path to the binary file containing the array data.
        shape (tuple): The shape of the array to be loaded.
        dtype (jnp.dtype): The data type of the array to be loaded.

    Returns:
        jnp.ndarray: The loaded JAX array.
    """
    with open(filename, "rb") as f:
        data = f.read()
    array = jnp.frombuffer(data, dtype=dtype).reshape(shape)
    return array
