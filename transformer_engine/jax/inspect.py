# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX array inspection utilities."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import ffi

from .cpp_extensions.base import BasePrimitive, register_primitive

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
    ):
        """
        inspect abstract
        """
        return x_aval

    @staticmethod
    def lowering(
        ctx,
        x,
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
        )

    @staticmethod
    def impl(
        x,
    ):
        """
        inspect implementation
        """
        assert InspectPrimitive.inner_primitive is not None
        (x) = InspectPrimitive.inner_primitive.bind(
            x,
        )
        return x


register_primitive(InspectPrimitive)


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
    x = InspectPrimitive.outer_primitive.bind(x)
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
