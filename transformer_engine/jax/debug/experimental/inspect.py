# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Experimental JAX array inspection utilities."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import ffi
from jax.sharding import NamedSharding, PartitionSpec

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

    @staticmethod
    def partition(mesh, arg_infos, result_infos):
        """
        Identity in sharding: the output carries the same sharding as ``x``;
        the four scalar stats (x_min, x_max, x_mean, x_std) are fully
        replicated. Without this override the primitive falls back to
        ``BasePrimitive``'s abstract partition and any multi-device JIT
        rejects the call.
        """
        del result_infos
        x_sharding = arg_infos[0].sharding
        scalar_sharding = NamedSharding(mesh, PartitionSpec())
        arg_shardings = (
            x_sharding,
            scalar_sharding,
            scalar_sharding,
            scalar_sharding,
            scalar_sharding,
        )
        out_sharding = x_sharding

        def sharded_impl(x, x_min, x_max, x_mean, x_std):
            return InspectPrimitive.impl(x, x_min, x_max, x_mean, x_std)

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        """
        Five operands, one output. ``x`` and the output carry the same
        wildcard rank; the four scalar stats are rank-0 (empty operand
        entries between commas).
        """
        del args
        return "..., , , , -> ..."


register_primitive(InspectPrimitive)


def _inspect_array_inner(x: jnp.ndarray) -> jnp.ndarray:
    assert InspectPrimitive.outer_primitive is not None, (
        "InspectPrimitive FFI is not registered. Please ensure the C++ extension is properly built"
        " and registered."
    )
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
    del name  # Name is currently unused, but can be included in the future for more informative output
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
