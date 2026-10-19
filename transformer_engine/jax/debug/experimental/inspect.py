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
    # ``name`` is positional (index 5 in ``impl``) so ``custom_partitioning``
    # can resolve ``bind(..., name=...)`` kwargs back to that position.
    impl_static_args = (5,)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        x_aval,
        x_min_aval,
        x_max_aval,
        x_mean_aval,
        x_std_aval,
        *,
        name,
    ):
        """
        inspect abstract
        """
        del name
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
        *,
        name,
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
            name=name,
        )

    @staticmethod
    def impl(
        x,
        x_min,
        x_max,
        x_mean,
        x_std,
        name,
    ):
        """inspect implementation"""
        assert InspectPrimitive.inner_primitive is not None
        x = InspectPrimitive.inner_primitive.bind(
            x,
            x_min,
            x_max,
            x_mean,
            x_std,
            name=name,
        )
        return x

    @staticmethod
    def partition(name, mesh, arg_infos, result_infos):
        """Identity sharding: output matches ``x``; scalar stats are replicated."""
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
            return InspectPrimitive.impl(x, x_min, x_max, x_mean, x_std, name)

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        """``x`` and output share rank; the four scalar stats are rank-0."""
        del args
        return "..., , , , -> ..."


register_primitive(InspectPrimitive)


def _inspect_array_inner(x: jnp.ndarray, name: str) -> jnp.ndarray:
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
        name=name,
    )


# ``name`` is carried as a custom_vjp nondiff argument so it stays static
# at compile time and lands on the FFI as a string attribute.
@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _inspect(x, name):
    """ """
    output, _ = _inspect_fwd_rule(x, name)
    return output


def _inspect_fwd_rule(x, name):
    """"""
    ctx = ()
    x = _inspect_array_inner(x, name)
    return x, ctx


def _inspect_bwd_rule(name, ctx, grad):
    """"""
    del name, ctx
    return (grad,)


_inspect.defvjp(_inspect_fwd_rule, _inspect_bwd_rule)


def inspect_array(x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Inspect a JAX array by dumping its data and stats to disk per-rank.

    Each call writes two files per rank, keyed by ``name`` so multiple
    probes in the same program produce distinct dumps:

    * ``my_tensor_gpu{device}_{sanitized_name}.bin`` -- raw bytes.
    * ``my_tensor_gpu{device}_{sanitized_name}_meta.json`` -- ``name``,
      shape, dtype, and min/max/mean/std summary stats.

    A summary line is also printed to stdout, including ``name``.

    ``name`` is a static (non-traced) attribute. Characters outside
    ``[A-Za-z0-9._-]`` are mapped to ``_`` in the filename, but the
    original name is preserved in the JSON metadata and log line.

    Args:
        x (jnp.ndarray): The JAX array to inspect.
        name (str): Identifier for this probe; used in filenames and logs.
    """
    return _inspect(x, name)


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
