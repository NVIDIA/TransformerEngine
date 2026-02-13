# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Experimental JAX array inspection utilities."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import ffi

from transformer_engine.jax.cpp_extensions.base import BasePrimitive, register_primitive

__all__ = ["compare", "compare_vjp", "inspect_array", "load_array_dump"]


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


def _tensor_to_image(tensor, value_range=None):
    import numpy as np
    from PIL import Image

    # Convert to numpy
    tensor_np = jnp.array(tensor, dtype=jnp.float32)

    # Replace NaNs with a large value for visualization
    tensor_np = jnp.where(jnp.isnan(tensor_np), 5000, tensor_np)

    # Determine normalization range
    if value_range is None:
        min_val = tensor_np.min()
        max_val = tensor_np.max()
    else:
        min_val, max_val = value_range

    # Normalize to 0-255 range for visualization
    range_val = max_val - min_val + 1e-8
    normalized = jnp.clip((tensor_np - min_val) / range_val * 255, 0, 255)

    # Downsample by averaging 4x4 blocks
    h, w = normalized.shape
    new_h, new_w = h // 4, w // 4
    normalized = normalized[: new_h * 4, : new_w * 4]  # Trim to multiple of 4
    normalized = normalized.reshape(new_h, 4, new_w, 4).mean(axis=(1, 3))
    normalized = np.array(normalized)
    normalized_uint8 = normalized.astype(np.uint8)

    # Create grayscale image
    img = Image.fromarray(normalized_uint8, mode="L")
    return img


_count = 0


def _tensor_diff_to_image(out, ref):
    import os
    import math

    os.makedirs("debug_outputs", exist_ok=True)

    global _count

    if _count > 50:
        return

    out = out.reshape((math.prod(out.shape[:-1]), out.shape[-1])).astype(jnp.float32)
    ref = ref.reshape((math.prod(ref.shape[:-1]), ref.shape[-1])).astype(jnp.float32)

    _tensor_to_image(out, value_range=(jnp.min(ref), jnp.max(ref))).save(
        f"debug_outputs/output_te_{_count}.png"
    )
    _tensor_to_image(ref, value_range=(jnp.min(ref), jnp.max(ref))).save(
        f"debug_outputs/output_ref_{_count}.png"
    )
    diff = jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))
    _tensor_to_image(
        diff,
        value_range=(jnp.min(diff), jnp.max(diff)),
        # value_range=(jnp.min(ref), jnp.max(ref)),
        # value_range=(0, 0.5)
    ).save(f"debug_outputs/output_diff_{_count}.png")

    _count += 1


def compare_vjp(f1: callable, f2: callable, name: str) -> callable:
    """Utility function to compare the outputs of two functions and in the forward and backward passes.

    Handles non-differentiable arguments (e.g., integer arrays) gracefully by
    detecting float0 gradients and passing them through without comparison.

    Args:
        f1 (callable): The first function to compare.
        f2 (callable): The second function to compare.
        name (str): The name of the comparison for identification in the output.

    Returns:
        callable: A new function that compares the outputs of `f1` and `f2` when called and returns the result of `f1`.
    """

    @jax.custom_vjp
    def _f(*args):
        return _f_fwd_rule(*args)[0]

    def _f_fwd_rule(*args):
        out1, f1_vjp_func = jax.vjp(f1, *args)
        out2, f2_vjp_func = jax.vjp(f2, *args)
        out = compare(out1, out2, name + "_fwd")
        return out, (f1_vjp_func, f2_vjp_func, args[2])

    def _has_float0(x):
        """Check if a pytree leaf or structure contains float0 dtypes."""
        leaves = jax.tree_util.tree_leaves(x)
        return any(hasattr(leaf, "dtype") and leaf.dtype == jax.dtypes.float0 for leaf in leaves)

    def _f_bwd_rule(res, g):
        f1_vjp_func, f2_vjp_func, group_sizes = res
        f1_grads = f1_vjp_func(g)
        f2_grads = f2_vjp_func(g)
        out_grads = []
        jax.debug.print("Group sizes: {}", group_sizes)
        for i, (g1, g2) in enumerate(zip(f1_grads, f2_grads)):
            # Integer/non-differentiable arguments produce float0 gradients
            # which don't support arithmetic. Pass them through without comparison.
            if _has_float0(g1):
                out_grads.append(g1)
            elif isinstance(g1, jnp.ndarray):
                # jax.debug.print("F1 {name}: min={min}, max={max}, mean={mean}, std={std}", name=name + f"_grad_{i}", min=jnp.min(g1), max=jnp.max(g1), mean=jnp.mean(g1), std=jnp.std(g1))
                # jax.debug.print("F2 {name}: min={min}, max={max}, mean={mean}, std={std}", name=name + f"_grad_{i}", min=jnp.min(g2), max=jnp.max(g2), mean=jnp.mean(g2), std=jnp.std(g2))
                # if i == 1: # wgrad
                #     jax.debug.callback(_tensor_diff_to_image, g1, g2)
                out_grads.append(compare(g1, g2, name + f"_grad_{i}"))
            else:
                # g1 is a pytree of arrays — compare leaf by leaf
                g1_flat, tree_def = jax.tree_util.tree_flatten(g1)
                g2_flat, _ = jax.tree_util.tree_flatten(g2)
                compared = [
                    compare(a, b, name + f"_grad_{i}_{j}")
                    for j, (a, b) in enumerate(zip(g1_flat, g2_flat))
                ]
                out_grads.append(jax.tree_util.tree_unflatten(tree_def, compared))
        return tuple(out_grads)

    _f.defvjp(_f_fwd_rule, _f_bwd_rule)

    return _f


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
