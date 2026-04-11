# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Group-size alignment validation FFI primitive."""

import jax.numpy as jnp
from jax import ffi

from transformer_engine.jax.cpp_extensions.base import BasePrimitive, register_primitive

__all__ = ["validate_group_sizes"]


class ValidateGroupSizesPrimitive(BasePrimitive):
    """
    Pass-through primitive that validates MoE group size alignment on the host.

    Aliases input to output (no memcpy). Not CUDA-graph compatible: performs a
    device-to-host copy of group_sizes, stream-syncs, then checks that every
    element is divisible by align_size.
    """

    name = "te_validate_group_sizes_ffi"
    multiple_results = False
    impl_static_args = (1,)  # align_size is a static Python int (arg index 1)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(group_sizes_aval, *, align_size):
        """validate_group_sizes abstract"""
        assert group_sizes_aval.dtype == jnp.int32, (
            f"group_sizes must have dtype int32, got {group_sizes_aval.dtype}"
        )
        del align_size
        return group_sizes_aval  # pass-through: same shape and dtype

    @staticmethod
    def lowering(ctx, group_sizes, align_size):
        """validate_group_sizes lowering rules"""
        return ffi.ffi_lowering(
            ValidateGroupSizesPrimitive.name,
            operand_output_aliases={0: 0},  # donate input buffer to output buffer
        )(ctx, group_sizes, config={"align_size": align_size})

    @staticmethod
    def impl(group_sizes, align_size):
        """validate_group_sizes implementation"""
        assert ValidateGroupSizesPrimitive.inner_primitive is not None
        return ValidateGroupSizesPrimitive.inner_primitive.bind(group_sizes, align_size=align_size)


register_primitive(ValidateGroupSizesPrimitive)


def validate_group_sizes(group_sizes: jnp.ndarray, align_size: int) -> jnp.ndarray:
    """Pass-through primitive that asserts all group sizes are divisible by align_size.

    Not CUDA-graph compatible. On each call, copies group_sizes to host and
    stream-syncs to perform the alignment check. Enable in grouped_dense via
    NVTE_JAX_VALIDATE_GROUP_SIZE_ALIGNMENT=1.

    Args:
        group_sizes: int32 array of shape (num_experts,) specifying per-group sizes.
        align_size: Required alignment divisor (e.g. 128).

    Returns:
        group_sizes unchanged (input buffer aliased to output, no memcpy).

    Raises:
        RuntimeError: If any group_sizes[i] % align_size != 0, with the index
            and value of the first offending entry in the error message.
    """
    assert ValidateGroupSizesPrimitive.outer_primitive is not None, (
        "ValidateGroupSizesPrimitive FFI is not registered. Please ensure the C++ extension is"
        " properly built and registered."
    )
    return ValidateGroupSizesPrimitive.outer_primitive.bind(group_sizes, align_size=align_size)
