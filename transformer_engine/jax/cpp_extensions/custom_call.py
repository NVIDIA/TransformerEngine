# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom call"""
from dataclasses import dataclass
from enum import IntEnum

import jax
from jax.interpreters import mlir
import transformer_engine_jax

from .misc import is_ffi_enabled

try:
    from jaxlib.hlo_helpers import custom_call
except ImportError:
    # Newer JAX changed its API. But we want to support a few JAX
    # version, so we still need this import.
    pass


class CustomCallAPIVersion(IntEnum):
    """Enum for selecting between old and new custom call registration API"""

    OPAQUE = 0
    FFI = 1


for _name, _value in transformer_engine_jax.registrations().items():
    if _name.endswith("_ffi"):
        if is_ffi_enabled():
            jax.ffi.register_ffi_target(
                _name, _value, platform="CUDA", api_version=CustomCallAPIVersion.FFI.value
            )
    else:
        jax.ffi.register_ffi_target(
            _name, _value, platform="CUDA", api_version=CustomCallAPIVersion.OPAQUE.value
        )


@dataclass
class CustomCallArgsWrapper:
    """
    wrapper of XLA custom call args
    """

    def __init__(
        self,
        output_types,
        operands,
        operand_shapes,
        operand_specific_layouts=None,
        output_specific_layouts=None,
    ):
        self.output_types = output_types
        self.operands = operands
        self.operand_layouts = CustomCallArgsWrapper.generate_layouts(
            operand_shapes, operand_specific_layouts
        )
        output_shapes = [x.shape for x in output_types]
        self.output_layouts = CustomCallArgsWrapper.generate_layouts(
            output_shapes, output_specific_layouts
        )

    @staticmethod
    def generate_layouts(shapes, specific_layouts):
        """
        setup layouts for XLA custom call
        """

        def default_layout(shape):
            return range(len(shape) - 1, -1, -1)

        if specific_layouts is None:
            specific_layouts = {}

        layouts = []
        for idx, shape in enumerate(shapes):
            if idx in specific_layouts:
                layouts.append(specific_layouts[idx])
            else:
                layouts.append(default_layout(shape))
        return layouts


def custom_caller(name, args, opaque, has_side_effect, **kwargs):
    """
    XLA custom call warpper
    """
    if hasattr(mlir, "custom_call"):
        out = mlir.custom_call(
            name,
            result_types=args.output_types,
            operands=args.operands,
            operand_layouts=args.operand_layouts,
            result_layouts=args.output_layouts,
            backend_config=opaque,
            has_side_effect=has_side_effect,
            **kwargs,
        ).results
    else:
        # Need to disable one pylint error as the second function
        # parameter name recenctly in JAX. Otherwise we won't be
        # compatible with multiple JAX version.
        out = custom_call(  # pylint: disable=too-many-function-args
            name,
            args.output_types,
            operands=args.operands,
            operand_layouts=args.operand_layouts,
            result_layouts=args.output_layouts,
            backend_config=opaque,
            has_side_effect=has_side_effect,
            **kwargs,
        )
    return out
