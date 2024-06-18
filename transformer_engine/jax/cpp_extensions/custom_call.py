# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom call"""
from dataclasses import dataclass

from jax.lib import xla_client
from jax.interpreters import mlir

from transformer_engine import transformer_engine_jax

from .misc import jax_version_meet_requirement

try:
    from jaxlib.hlo_helpers import custom_call
except ImportError:
    # Newer JAX changed its API. But we want to support a few JAX
    # version, so we still need this import.
    pass

for _name, _value in transformer_engine_jax.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")


if jax_version_meet_requirement():
    for _name, _value in transformer_engine_jax.registrations_with_ffi().items():
        # TODO: add prep, init, exec_capsule later
        xla_client.register_custom_call_target(_name, _value, platform="CUDA", api_version=1)


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
        self.output_shapes = [x.shape for x in output_types]
        self.output_layouts = CustomCallArgsWrapper.generate_layouts(self.output_shapes,
                                                                     output_specific_layouts)

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
            **kwargs
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
            **kwargs
        )
    return out

#   Reference for custom_call from JAX
#   (https://github.com/google/jax/blob/956226c929a34741c1f0a396b96e721c61ae5794/
#   jax/_src/interpreters/mlir.py#L2589)
#    def custom_call(
#        call_target_name: str,
#        *,
#        result_types: Sequence[ir.Type],
#        operands: Sequence[ir.Value],
#        backend_config: str | bytes | dict[str, ir.Attribute] = "",
#        has_side_effect: bool = False,
#        result_shapes: Sequence[ir.Value] | None = None,
#        called_computations: Sequence[str] = (),
#        api_version: int = 2,
#        operand_output_aliases: dict[int, int] | None = None,
#        operand_layouts: Sequence[Sequence[int]] | None = None,
#        result_layouts: Sequence[Sequence[int]] | None = None,
#        extra_attributes: dict[str, ir.Attribute] | None = None,
#        ) -> ir.Operation:

def custom_caller_with_ffi(name, args, backend_config, **kwargs):
    """
    New XLA custom call warpper
    """

    out = mlir.custom_call(name,
                           api_version = 4,
                           result_types=args.output_types,
                           operands=args.operands,
                           backend_config=backend_config,
                           result_shapes=args.output_shapes,
                           operand_layouts=args.operand_layouts,
                           result_layouts=args.output_layouts,
                           **kwargs,
                           ).results
    return out
