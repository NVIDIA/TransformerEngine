# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile compatible fuser for Transformer Engine operations."""

from __future__ import annotations
from typing import Any, Optional

import torch

from ...quantization import FP8GlobalStateManager
from ..op import BasicOperation, FusibleOperation

from .opaque_kwargs import opaque_kwargs_from_dicts
from .ops_container import OpsContainer
from .operators import fused_forward_impl, NONE_RECIPE


class TorchCompileCompatibleFuser:
    """Fuser for torch.compile(fullgraph=True) compatibility.
    
    This class wraps a sequence of FusibleOperations and provides a callable
    that works with torch.compile without graph breaks. The fusion logic
    is hidden inside custom operators.
    
    Usage:
        ops = [LinearOp(...), BiasOp(...), ActivationOp(...)]
        fuser = TorchCompileCompatibleFuser(ops)
        
        @torch.compile(fullgraph=True)
        def forward(x):
            return fuser(x)
    
    Note: The fuser must be created OUTSIDE the compiled region, as OpsContainer
    is a reference-type opaque object.
    """
    
    def __init__(self, ops: list[FusibleOperation]) -> None:
        """Initialize the fuser with a list of operations.
        
        Args:
            ops: List of FusibleOperation instances (can include FusedOperations)
        """
        # Flatten to basic operations
        basic_ops: list[BasicOperation] = []
        for op in ops:
            if op.is_fused_op:
                basic_ops.extend(op.basic_ops)
            else:
                basic_ops.append(op)
        
        # Create OpsContainer (outside compiled region)
        self.ops_container = OpsContainer(basic_ops)
        
        # Cache num_ops and default kwargs (avoid accessing these in compiled region)
        self._num_ops = len(basic_ops)
        self._default_kwargs_opaque = opaque_kwargs_from_dicts([{}] * len(basic_ops))
        
        # Flatten parameters for autograd tracking
        self._flat_params = [p for op in basic_ops for p in op.parameters()]
        
        # Track extra inputs/outputs
        self.num_extra_inputs = sum(op.num_extra_inputs for op in basic_ops)
        self.num_extra_outputs = sum(op.num_extra_outputs for op in basic_ops)
        
        # Keep reference to basic ops for module compatibility
        self._basic_ops = basic_ops
    
    def __call__(
        self,
        input: torch.Tensor,
        *extra_inputs: torch.Tensor,
        basic_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply the fused operations to input.
        
        Args:
            input: Input tensor
            *extra_inputs: Extra tensor inputs for operations that need them
            basic_op_kwargs: Optional list of kwargs dicts, one per basic operation
            
        Returns:
            Output tensor, or tuple of (output, *extra_outputs) if any operation
            produces extra outputs.
        """
        # Get recipe from global state
        # Use NONE_RECIPE singleton when FP8 is disabled (cannot pass None to custom_op)
        if FP8GlobalStateManager.is_fp8_enabled():
            recipe = FP8GlobalStateManager.get_fp8_recipe()
        else:
            recipe = NONE_RECIPE
        
        # Create OpaqueKwargs
        # Use cached default kwargs to avoid accessing self._num_ops in compiled region
        if basic_op_kwargs is None:
            kwargs_opaque = self._default_kwargs_opaque
        else:
            kwargs_opaque = opaque_kwargs_from_dicts(basic_op_kwargs)
        
        # Verify extra inputs count
        if len(extra_inputs) != self.num_extra_inputs:
            raise ValueError(
                f"Expected {self.num_extra_inputs} extra inputs, "
                f"got {len(extra_inputs)}"
            )
        
        # Call the custom op - returns [output, *non_aliased_tensors_to_save, *extra_outputs]
        # Aliased tensors are NOT included (reconstructed in backward)
        flat_result = fused_forward_impl(
            input,
            self.ops_container,
            recipe,
            kwargs_opaque,
            self._flat_params,
            list(extra_inputs),
        )
        
        # Parse flat result
        output = flat_result[0]
        # non_aliased_tensors_to_save are in the middle (handled by autograd), we skip them
        # extra_outputs are at the end
        num_extra_outputs = self.num_extra_outputs
        if num_extra_outputs > 0:
            extra_outputs = flat_result[-num_extra_outputs:]
            return (output, *extra_outputs)
        return output
    
    def parameters(self):
        """Iterate over all parameters in the fused operations."""
        return iter(self._flat_params)
    
    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Iterate over named parameters."""
        for idx, op in enumerate(self._basic_ops):
            op_prefix = f"{prefix}op_{idx}." if prefix else f"op_{idx}."
            for name, param in op.named_parameters(prefix="", recurse=recurse):
                yield op_prefix + name, param
