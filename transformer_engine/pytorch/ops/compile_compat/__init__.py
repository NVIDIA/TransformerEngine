# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile compatibility module for Transformer Engine operations.

This module provides components to make te.ops work with 
torch.compile(fullgraph=True) by using custom operators that wrap fusion logic.

Usage:
    from transformer_engine.pytorch.ops.compile_compat import TorchCompileCompatibleFuser
    
    # Create fuser OUTSIDE compiled region
    fuser = TorchCompileCompatibleFuser([op1, op2, op3])
    
    @torch.compile(fullgraph=True)
    def forward(x):
        return fuser(x)
"""

# Import and re-export public API
# Note: NoneRecipe is used as sentinel when recipe is None (FP8 disabled)
from .tensor_info import TensorInfo, TensorInfoList, PseudoForwardResult
from .opaque_kwargs import OpaqueKwargs
from .ops_container import OpsContainer
from .operators import fused_forward_impl, fused_backward_impl, NoneRecipe, NONE_RECIPE
from .fuser import TorchCompileCompatibleFuser

__all__ = [
    # Main API
    "TorchCompileCompatibleFuser",
    
    # Supporting classes
    "TensorInfo",
    "TensorInfoList",
    "PseudoForwardResult", 
    "OpaqueKwargs",
    "OpsContainer",
    "NoneRecipe",
    "NONE_RECIPE",
    
    # Custom operators (for advanced usage)
    "fused_forward_impl",
    "fused_backward_impl",
]
