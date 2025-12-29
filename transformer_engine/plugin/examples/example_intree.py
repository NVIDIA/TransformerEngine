#!/usr/bin/env python3
# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Example: In-tree Backend Registration

Use case: Add implementation directly to the codebase (open source contribution)

Run:
    python example_intree.py
"""

import torch
from transformer_engine.plugin.core import (
    OpRegistry,
    OpManager,
    OpImpl,
    BackendImplKind,
    SelectionPolicy,
    set_global_policy,
)


# ============================================================
# Step 1: Define your operator implementation
# ============================================================
def my_rmsnorm_fwd(input, weight, eps=1e-5, **kwargs):
    """Custom RMSNorm implementation"""
    print("  >>> [MyBackend] my_rmsnorm_fwd called!")
    variance = input.pow(2).mean(-1, keepdim=True)
    output = input * torch.rsqrt(variance + eps) * weight
    rsigma = torch.rsqrt(variance + eps)
    return output, rsigma


# Optional: Define availability check function
my_rmsnorm_fwd._is_available = lambda: True


# ============================================================
# Step 2: Register to Registry
# ============================================================
registry = OpRegistry()

registry.register_impl(OpImpl(
    op_name="rmsnorm_fwd",           # Operator name
    impl_id="vendor.mybackend",      # Implementation ID (unique identifier)
    kind=BackendImplKind.VENDOR,     # Type: VENDOR / DEFAULT / REFERENCE
    vendor="mybackend",              # Vendor name
    fn=my_rmsnorm_fwd,               # Implementation function
    priority=200,                    # Priority (higher = preferred)
))


# ============================================================
# Step 3: Create Manager and call operator
# ============================================================
manager = OpManager(registry)

# Set policy: prefer vendor backend
set_global_policy(SelectionPolicy(prefer="vendor"))

# Prepare test data
input_tensor = torch.randn(2, 4, 8)
weight = torch.ones(8)

# Call operator - will automatically select highest priority implementation
print("\nCalling rmsnorm_fwd:")
output, rsigma = manager.call("rmsnorm_fwd", input_tensor, weight, eps=1e-5)

print(f"\nInput shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print("\nSuccess! Your custom backend was used.")
