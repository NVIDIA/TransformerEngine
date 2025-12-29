#!/usr/bin/env python3
# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Example: Out-of-tree Backend Registration

Use case: Standalone plugin package (closed-source / third-party)

Run:
    # Method 1: Load plugin module via environment variable
    TE_FL_PLUGIN_MODULES=my_vendor_plugin python example_outtree.py

    # Method 2: Install plugin package with entry_points via pip
    pip install my-vendor-plugin
    python example_outtree.py
"""

import sys
import types
import torch


# ============================================================
# Step 1: Create plugin module (simulates a pip-installed package)
# ============================================================
def create_plugin_module():
    """
    Simulate a standalone plugin module.

    In practice, this code would be in a separate pip package, e.g.:
    - my_vendor_plugin/__init__.py
    """

    # Create module
    plugin_module = types.ModuleType("my_vendor_plugin")

    # Define operator implementation
    def my_rmsnorm_fwd(input, weight, eps=1e-5, **kwargs):
        """Custom RMSNorm implementation"""
        print("  >>> [MyVendorPlugin] my_rmsnorm_fwd called!")
        variance = input.pow(2).mean(-1, keepdim=True)
        output = input * torch.rsqrt(variance + eps) * weight
        rsigma = torch.rsqrt(variance + eps)
        return output, rsigma

    my_rmsnorm_fwd._is_available = lambda: True

    # Define register function (must have 'register' or 'te_fl_register' function)
    def register(registry):
        """
        Plugin registration function - called automatically by TE-FL.

        Args:
            registry: OpRegistry instance
        """
        from transformer_engine.plugin.core import (
            OpImpl,
            BackendImplKind,
        )

        print("[MyVendorPlugin] Registering operator implementations...")

        registry.register_impl(OpImpl(
            op_name="rmsnorm_fwd",
            impl_id="vendor.myvendor",
            kind=BackendImplKind.VENDOR,
            vendor="myvendor",
            fn=my_rmsnorm_fwd,
            priority=200,
        ))

        print("[MyVendorPlugin] Registration complete!")

    # Add register function to module
    plugin_module.register = register

    return plugin_module


# ============================================================
# Step 2: Register plugin module to sys.modules (simulates pip install)
# ============================================================
plugin = create_plugin_module()
sys.modules["my_vendor_plugin"] = plugin


# ============================================================
# Step 3: Set environment variables for TE-FL auto-discovery
# ============================================================
import os
os.environ["TE_FL_PLUGIN_MODULES"] = "my_vendor_plugin"
os.environ["TE_FL_PREFER"] = "vendor"  # Prefer vendor backend


# ============================================================
# Step 4: Import TE-FL (will auto-discover and load plugin)
# ============================================================
from transformer_engine.plugin.core import (
    get_manager,
    reset_default_manager,
)

# Reset manager to trigger plugin discovery
reset_default_manager()
manager = get_manager()


# ============================================================
# Step 5: Call operator
# ============================================================
input_tensor = torch.randn(2, 4, 8)
weight = torch.ones(8)

print("\nCalling rmsnorm_fwd:")
output, rsigma = manager.call("rmsnorm_fwd", input_tensor, weight, eps=1e-5)

print(f"\nInput shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print("\nSuccess! Your out-of-tree plugin was loaded and used.")
