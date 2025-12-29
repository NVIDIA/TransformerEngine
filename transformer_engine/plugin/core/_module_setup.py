# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Module setup for core plugin system.

This module handles the registration of core modules in sys.modules
with both full and short names to support relative imports in backends.
"""

import sys
from pathlib import Path


def setup_module_aliases():
    """
    Register core modules under both full and short names.

    This allows backends to use relative imports like:
        from ...ops import TEFLBackendBase
        from ...types import OpImpl, BackendImplKind

    And ensures they work correctly regardless of how the module is imported.
    """
    # Get the current package
    current_package = sys.modules.get("transformer_engine.plugin.core")
    if current_package is None:
        return

    # Register the main package under short name
    sys.modules["core"] = current_package

    # List of submodules to register
    submodule_names = [
        "ops",
        "logger",
        "types",
        "logger_manager",
        "policy",
        "operator_registry",
        "registry",
        "discovery",
    ]

    # Register each submodule under short name
    for name in submodule_names:
        full_name = f"transformer_engine.plugin.core.{name}"
        short_name = f"core.{name}"

        if full_name in sys.modules and short_name not in sys.modules:
            sys.modules[short_name] = sys.modules[full_name]

    # Register backends package
    backends_full = "transformer_engine.plugin.core.backends"
    backends_short = "core.backends"
    if backends_full in sys.modules and backends_short not in sys.modules:
        sys.modules[backends_short] = sys.modules[backends_full]

    # Register parent plugin package if needed
    if "transformer_engine.plugin" not in sys.modules:
        import types
        plugin_dir = Path(__file__).parent.parent
        plugin_pkg = types.ModuleType("transformer_engine.plugin")
        plugin_pkg.__path__ = [str(plugin_dir)]
        sys.modules["transformer_engine.plugin"] = plugin_pkg


def register_as_transformer_engine_torch():
    """
    Register the tefl module as transformer_engine_torch.

    This provides backward compatibility with code that expects
    transformer_engine_torch to be available.
    """
    # Only register if not already present
    if "transformer_engine_torch" in sys.modules:
        return

    try:
        from .ops import get_tefl_module
        tefl_module = get_tefl_module()
        sys.modules["transformer_engine_torch"] = tefl_module
    except Exception as e:
        import traceback
        print(f"[TEFL Setup] Warning: Could not register transformer_engine_torch: {e}")
        traceback.print_exc()

        # Create a minimal placeholder module to avoid import errors
        # This allows the system to at least import without crashing
        import types
        placeholder = types.ModuleType("transformer_engine_torch")
        placeholder.__doc__ = "Placeholder module - TEFL backend not available"
        sys.modules["transformer_engine_torch"] = placeholder
