# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Callable, List, Optional, Tuple

from .logger_manager import get_logger

PLUGIN_GROUP = "te_fl.plugin"

PLUGIN_MODULES_ENV = "TE_FL_PLUGIN_MODULES"

logger = get_logger()

_discovered_plugin: List[Tuple[str, str, bool]] = []

def _log_debug(msg: str) -> None:
    logger.debug(msg)

def _log_info(msg: str) -> None:
    logger.info(msg)

def _log_warning(msg: str) -> None:
    logger.warning(msg)

def _log_error(msg: str) -> None:
    logger.error(msg)

def _get_entry_points():
    try:
        from importlib.metadata import entry_points
    except ImportError:
        try:
            from importlib_metadata import entry_points
        except ImportError:
            _log_debug("importlib.metadata not available, skipping entry points discovery")
            return []

    try:
        eps = entry_points()

        if hasattr(eps, "select"):
            return list(eps.select(group=PLUGIN_GROUP))

        if isinstance(eps, dict):
            return eps.get(PLUGIN_GROUP, [])

        if hasattr(eps, "get"):
            return eps.get(PLUGIN_GROUP, [])

        return []

    except Exception as e:
        _log_warning(f"Error accessing entry points: {e}")
        return []

def _call_register_function(
    obj: Any,
    registry_module: Any,
    source_name: str,
) -> bool:
    if callable(obj) and not isinstance(obj, type):
        try:
            obj(registry_module)
            _log_info(f"Registered plugin from {source_name} (direct callable)")
            return True
        except Exception as e:
            _log_error(f"Error calling plugin {source_name}: {e}")
            return False

    register_fn = getattr(obj, "te_fl_register", None) or getattr(obj, "register", None)

    if callable(register_fn):
        try:
            register_fn(registry_module)
            _log_info(f"Registered plugin from {source_name}")
            return True
        except Exception as e:
            _log_error(f"Error calling register function in {source_name}: {e}")
            return False

    _log_debug(f"No register function found in {source_name}")
    return False

def discover_from_entry_points(registry_module: Any) -> int:
    loaded = 0
    entry_points_list = _get_entry_points()

    if not entry_points_list:
        _log_debug("No entry points found for group: " + PLUGIN_GROUP)
        return 0

    _log_debug(f"Found {len(entry_points_list)} entry points")

    for ep in entry_points_list:
        ep_name = getattr(ep, "name", str(ep))
        try:
            _log_debug(f"Loading entry point: {ep_name}")
            obj = ep.load()

            if _call_register_function(obj, registry_module, f"entry_point:{ep_name}"):
                _discovered_plugin.append((ep_name, "entry_point", True))
                loaded += 1
            else:
                _discovered_plugin.append((ep_name, "entry_point", False))

        except Exception as e:
            _log_error(f"Failed to load entry point {ep_name}: {e}")
            _discovered_plugin.append((ep_name, "entry_point", False))

    return loaded

def discover_from_env_modules(registry_module: Any) -> int:
    modules_str = os.environ.get(PLUGIN_MODULES_ENV, "").strip()

    if not modules_str:
        return 0

    loaded = 0
    module_names = [m.strip() for m in modules_str.split(",") if m.strip()]

    _log_debug(f"Loading plugin from env var: {module_names}")

    for mod_name in module_names:
        try:
            _log_debug(f"Importing module: {mod_name}")
            mod = importlib.import_module(mod_name)

            if _call_register_function(mod, registry_module, f"env_module:{mod_name}"):
                _discovered_plugin.append((mod_name, "env_module", True))
                loaded += 1
            else:
                _discovered_plugin.append((mod_name, "env_module", False))

        except ImportError as e:
            _log_error(f"Failed to import plugin module {mod_name}: {e}")
            _discovered_plugin.append((mod_name, "env_module", False))
        except Exception as e:
            _log_error(f"Error loading plugin module {mod_name}: {e}")
            _discovered_plugin.append((mod_name, "env_module", False))

    return loaded

def discover_plugin(registry_module: Any) -> int:
    """
    Main plugin discovery function.

    Discovers and registers plugin from:
    1. Entry points (group: 'te_fl.plugin')
    2. Environment variable modules (TE_FL_PLUGIN_MODULES)

    Args:
        registry_module: OpRegistry instance to register plugin to

    Returns:
        Number of successfully loaded plugin
    """
    if registry_module is None:
        _log_warning("Registry module is None, skipping plugin discovery")
        return 0

    _log_debug("Starting plugin discovery...")

    total = 0

    total += discover_from_entry_points(registry_module)

    total += discover_from_env_modules(registry_module)

    _log_debug(f"Plugin discovery complete. Loaded {total} plugin.")

    return total

# Alias for compatibility with different naming conventions
discover_op_plugin = discover_plugin

def get_discovered_plugin() -> List[Tuple[str, str, bool]]:
    """Get list of discovered plugin (name, source, success)"""
    return _discovered_plugin.copy()

def clear_discovered_plugin() -> None:
    """Clear the discovered plugin list (for testing)"""
    _discovered_plugin.clear()


