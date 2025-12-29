# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from .types import BackendImplKind, OpImpl, match_token

from .ops import (
    TEFLBackendBase,
    TEFLModule,
    get_tefl_module,
    reset_tefl_module,
    get_registry,
    get_manager,
    reset_registry,
)

from .logger_manager import Logger, LoggerManager
from .policy import (
    SelectionPolicy,
    PolicyManager,
    get_policy,
    set_global_policy,
    reset_global_policy,
    policy_context,
    policy_from_env,
    get_policy_epoch,
    bump_policy_epoch,
    with_strict_mode,
    with_preference,
    with_allowed_vendors,
    with_denied_vendors,
    PREFER_DEFAULT,
    PREFER_VENDOR,
    PREFER_REFERENCE,
    VALID_PREFER_VALUES,
)

from .manager import OpManager, get_default_manager, reset_default_manager
from .registry import OpRegistry


from .discovery import (
    discover_plugin,
    discover_from_entry_points,
    discover_from_env_modules,
    get_discovered_plugin,
    clear_discovered_plugin,
    PLUGIN_GROUP,
    PLUGIN_MODULES_ENV,
)

# Setup module aliases BEFORE importing backends to support relative imports
from ._module_setup import setup_module_aliases, register_as_transformer_engine_torch
setup_module_aliases()

# Import backends - this loads all available backends (flagos, reference, vendor/cuda, etc.)
from . import backends

# Register transformer_engine_torch AFTER backends are loaded
# so that get_tefl_module() can find a registered backend
register_as_transformer_engine_torch()
