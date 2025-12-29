# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from .core import (
    TEFLBackendBase,
    TEFLModule,
    get_tefl_module as _get_tefl_module,
    get_registry,
)

def __getattr__(name):
    if name == "tefl":
        return _get_tefl_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TEFLBackendBase",
    "TEFLModule",
    "get_tefl_module",
    "get_registry",
    "tefl",
]
