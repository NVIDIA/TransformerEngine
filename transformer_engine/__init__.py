# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level package"""
import importlib.util
import sys
from types import ModuleType

from ._version import __version__
from . import common

def _lazy_import(name: str) -> ModuleType:
    """Construct a module that is imported the first time it is used"""
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module

# Lazily import frameworks
pytorch: ModuleType = _lazy_import("transformer_engine.pytorch")
jax: ModuleType = _lazy_import("transformer_engine.jax")
paddle: ModuleType = _lazy_import("transformer_engine.paddle")

__all__ = [
    "__version__",
    "common",
    "jax",
    "paddle",
    "pytorch",
]
