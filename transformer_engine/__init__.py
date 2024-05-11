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

# Import framework submodules
# Note: Load module lazily if import fails. This way a useful import
# error will be thrown if the user attempts to access the module.
try:
    from . import pytorch
except ImportError:
    pytorch = _lazy_import("transformer_engine.pytorch")
try:
    from . import jax
except ImportError:
    jax = _lazy_import("transformer_engine.jax")
try:
    from . import paddle
except ImportError:
    paddle = _lazy_import("transformer_engine.paddle")

__all__ = [
    "__version__",
    "common",
    "jax",
    "paddle",
    "pytorch",
]
