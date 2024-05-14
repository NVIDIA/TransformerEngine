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

# See which frameworks are available
try:
    import torch
    _FOUND_PYTORCH = True
except ImportError:
    _FOUND_PYTORCH = False
try:
    import jax
    _FOUND_JAX = True
except ImportError:
    _FOUND_JAX = False
try:
    import paddle
    _FOUND_PADDLE = True
except ImportError:
    _FOUND_PADDLE = False

# Import framework submodules
# Note: Load module lazily if import fails. This way a useful import
# error will be thrown if the user attempts to access the module.
if _FOUND_PYTORCH:
    try:
        from . import pytorch
    except ImportError:
        pytorch = _lazy_import("transformer_engine.pytorch")
if _FOUND_JAX:
    try:
        from . import jax
    except ImportError:
        jax = _lazy_import("transformer_engine.jax")
if _FOUND_PADDLE:
    try:
        from . import paddle
    except ImportError:
        paddle = _lazy_import("transformer_engine.paddle")
