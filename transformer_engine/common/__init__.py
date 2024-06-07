# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""

import ctypes
import os
import platform
from pathlib import Path

import transformer_engine


def get_te_path():
    """Find Transformer Engine install path using pip"""
    return Path(transformer_engine.__path__[0]).parent


def _get_sys_extension():
    system = platform.system()
    if system == "Linux":
        extension = "so"
    elif system == "Darwin":
        extension = "dylib"
    elif system == "Windows":
        extension = "dll"
    else:
        raise RuntimeError(f"Unsupported operating system ({system})")

    return extension


def _load_library():
    """Load shared library with Transformer Engine C extensions"""

    so_path = get_te_path() / "transformer_engine" / f"libtransformer_engine.{_get_sys_extension()}"
    if not so_path.exists():
        so_path = get_te_path() / f"libtransformer_engine.{_get_sys_extension()}"
    assert so_path.exists(), f"Could not find libtransformer_engine.{_get_sys_extension()}"

    return ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)


if "NVTE_PROJECT_BUILDING" not in os.environ:
    _TE_LIB_CTYPES = _load_library()
