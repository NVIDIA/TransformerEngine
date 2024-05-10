# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""
import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path


def get_te_dirs():
    """Find Transformer Engine install by looking up two dirs or query path using pip"""

    yield str(Path(__file__).parents[1])

    command = [sys.executable, "-m", "pip", "show", "transformer_engine"]
    result = subprocess.run(command, capture_output=True, check=True, text=True)
    result = result.stdout.replace("\n", ":").split(":")
    yield result[result.index("Location") + 1].strip()

def get_dll_paths(suffix):
    return map(lambda x: os.path.join(x, suffix), get_te_dirs()

def get_shared_library_ext()
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

def _load_ctypes(lib_name="libtransformer_engine.", optional=False):
    """Load shared library with Transformer Engine C extensions"""
    lib_name = lib_name + get_shared_library_ext()
    for dll_path in filter(os.path.exists, get_dll_paths(lib_name)):
        return ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    if not optional:
        raise RuntimeError(f"Unable to find library {lib_name}")
    return None


_TE_LIB_CTYPES = _load_ctypes(lib_name="libtransformer_engine.")
_UB_LIB_CTYPES = _load_ctypes(lib_name="libtransformer_engine_userbuffers.", optional=True)
