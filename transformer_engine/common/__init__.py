# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""
import ctypes
import os
import platform
import subprocess


def get_te_path():
    """Find Transformer Engine install path using pip"""

    command = ["pip", "show", "transformer_engine"]
    result = subprocess.run(command, capture_output=True, check=True, text=True)
    result = result.stdout.replace("\n", ":").split(":")
    return result[result.index("Location")+1].strip()


def _load_library():
    """Load shared library with Transformer Engine C extensions"""

    system = platform.system()
    if system == "Linux":
        extension = "so"
    elif system == "Darwin":
        extension = "dylib"
    elif system == "Windows":
        extension = "dll"
    else:
        raise RuntimeError(f"Unsupported operating system ({system})")
    lib_name = "libtransformer_engine." + extension
    dll_path = get_te_path()
    dll_path = os.path.join(dll_path, lib_name)

    return ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)


def _load_mpi():
    """Load MPI shared library"""

    system = platform.system()
    if system == "Linux":
        extension = "so"
    elif system == "Darwin":
        extension = "dylib"
    elif system == "Windows":
        extension = "dll"
    else:
        raise RuntimeError(f"Unsupported operating system ({system})")
    lib_name = "libmpi." + extension
    MPI_HOME = os.environ.get("MPI_HOME", "/usr/local/mpi")
    NVTE_MPI_FOUND = os.path.exists(MPI_HOME)
    dll_path = os.path.join(MPI_HOME, "lib", lib_name)

    if NVTE_MPI_FOUND:
        return ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    return None


_TE_LIB_CTYPES = _load_mpi()
_TE_LIB_CTYPES = _load_library()
