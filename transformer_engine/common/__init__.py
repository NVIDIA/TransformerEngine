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


def _load_mpi_and_ubuf():
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

    MPI_HOME = os.environ.get("MPI_HOME", "/usr/local/mpi")
    mpi_lib_name = "libmpi." + extension
    mpi_dll_path = os.path.join(MPI_HOME, "lib", mpi_lib_name)
    ubuf_lib_name = "libtransformer_engine_ubuf." + extension
    te_path = get_te_path()
    ubuf_dll_path = os.path.join(te_path, ubuf_lib_name)
    mpi_lib_exists = os.path.exists(mpi_dll_path)
    ubuf_lib_exists = os.path.exists(ubuf_dll_path)

    if mpi_lib_exists and ubuf_lib_exists:
        return (
            ctypes.CDLL(mpi_dll_path, mode=ctypes.RTLD_GLOBAL),
            ctypes.CDLL(ubuf_dll_path, mode=ctypes.RTLD_GLOBAL),
        )
    return None, None


_MPI_LIB_CTYPES, _UBUF_LIB_CTYPES = _load_mpi_and_ubuf()
_TE_LIB_CTYPES = _load_library()
