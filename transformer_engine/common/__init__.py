# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""


def get_te_path():
    """Find TE path using pip"""

    import os

    te_info = os.popen("pip show transformer_engine").read().replace("\n", ":").split(":")
    return te_info[te_info.index("Location") + 1].strip()


def _load_library():
    """Load TE .so"""

    import os
    import ctypes
    import platform

    system = platform.system()
    if system == "Linux":
        extension = "so"
    elif system == "Darwin":
        extension = "dylib"
    elif system == "Windows":
        extension = "dll"
    else:
        raise "Unsupported operating system " + system + "."
    lib_name = "libtransformer_engine." + extension
    dll_path = get_te_path()
    dll_path = os.path.join(dll_path, lib_name)

    return ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)


_TE_LIB_CTYPES = _load_library()
