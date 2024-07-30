# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""

import glob
import sysconfig
import subprocess
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


def _load_cudnn():
    """Load CUDNN shared library."""

    lib_path = glob.glob(
        os.path.join(
            sysconfig.get_path("purelib"),
            f"nvidia/cudnn/lib/libcudnn.{_get_sys_extension()}.*[0-9]",
        )
    )

    if lib_path:
        assert (
            len(lib_path) == 1
        ), f"Found {len(lib_path)} libcudnn.{_get_sys_extension()}.x in nvidia-cudnn-cuXX."
        return ctypes.CDLL(lib_path[0], mode=ctypes.RTLD_GLOBAL)

    cudnn_home = os.environ.get("CUDNN_HOME") or os.environ.get("CUDNN_PATH")
    if cudnn_home:
        libs = glob.glob(f"{cudnn_home}/**/libcudnn.{_get_sys_extension()}*", recursive=True)
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        libs = glob.glob(f"{cuda_home}/**/libcudnn.{_get_sys_extension()}*", recursive=True)
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    return ctypes.CDLL(f"libcudnn.{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)


def _load_library():
    """Load shared library with Transformer Engine C extensions"""

    so_path = get_te_path() / "transformer_engine" / f"libtransformer_engine.{_get_sys_extension()}"
    if not so_path.exists():
        so_path = get_te_path() / f"libtransformer_engine.{_get_sys_extension()}"
    assert so_path.exists(), f"Could not find libtransformer_engine.{_get_sys_extension()}"

    return ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)


def _load_nvrtc():
    """Load NVRTC shared library."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        libs = glob.glob(f"{cuda_home}/**/libnvrtc.{_get_sys_extension()}*", recursive=True)
        libs = list(filter(lambda x: not ("stub" in x or "libnvrtc-builtins" in x), libs))
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    libs = subprocess.check_output("ldconfig -p | grep 'libnvrtc'", shell=True)
    libs = libs.decode("utf-8").split("\n")
    sos = []
    for lib in libs:
        if "stub" in lib or "libnvrtc-builtins" in lib:
            continue
        if "libnvrtc" in lib and "=>" in lib:
            sos.append(lib.split(">")[1].strip())
    if sos:
        return ctypes.CDLL(sos[0], mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(f"libnvrtc.{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)


if "NVTE_PROJECT_BUILDING" not in os.environ or bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
    _CUDNN_LIB_CTYPES = _load_cudnn()
    _NVRTC_LIB_CTYPES = _load_nvrtc()
    _TE_LIB_CTYPES = _load_library()
