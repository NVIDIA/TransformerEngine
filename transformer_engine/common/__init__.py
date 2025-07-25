# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""

import ctypes
import functools
import glob
import importlib
from importlib.metadata import version, metadata, PackageNotFoundError
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
from typing import Optional


_logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _is_pip_package_installed(package) -> bool:
    """Check if the given package is installed via pip."""

    # This is needed because we only want to return true
    # if the python package is installed via pip, and not
    # if it's importable in the current directory due to
    # the presence of the shared library module.
    try:
        metadata(package)
    except PackageNotFoundError:
        return False
    return True


@functools.lru_cache(maxsize=None)
def _find_shared_object_in_te_dir(te_path: Path, prefix: str) -> Optional[Path]:
    """
    Find a shared object file with the given prefix within the top level TE directory.

    The following locations are searched:
        1. Top level directory (editable install).
        2. `transformer_engine` directory (source install).
        3. `wheel_lib` directory (PyPI install).

    Returns None if no shared object files are found.
    Raises an error if multiple shared object files are found.
    """

    # Ensure top level dir exists and has the module before searching.
    if not te_path.is_dir() or not (te_path / "transformer_engine").exists():
        return None

    files = []
    search_paths = (
        te_path,  # Editable build.
        te_path / "transformer_engine",  # Regular source build.
        te_path / "transformer_engine/wheel_lib",  # PyPI.
    )

    # Search.
    for dir_path in search_paths:
        if not dir_path.is_dir():
            continue
        for file_path in dir_path.iterdir():
            if file_path.name.startswith(prefix) and file_path.suffix == _get_sys_extension():
                files.append(file_path)

    if len(files) == 0:
        return None
    if len(files) == 1:
        return files[0]
    raise RuntimeError(f"Multiple files found: {files}")


@functools.lru_cache(maxsize=None)
def _get_shared_object_file(library: str) -> Path:
    """
    Path to shared object file for a Transformer Engine library.

    TE libraries are 'core', 'torch', or 'jax'. This function first
    searches in the imported TE directory, and then in the
    site-packages directory.

    """

    # Check provided input and determine the correct prefix for .so.
    assert library in ("core", "torch", "jax"), f"Unsupported TE library {library}."
    if library == "core":
        so_prefix = "libtransformer_engine"
    else:
        so_prefix = f"transformer_engine_{library}"

    # Search for shared lib in imported directory
    te_path = Path(importlib.util.find_spec("transformer_engine").origin).parent.parent
    so_path = _find_shared_object_in_te_dir(te_path, so_prefix)
    if so_path is not None:
        return so_path

    # Search for shared lib in site-packages directory
    te_path = Path(sysconfig.get_paths()["purelib"])
    so_path = _find_shared_object_in_te_dir(te_path, so_prefix)
    if so_path is not None:
        return so_path

    raise FileNotFoundError(
        f"Could not find shared object file for Transformer Engine {library} lib."
    )


@functools.lru_cache(maxsize=None)
def load_framework_extension(framework: str) -> None:
    """
    Load shared library with Transformer Engine framework bindings
    and check verify correctness if installed via PyPI.
    """

    # Supported frameworks.
    assert framework in ("jax", "torch"), f"Unsupported framework {framework}"

    # Name of the framework extension library.
    module_name = f"transformer_engine_{framework}"

    # Name of the pip extra dependency for framework extensions from PyPI.
    extra_dep_name = module_name
    if framework == "torch":
        extra_dep_name = "pytorch"

    # If the framework extension pip package is installed, it means that TE is installed via
    # PyPI. For this case we need to make sure that the metapackage, the core lib, and framework
    # extension are all installed via PyPI and have matching version.
    if _is_pip_package_installed(module_name):
        assert _is_pip_package_installed(
            "transformer_engine"
        ), "Could not find `transformer-engine`."
        assert _is_pip_package_installed(
            "transformer_engine_cu12"
        ), "Could not find `transformer-engine-cu12`."
        assert (
            version(module_name)
            == version("transformer-engine")
            == version("transformer-engine-cu12")
        ), (
            "TransformerEngine package version mismatch. Found"
            f" {module_name} v{version(module_name)}, transformer-engine"
            f" v{version('transformer-engine')}, and transformer-engine-cu12"
            f" v{version('transformer-engine-cu12')}. Install transformer-engine using "
            f"'pip3 install transformer-engine[{extra_dep_name}]==VERSION'"
        )

    # If the core package is installed via PyPI, log if
    # the framework extension is not found from PyPI.
    # Note: Should we error? This is a rare use case.
    if _is_pip_package_installed("transformer-engine-cu12"):
        if not _is_pip_package_installed(module_name):
            _logger.info(
                "Could not find package %s. Install transformer-engine using "
                f"'pip3 install transformer-engine[{extra_dep_name}]==VERSION'",
                module_name,
            )

    # After all checks are completed, load the shared object file.
    spec = importlib.util.spec_from_file_location(module_name, _get_shared_object_file(framework))
    solib = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = solib
    spec.loader.exec_module(solib)


@functools.lru_cache(maxsize=None)
def _get_sys_extension() -> str:
    """File extension for shared objects."""
    system = platform.system()

    if system == "Linux":
        return ".so"
    if system == "Darwin":
        return ".dylib"
    if system == "Windows":
        return ".dll"
    raise RuntimeError(f"Unsupported operating system ({system})")


@functools.lru_cache(maxsize=None)
def _load_nvidia_cuda_library(lib_name: str):
    """
    Attempts to load shared object file installed via pip.

    `lib_name`: Name of package as found in the `nvidia` dir in python environment.
    """

    so_paths = glob.glob(
        os.path.join(
            sysconfig.get_path("purelib"),
            f"nvidia/{lib_name}/lib/lib*{_get_sys_extension()}.*[0-9]",
        )
    )

    path_found = len(so_paths) > 0
    ctypes_handles = []

    if path_found:
        for so_path in so_paths:
            ctypes_handles.append(ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL))

    return path_found, ctypes_handles


@functools.lru_cache(maxsize=None)
def _nvidia_cudart_include_dir() -> str:
    """Returns the include directory for cuda_runtime.h if exists in python environment."""

    try:
        import nvidia
    except ModuleNotFoundError:
        return ""

    include_dir = Path(nvidia.__file__).parent / "cuda_runtime"
    return str(include_dir) if include_dir.exists() else ""


@functools.lru_cache(maxsize=None)
def _load_cudnn():
    """Load CUDNN shared library."""

    # Attempt to locate cuDNN in CUDNN_HOME or CUDNN_PATH, if either is set
    cudnn_home = os.environ.get("CUDNN_HOME") or os.environ.get("CUDNN_PATH")
    if cudnn_home:
        libs = glob.glob(f"{cudnn_home}/**/libcudnn{_get_sys_extension()}*", recursive=True)
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    # Attempt to locate cuDNN in CUDA_HOME, CUDA_PATH or /usr/local/cuda
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    libs = glob.glob(f"{cuda_home}/**/libcudnn{_get_sys_extension()}*", recursive=True)
    libs.sort(reverse=True, key=os.path.basename)
    if libs:
        return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    # Attempt to locate cuDNN in Python dist-packages
    found, handle = _load_nvidia_cuda_library("cudnn")
    if found:
        return handle

    # Attempt to locate libcudnn via ldconfig
    libs = subprocess.check_output(
        f"ldconfig -p | grep 'libcudnn{_get_sys_extension()}'", shell=True
    )
    libs = libs.decode("utf-8").split("\n")
    sos = []
    for lib in libs:
        if "libcudnn" in lib and "=>" in lib:
            sos.append(lib.split(">")[1].strip())
    if sos:
        return ctypes.CDLL(sos[0], mode=ctypes.RTLD_GLOBAL)

    # If all else fails, assume that it is in LD_LIBRARY_PATH and error out otherwise
    return ctypes.CDLL(f"libcudnn{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)


@functools.lru_cache(maxsize=None)
def _load_nvrtc():
    """Load NVRTC shared library."""
    # Attempt to locate NVRTC in CUDA_HOME, CUDA_PATH or /usr/local/cuda
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    libs = glob.glob(f"{cuda_home}/**/libnvrtc{_get_sys_extension()}*", recursive=True)
    libs = list(filter(lambda x: not ("stub" in x or "libnvrtc-builtins" in x), libs))
    libs.sort(reverse=True, key=os.path.basename)
    if libs:
        return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    # Attempt to locate NVRTC in Python dist-packages
    found, handle = _load_nvidia_cuda_library("cuda_nvrtc")
    if found:
        return handle

    # Attempt to locate NVRTC via ldconfig
    libs = subprocess.check_output(
        f"ldconfig -p | grep 'libnvrtc{_get_sys_extension()}'", shell=True
    )
    libs = libs.decode("utf-8").split("\n")
    sos = []
    for lib in libs:
        if "libnvrtc" in lib and "=>" in lib:
            sos.append(lib.split(">")[1].strip())
    if sos:
        return ctypes.CDLL(sos[0], mode=ctypes.RTLD_GLOBAL)

    # If all else fails, assume that it is in LD_LIBRARY_PATH and error out otherwise
    return ctypes.CDLL(f"libnvrtc{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)


@functools.lru_cache(maxsize=None)
def _load_core_library():
    """Load shared library with Transformer Engine C extensions"""
    return ctypes.CDLL(_get_shared_object_file("core"), mode=ctypes.RTLD_GLOBAL)


if "NVTE_PROJECT_BUILDING" not in os.environ or bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
    _CUDNN_LIB_CTYPES = _load_cudnn()
    _NVRTC_LIB_CTYPES = _load_nvrtc()
    _CUBLAS_LIB_CTYPES = _load_nvidia_cuda_library("cublas")
    _CUDART_LIB_CTYPES = _load_nvidia_cuda_library("cuda_runtime")
    _TE_LIB_CTYPES = _load_core_library()

    # Needed to find the correct headers for NVRTC kernels.
    if not os.getenv("NVTE_CUDA_INCLUDE_DIR") and _nvidia_cudart_include_dir():
        os.environ["NVTE_CUDA_INCLUDE_DIR"] = _nvidia_cudart_include_dir()
