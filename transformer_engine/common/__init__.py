# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""

import sys
import glob
import sysconfig
import subprocess
import ctypes
import logging
import os
import platform
import importlib
import functools
from pathlib import Path
from importlib.metadata import version, metadata, PackageNotFoundError


_logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _is_pip_package_installed(package):
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
def _find_shared_object_in_te_dir(te_path: Path, prefix: str):
    """
    Find a shared object file of given prefix in the top level TE directory.
    Only the following locations are searched to avoid stray SOs and build
    artifacts:
        1. The given top level directory (editable install).
        2. `transformer_engine` named directories (source install).
        3. `wheel_lib` named directories (PyPI install).

    Returns None if no shared object files are found.
    Raises an error if multiple shared object files are found.
    """

    # Ensure top level dir exists and has the module. before searching.
    if not te_path.exists() or not (te_path / "transformer_engine").exists():
        return None

    files = []
    search_paths = (
        te_path,
        te_path / "transformer_engine",
        te_path / "transformer_engine/wheel_lib",
        te_path / "wheel_lib",
    )

    # Search.
    for dirname, _, names in os.walk(te_path):
        if Path(dirname) in search_paths:
            for name in names:
                if name.startswith(prefix) and name.endswith(f".{_get_sys_extension()}"):
                    files.append(Path(dirname, name))

    if len(files) == 0:
        return None
    if len(files) == 1:
        return files[0]
    raise RuntimeError(f"Multiple files found: {files}")


@functools.lru_cache(maxsize=None)
def _get_shared_object_file(library: str) -> Path:
    """
    Return the path of the shared object file for the given TE
    library, one of 'core', 'torch', or 'jax'.

    Several factors affect finding the correct location of the shared object:
        1. System and environment.
        2. If the installation is from source or via PyPI.
            - Source installed .sos are placed in top level dir
            - Wheel/PyPI installed .sos are placed in 'wheel_lib' dir to avoid conflicts.
        3. For source installations, is the install editable/inplace?
        4. The user directory from where TE is being imported.
    """

    # Check provided input and determine the correct prefix for .so.
    assert library in ("core", "torch", "jax"), f"Unsupported TE library {library}."
    if library == "core":
        so_prefix = "libtransformer_engine"
    else:
        so_prefix = f"transformer_engine_{library}"

    # Check TE install location (will be local if TE is available in current dir for import).
    te_install_dir = Path(importlib.util.find_spec("transformer_engine").origin).parent.parent
    so_path_in_install_dir = _find_shared_object_in_te_dir(te_install_dir, so_prefix)

    # Check default python package install location in system.
    site_packages_dir = Path(sysconfig.get_paths()["purelib"])
    so_path_in_default_dir = _find_shared_object_in_te_dir(site_packages_dir, so_prefix)

    # Case 1: Typical user workflow: Both locations are the same, return any result.
    if te_install_dir == site_packages_dir:
        assert (
            so_path_in_install_dir is not None
        ), f"Could not find shared object file for Transformer Engine {library} lib."
        return so_path_in_install_dir

    # Case 2: ERR! Both locations are different but returned a valid result.
    # NOTE: Unlike for source installations, pip does not wipe out artifacts from
    # editable builds. In case developers are executing inside a TE directory via
    # an inplace build, and then move to a regular build, the local shared object
    # file will be incorrectly picked up without the following logic.
    if so_path_in_install_dir is not None and so_path_in_default_dir is not None:
        raise RuntimeError(
            f"Found multiple shared object files: {so_path_in_install_dir} and"
            f" {so_path_in_default_dir}. Remove local shared objects installed"
            f" here {so_path_in_install_dir} or change the working directory to"
            "execute from outside TE."
        )

    # Case 3: Typical dev workflow: Editable install
    if so_path_in_install_dir is not None:
        return so_path_in_install_dir

    # Case 4: Executing from inside a TE directory without an inplace build available.
    if so_path_in_default_dir is not None:
        return so_path_in_default_dir

    raise RuntimeError(f"Could not find shared object file for Transformer Engine {library} lib.")


@functools.lru_cache(maxsize=None)
def load_framework_extension(framework: str):
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


@functools.lru_cache(maxsize=None)
def _load_nvidia_cuda_library(lib_name: str):
    """
    Attempts to load shared object file installed via pip.

    `lib_name`: Name of package as found in the `nvidia` dir in python environment.
    """

    so_paths = glob.glob(
        os.path.join(
            sysconfig.get_path("purelib"),
            f"nvidia/{lib_name}/lib/lib*.{_get_sys_extension()}.*[0-9]",
        )
    )

    path_found = len(so_paths) > 0
    ctypes_handles = []

    if path_found:
        for so_path in so_paths:
            ctypes_handles.append(ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL))

    return path_found, ctypes_handles


@functools.lru_cache(maxsize=None)
def _nvidia_cudart_include_dir():
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
        libs = glob.glob(f"{cudnn_home}/**/libcudnn.{_get_sys_extension()}*", recursive=True)
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    # Attempt to locate cuDNN in CUDA_HOME, CUDA_PATH or /usr/local/cuda
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    libs = glob.glob(f"{cuda_home}/**/libcudnn.{_get_sys_extension()}*", recursive=True)
    libs.sort(reverse=True, key=os.path.basename)
    if libs:
        return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    # Attempt to locate cuDNN in Python dist-packages
    found, handle = _load_nvidia_cuda_library("cudnn")
    if found:
        return handle

    # If all else fails, assume that it is in LD_LIBRARY_PATH and error out otherwise
    return ctypes.CDLL(f"libcudnn.{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)


@functools.lru_cache(maxsize=None)
def _load_nvrtc():
    """Load NVRTC shared library."""
    # Attempt to locate NVRTC in CUDA_HOME, CUDA_PATH or /usr/local/cuda
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    libs = glob.glob(f"{cuda_home}/**/libnvrtc.{_get_sys_extension()}*", recursive=True)
    libs = list(filter(lambda x: not ("stub" in x or "libnvrtc-builtins" in x), libs))
    libs.sort(reverse=True, key=os.path.basename)
    if libs:
        return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    # Attempt to locate NVRTC in Python dist-packages
    found, handle = _load_nvidia_cuda_library("cuda_nvrtc")
    if found:
        return handle

    # Attempt to locate NVRTC via ldconfig
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

    # If all else fails, assume that it is in LD_LIBRARY_PATH and error out otherwise
    return ctypes.CDLL(f"libnvrtc.{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)


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
