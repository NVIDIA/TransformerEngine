# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import functools
import glob
import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Tuple, Union


@functools.lru_cache(maxsize=None)
def debug_build_enabled() -> bool:
    """Whether to build with a debug configuration"""
    for arg in sys.argv:
        if arg == "--debug":
            sys.argv.remove(arg)
            return True
    if int(os.getenv("NVTE_BUILD_DEBUG", "0")):
        return True
    return False


@functools.lru_cache(maxsize=None)
def get_max_jobs_for_parallel_build() -> int:
    """Number of parallel jobs for Nina build"""

    # Default: maximum parallel jobs
    num_jobs = 0

    # Check environment variable
    if os.getenv("NVTE_BUILD_MAX_JOBS"):
        num_jobs = int(os.getenv("NVTE_BUILD_MAX_JOBS"))
    elif os.getenv("MAX_JOBS"):
        num_jobs = int(os.getenv("MAX_JOBS"))

    # Check command-line arguments
    for arg in sys.argv.copy():
        if arg.startswith("--parallel="):
            num_jobs = int(arg.replace("--parallel=", ""))
            sys.argv.remove(arg)

    return num_jobs


def all_files_in_dir(path, name_extension=None):
    all_files = []
    for dirname, _, names in os.walk(path):
        for name in names:
            if name_extension is not None and name_extension not in name:
                continue
            all_files.append(Path(dirname, name))
    return all_files


def remove_dups(_list: List):
    return list(set(_list))


def found_cmake() -> bool:
    """ "Check if valid CMake is available

    CMake 3.18 or newer is required.

    """

    # Check if CMake is available
    try:
        _cmake_bin = cmake_bin()
    except FileNotFoundError:
        return False

    # Query CMake for version info
    output = subprocess.run(
        [_cmake_bin, "--version"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"version\s*([\d.]+)", output.stdout)
    version = match.group(1).split(".")
    version = tuple(int(v) for v in version)
    return version >= (3, 18)


def cmake_bin() -> Path:
    """Get CMake executable

    Throws FileNotFoundError if not found.

    """

    # Search in CMake Python package
    _cmake_bin: Optional[Path] = None
    try:
        from cmake import CMAKE_BIN_DIR
    except ImportError:
        pass
    else:
        _cmake_bin = Path(CMAKE_BIN_DIR).resolve() / "cmake"
        if not _cmake_bin.is_file():
            _cmake_bin = None

    # Search in path
    if _cmake_bin is None:
        _cmake_bin = shutil.which("cmake")
        if _cmake_bin is not None:
            _cmake_bin = Path(_cmake_bin).resolve()

    # Return executable if found
    if _cmake_bin is None:
        raise FileNotFoundError("Could not find CMake executable")
    return _cmake_bin


def found_ninja() -> bool:
    """ "Check if Ninja is available"""
    return shutil.which("ninja") is not None


def found_pybind11() -> bool:
    """ "Check if pybind11 is available"""

    # Check if Python package is installed
    try:
        import pybind11
    except ImportError:
        pass
    else:
        return True

    # Check if CMake can find pybind11
    if not found_cmake():
        return False
    try:
        subprocess.run(
            [
                "cmake",
                "--find-package",
                "-DMODE=EXIST",
                "-DNAME=pybind11",
                "-DCOMPILER_ID=CXX",
                "-DLANGUAGE=CXX",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (CalledProcessError, OSError):
        pass
    else:
        return True
    return False


@functools.lru_cache(maxsize=None)
def cuda_path() -> Tuple[str, str]:
    """CUDA root path and NVCC binary path as a tuple.

    Throws FileNotFoundError if NVCC is not found."""
    # Try finding NVCC
    nvcc_bin: Optional[Path] = None
    if nvcc_bin is None and os.getenv("CUDA_HOME"):
        # Check in CUDA_HOME
        cuda_home = Path(os.getenv("CUDA_HOME"))
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if nvcc_bin is None:
        # Check if nvcc is in path
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin is not None:
            cuda_home = Path(nvcc_bin.rstrip("/bin/nvcc"))
            nvcc_bin = Path(nvcc_bin)
    if nvcc_bin is None:
        # Last-ditch guess in /usr/local/cuda
        cuda_home = Path("/usr/local/cuda")
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise FileNotFoundError(f"Could not find NVCC at {nvcc_bin}")

    return cuda_home, nvcc_bin


@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    version = cuda_version()
    if os.getenv("NVTE_CUDA_ARCHS") is None:
        os.environ["NVTE_CUDA_ARCHS"] = (
            "70;80;89;90;100;120" if version >= (12, 8) else "70;80;89;90"
        )
    return os.getenv("NVTE_CUDA_ARCHS")


def cuda_version() -> Tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) tuple."""
    # Query NVCC for version info
    _, nvcc_bin = cuda_path()
    output = subprocess.run(
        [nvcc_bin, "-V"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    version = match.group(1).split(".")
    return tuple(int(v) for v in version)


def get_frameworks() -> List[str]:
    """DL frameworks to build support for"""
    _frameworks: List[str] = []
    supported_frameworks = ["pytorch", "jax"]

    # Check environment variable
    if os.getenv("NVTE_FRAMEWORK"):
        _frameworks.extend(os.getenv("NVTE_FRAMEWORK").split(","))

    # Check command-line arguments
    for arg in sys.argv.copy():
        if arg.startswith("--framework="):
            _frameworks.extend(arg.replace("--framework=", "").split(","))
            sys.argv.remove(arg)

    # Detect installed frameworks if not explicitly specified
    if not _frameworks:
        try:
            import torch
        except ImportError:
            pass
        else:
            _frameworks.append("pytorch")
        try:
            import jax
        except ImportError:
            pass
        else:
            _frameworks.append("jax")

    # Special framework names
    if "all" in _frameworks:
        _frameworks = supported_frameworks.copy()
    if "none" in _frameworks:
        _frameworks = []

    # Check that frameworks are valid
    _frameworks = [framework.lower() for framework in _frameworks]
    for framework in _frameworks:
        if framework not in supported_frameworks:
            raise ValueError(f"Transformer Engine does not support framework={framework}")

    return _frameworks


def copy_common_headers(
    src_dir: Union[Path, str],
    dst_dir: Union[Path, str],
) -> None:
    """Copy headers from core library

    src_dir should be the transformer_engine directory within the root
    Transformer Engine repository. All .h and .cuh files within
    transformer_engine/common are copied into dst_dir. Relative paths
    are preserved.

    """

    # Find common header files in src dir
    headers = glob.glob(
        os.path.join(str(src_dir), "common", "**", "*.h"),
        recursive=True,
    )
    headers.extend(
        glob.glob(
            os.path.join(str(src_dir), "common", "**", "*.cuh"),
            recursive=True,
        )
    )
    headers = [Path(path) for path in headers]

    # Copy common header files to dst dir
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    for path in headers:
        new_path = dst_dir / path.relative_to(src_dir)
        new_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(path, new_path)


def install_and_import(package):
    """Install a package via pip (if not already installed) and import into globals."""
    main_package = package.split("[")[0]
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    globals()[main_package] = importlib.import_module(main_package)


def uninstall_te_wheel_packages():
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "transformer_engine_cu12",
            "transformer_engine_torch",
            "transformer_engine_jax",
        ]
    )
