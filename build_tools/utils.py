# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import os
import re
import glob
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Tuple


@cache
def debug_build_enabled() -> bool:
    """Whether to build with a debug configuration"""
    for arg in sys.argv:
        if arg == "--debug":
            sys.argv.remove(arg)
            return True
    if int(os.getenv("NVTE_BUILD_DEBUG", "0")):
        return True
    return False


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


@cache
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
    supported_frameworks = ["pytorch", "jax", "paddle"]

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
        try:
            import paddle
        except ImportError:
            pass
        else:
            _frameworks.append("paddle")

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


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        path = Path(path)
        for filename in filenames:
            paths.append(str(path / filename).replace(f"{directory}/", ""))
    return paths


def copy_common_headers(te_src, dst):
    headers = te_src / "common"
    for file_path in glob.glob(os.path.join(str(headers), "**", "*.h"), recursive=True):
        new_path = os.path.join(dst, file_path[len(str(te_src)) + 1 :])
        Path(new_path).parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(file_path, new_path)


def install_and_import(package):
    """Install a package via pip (if not already installed) and import into globals."""
    import importlib

    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = importlib.import_module(package)
