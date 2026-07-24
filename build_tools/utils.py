# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import platform
from pathlib import Path
from importlib.metadata import PackageNotFoundError, distribution, version as get_version
from subprocess import CalledProcessError
from typing import Callable, List, Optional, Tuple, Union


# Needs to stay consistent with .pre-commit-config.yaml config.
def min_python_version() -> Tuple[int]:
    """Minimum supported Python version."""
    return (3, 10, 0)


def min_python_version_str() -> str:
    """String representing minimum supported Python version."""
    return ".".join(map(str, min_python_version()))


if sys.version_info < min_python_version():
    raise RuntimeError(
        f"Transformer Engine requires Python {min_python_version_str()} or newer, "
        f"but found Python {platform.python_version()}."
    )


@functools.lru_cache(maxsize=None)
def debug_build_enabled() -> bool:
    """Whether to build with a debug configuration"""
    return bool(int(os.getenv("NVTE_BUILD_DEBUG", "0")))


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
            if name_extension is not None and not name.endswith(f".{name_extension}"):
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
def cuda_home_path() -> Optional[Path]:
    """Returns the CUDA home path. This path should contain binaries (e.g. nvcc), headers, and libraries.

    Returns `None` if CUDA is not found."""
    if cuda_home := os.getenv("CUDA_HOME"):
        return Path(cuda_home)

    # Check if site-packages contains an `nvidia` directory
    for site_package in sys.path:
        if not Path(site_package).is_dir():
            continue

        nvidia_dir = Path(site_package) / "nvidia"
        if not nvidia_dir.is_dir():
            continue

        if Path(nvidia_dir / "bin").is_dir():
            return nvidia_dir

        # In this case there must be a "CUDA version directory" like `cu12` or `cu13`
        # that contains the binaries, headers, and libraries
        for cuda_version_dir in nvidia_dir.iterdir():
            if not cuda_version_dir.is_dir():
                continue

            # Verify that the directory name matches the expected `cu##` format
            if not re.match(r"cu\d+", cuda_version_dir.name):
                continue

            if not (cuda_version_dir / "bin").is_dir():
                continue

            return cuda_version_dir

    return None


@functools.lru_cache(maxsize=None)
def nccl_root_path() -> Optional[Path]:
    """Return the NCCL installation root.

    Returns `None` if NCCL is not found."""
    if (cuda_home := cuda_home_path()) is not None:
        nccl_root = cuda_home.parent / "nccl"
        if nccl_root.is_dir():
            return nccl_root

    # Check NCCL Python packages
    for package_name in ["nvidia-nccl-cu13", "nvidia-nccl-cu12"]:
        try:
            nccl_distribution = distribution(package_name)
        except PackageNotFoundError:
            continue

        nccl_root = Path(nccl_distribution.locate_file("nvidia/nccl"))
        if nccl_root.is_dir():
            return nccl_root

    return None


@functools.lru_cache(maxsize=None)
def nccl_include_path() -> Optional[Path]:
    """Return the NCCL include directory."""
    if (nccl_root := nccl_root_path()) is not None:
        include_path = nccl_root / "include"
        if include_path.is_dir():
            return include_path

    return None


@functools.lru_cache(maxsize=None)
def nccl_lib_path() -> Optional[Path]:
    """Return the NCCL shared library path."""
    if (nccl_root := nccl_root_path()) is not None:
        lib_path = nccl_root / "lib" / "libnccl.so.2"
        if lib_path.is_file():
            return lib_path

    return None


@functools.lru_cache(maxsize=None)
def cuda_toolkit_include_path() -> Tuple[str, str]:
    """Returns root path for cuda toolkit includes.

    return `None` if CUDA is not found."""
    # Try finding CUDA
    cuda_home: Optional[Path] = None
    if cuda_home is None and os.getenv("CUDA_HOME"):
        # Check in CUDA_HOME
        cuda_home = Path(os.getenv("CUDA_HOME")) / "include"
    if cuda_home is None:
        # Check in NVCC
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin is not None:
            cuda_home = Path(nvcc_bin.rstrip("/bin/nvcc")) / "include"
    if cuda_home is None:
        # Last-ditch guess in /usr/local/cuda
        if Path("/usr/local/cuda").is_dir():
            cuda_home = Path("/usr/local/cuda") / "include"
    return cuda_home


@functools.lru_cache(maxsize=None)
def nvcc_path() -> Optional[Path]:
    """Get the NVCC binary path.

    Returns `None` if NVCC is not found.
    """

    def lookup_via_cuda_home() -> Optional[str]:
        if (cuda_home := cuda_home_path()) is not None:
            return cuda_home / "bin" / "nvcc"
        return None

    def lookup_via_path() -> Optional[str]:
        if (nvcc_bin := shutil.which("nvcc")) is not None:
            return nvcc_bin
        return None

    def lookup_via_distribution() -> Optional[str]:
        try:
            return distribution("nvidia-cuda-nvcc").locate_file("bin/nvcc")
        except PackageNotFoundError:
            return None

    def lookup_via_local_cuda() -> Optional[str]:
        return "/usr/local/cuda/bin/nvcc"

    nvcc_lookup_funcs: List[Callable[[], Optional[str]]] = [
        lookup_via_cuda_home,
        lookup_via_path,
        lookup_via_distribution,
        lookup_via_local_cuda,
    ]

    for nvcc_lookup_func in nvcc_lookup_funcs:
        nvcc_bin = nvcc_lookup_func()

        if nvcc_bin is None:
            continue

        nvcc_bin_path = Path(nvcc_bin)
        if nvcc_bin_path.is_file():
            return nvcc_bin_path

    return None


@functools.lru_cache(maxsize=None)
def get_cuda_include_dirs() -> Tuple[str, str]:
    """Returns the CUDA header directory."""

    force_wheels = bool(int(os.getenv("NVTE_BUILD_USE_NVIDIA_WHEELS", "0")))
    # If cuda is installed via toolkit, all necessary headers
    # are bundled inside the top level cuda directory.
    if not force_wheels and cuda_toolkit_include_path() is not None:
        return [cuda_toolkit_include_path()]

    # Use pip wheels to include all headers.
    try:
        import nvidia
    except ModuleNotFoundError as e:
        raise RuntimeError("CUDA not found.")

    if nvidia.__file__ is not None:
        cuda_root = Path(nvidia.__file__).parent
    else:
        cuda_root = Path(nvidia.__path__[0])  # namespace
    return [
        subdir / "include"
        for subdir in cuda_root.iterdir()
        if subdir.is_dir() and (subdir / "include").is_dir()
    ]


@functools.lru_cache(maxsize=None)
def cudnn_frontend_include_path() -> Path:
    """Return the C++ include directory from nvidia-cudnn-frontend."""
    package = "nvidia-cudnn-frontend"
    try:
        include_dir = Path(distribution(package).locate_file("include")).resolve()
    except PackageNotFoundError as e:
        raise RuntimeError(
            f"{package} is required to build Transformer Engine. "
            f"Install it with `pip install {package}`."
        ) from e

    header = include_dir / "cudnn_frontend.h"
    if not header.is_file():
        raise RuntimeError(
            f"The {package} installation does not contain the expected header {header}."
        )
    return include_dir


@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    archs = os.getenv("NVTE_CUDA_ARCHS")
    if archs is None:
        version = cuda_version()
        if version >= (13, 0):
            archs = "75;80;89;90;100;120"
        elif version >= (12, 8):
            archs = "70;80;89;90;100;120"
        else:
            archs = "70;80;89;90"
    return archs


def nccl_ep_enabled(archs: str = None) -> bool:
    """Return True when NCCL EP should be compiled into this build.

    Reads NVTE_WITH_NCCL_EP (default on). Auto-skips with a printed warning
    when no arch >= 90 is targeted; raises RuntimeError if the flag was
    explicitly set to 1 but no qualifying arch is present. Mirrors the same
    logic in both TE/Common (setup.py) and TE/JAX (build_tools/jax.py) so a
    single env var controls both sides consistently.
    """
    if archs is None:
        archs = cuda_archs()
    nccl_ep_env = os.getenv("NVTE_WITH_NCCL_EP")
    nccl_ep_explicit = nccl_ep_env is not None
    build_ep = bool(int(nccl_ep_env if nccl_ep_explicit else "1"))
    if build_ep:
        arch_tokens = [a.strip() for a in str(archs or "").split(";") if a.strip()]
        has_hopper_or_newer = any(
            t.lower() == "native" or (t.rstrip("af").isdigit() and int(t.rstrip("af")) >= 90)
            for t in arch_tokens
        )
        if not has_hopper_or_newer:
            if nccl_ep_explicit:
                raise RuntimeError(
                    f"NVTE_WITH_NCCL_EP=1 was set but NVTE_CUDA_ARCHS ('{archs}') "
                    "contains no arch >= 90. NCCL EP requires Hopper or newer."
                )
            print(f"[NCCL EP] No arch >= 90 in NVTE_CUDA_ARCHS ('{archs}'); skipping build.")
            build_ep = False
    return build_ep


def cuda_version() -> Tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) tuple.

    Try to get cuda version by locating the nvcc executable and running nvcc --version. If
    nvcc is not found, look for the cuda runtime package pip `nvidia-cuda-runtime-cu12`
    and check pip version.
    """

    if (nvcc_bin := nvcc_path()) is not None:
        output = subprocess.run(
            [str(nvcc_bin), "-V"],
            capture_output=True,
            check=True,
            universal_newlines=True,
        )
        match = re.search(r"release\s*([\d.]+)", output.stdout)
        version = match.group(1).split(".")
        return tuple(int(v) for v in version)

    version_str: Optional[str] = None
    package_names = ["nvidia-cuda-runtime", "nvidia-cuda-runtime-cu13", "nvidia-cuda-runtime-cu12"]

    for package_name in package_names:
        try:
            version_str = get_version(package_name)
        except PackageNotFoundError:
            pass
        else:
            return tuple(int(part) for part in version_str.split(".") if part.isdigit())

    raise RuntimeError(
        f"Could neither find NVCC executable nor CUDA runtime Python package for {package_names}."
    )


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


def setup_mpi_flags(include_dirs: List, cxx_flags: List) -> None:
    """Add MPI include path and compile definition if NVTE_UB_WITH_MPI is enabled."""
    if bool(int(os.getenv("NVTE_UB_WITH_MPI", "0"))):
        assert (
            os.getenv("MPI_HOME") is not None
        ), "MPI_HOME=/path/to/mpi must be set when compiling with NVTE_UB_WITH_MPI=1!"
        mpi_path = Path(os.getenv("MPI_HOME"))
        include_dirs.append(mpi_path / "include")
        cxx_flags.append("-DNVTE_UB_WITH_MPI")


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
