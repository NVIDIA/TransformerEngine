# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""
import os
from pathlib import Path

import setuptools

from .utils import all_files_in_dir, cuda_version, get_cuda_include_dirs, debug_build_enabled
from typing import List


def _all_files_in_dir_excluding(path, name_extension=None, exclude_dirs=None):
    """Like all_files_in_dir but excludes specified subdirectory names."""
    exclude_dirs = set(exclude_dirs or [])
    all_files = []
    for dirname, dirnames, names in os.walk(path):
        # Filter out excluded directories in-place to prevent os.walk from descending
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for name in names:
            if name_extension is not None and not name.endswith(f".{name_extension}"):
                continue
            all_files.append(Path(dirname, name))
    return all_files


def install_requirements() -> List[str]:
    """Install dependencies for TE/PyTorch extensions."""
    return ["torch>=2.1", "einops", "onnxscript", "onnx", "packaging", "pydantic", "nvdlfw-inspect"]


def test_requirements() -> List[str]:
    """Test dependencies for TE/PyTorch extensions."""
    return [
        "numpy",
        "torchvision",
        "transformers",
        "torchao==0.13",
        "onnxruntime",
        "onnxruntime_extensions",
    ]


def setup_pytorch_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files (exclude stable/ subdirectory, built separately)
    sources = _all_files_in_dir_excluding(
        Path(csrc_source_files), name_extension="cpp", exclude_dirs=["stable"]
    )

    # Header files
    include_dirs = get_cuda_include_dirs()
    include_dirs.extend(
        [
            common_header_files,
            common_header_files / "common",
            common_header_files / "common" / "include",
            csrc_header_files,
        ]
    )

    # Compiler flags
    cxx_flags = ["-O3", "-fvisibility=hidden"]
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")

    # Version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA version")
    else:
        if version < (12, 0):
            raise RuntimeError("Transformer Engine requires CUDA 12.0 or newer")

    if bool(int(os.getenv("NVTE_UB_WITH_MPI", "0"))):
        assert (
            os.getenv("MPI_HOME") is not None
        ), "MPI_HOME=/path/to/mpi must be set when compiling with NVTE_UB_WITH_MPI=1!"
        mpi_path = Path(os.getenv("MPI_HOME"))
        include_dirs.append(mpi_path / "include")
        cxx_flags.append("-DNVTE_UB_WITH_MPI")

    library_dirs = []
    libraries = []
    if bool(int(os.getenv("NVTE_ENABLE_NVSHMEM", 0))):
        assert (
            os.getenv("NVSHMEM_HOME") is not None
        ), "NVSHMEM_HOME must be set when compiling with NVTE_ENABLE_NVSHMEM=1"
        nvshmem_home = Path(os.getenv("NVSHMEM_HOME"))
        include_dirs.append(nvshmem_home / "include")
        library_dirs.append(nvshmem_home / "lib")
        libraries.append("nvshmem_host")
        cxx_flags.append("-DNVTE_ENABLE_NVSHMEM")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CppExtension

    return CppExtension(
        name="transformer_engine_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={"cxx": cxx_flags},
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )


def setup_pytorch_stable_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup stable ABI extension for PyTorch support.

    This extension uses only the PyTorch stable ABI (torch/csrc/stable/),
    producing a binary that is compatible across PyTorch versions.
    It does NOT use CppExtension to avoid pulling in unstable ATen headers.
    """
    import torch

    # Source files from csrc/stable/ directory
    stable_dir = Path(csrc_source_files) / "stable"
    sources = all_files_in_dir(stable_dir, name_extension="cpp")
    if not sources:
        return None

    # Include directories
    include_dirs = get_cuda_include_dirs()
    include_dirs.extend(
        [
            common_header_files,
            common_header_files / "common",
            common_header_files / "common" / "include",
            csrc_header_files,
            # PyTorch headers (for stable ABI only)
            Path(torch.utils.cmake_prefix_path).parent.parent / "include",
        ]
    )

    # Compiler flags
    cxx_flags = ["-O3", "-fvisibility=hidden", "-std=c++17", "-DUSE_CUDA"]
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")

    # Library directories and libraries
    # Find the TE common library (libtransformer_engine.so)
    te_lib_dir = Path(csrc_source_files).parent.parent.parent
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    cuda_lib_dir = os.path.join(cuda_home, "lib64")
    if not os.path.isdir(cuda_lib_dir):
        cuda_lib_dir = os.path.join(cuda_home, "lib")
    library_dirs = [
        str(Path(torch.utils.cmake_prefix_path).parent.parent / "lib"),
        str(te_lib_dir),
        cuda_lib_dir,
    ]
    libraries = ["torch", "torch_cpu", "c10", "cudart", "transformer_engine"]

    # Set rpath so the stable extension can find libtransformer_engine.so at runtime.
    # Use $ORIGIN for co-located libraries plus the absolute path for editable installs.
    extra_link_args = [
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{te_lib_dir.resolve()}",
    ]

    return setuptools.Extension(
        name="te_stable_abi",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args=cxx_flags,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_link_args=extra_link_args,
    )
