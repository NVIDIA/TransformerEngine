# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""
import os
from pathlib import Path

import setuptools

from .utils import all_files_in_dir, cuda_version, get_cuda_include_dirs, debug_build_enabled
from typing import List


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

    # Source files
    sources = all_files_in_dir(Path(csrc_source_files), name_extension="cpp")
    build_native_cp_transport = bool(int(os.getenv("NVTE_WITH_NCCL_DEVICE_CP", "0")))
    if build_native_cp_transport:
        sources.extend(
            path
            for path in all_files_in_dir(Path(csrc_source_files), name_extension="cu")
            if path.name == "cp_native_transport.cu"
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

    extra_compile_args = {"cxx": cxx_flags}
    if build_native_cp_transport:
        nvcc_flags = ["-O3", "-std=c++17"]
        nccl_home = os.getenv("NCCL_HOME")
        nccl_include_dir = os.getenv("NVTE_NCCL_INCLUDE_DIR")
        nccl_library_dir = os.getenv("NVTE_NCCL_LIBRARY_DIR")
        if nccl_home:
            nccl_home = Path(nccl_home)
            nccl_include_dir = nccl_include_dir or str(nccl_home / "include")
            nccl_library_dir = nccl_library_dir or str(nccl_home / "lib")
        if nccl_include_dir:
            include_dirs.append(Path(nccl_include_dir))
        if nccl_library_dir:
            library_dirs.append(Path(nccl_library_dir))
        libraries.append("nccl")
        cxx_flags.append("-DNVTE_WITH_NCCL_DEVICE_CP")
        nvcc_flags.append("-DNVTE_WITH_NCCL_DEVICE_CP")
        extra_compile_args["nvcc"] = nvcc_flags

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    extension_cls = CUDAExtension if build_native_cp_transport else CppExtension

    return extension_cls(
        name="transformer_engine_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args=extra_compile_args,
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
