# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""

import importlib.util
import os
from pathlib import Path
from importlib import metadata

import setuptools

from .utils import (
    all_files_in_dir,
    cuda_version,
    get_cuda_include_dirs,
    debug_build_enabled,
    setup_mpi_flags,
)
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/PyTorch extensions."""
    return [
        "torch>=2.1",
        "einops",
        "onnxscript",
        "onnx",
        "packaging",
        "pydantic",
        "nvdlfw-inspect",
        "apache-tvm-ffi",
        "nvidia-cutlass-dsl>=4.5.0",
    ]


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

    # apache-tvm-ffi: headers for the C++ API (Module / Function / TensorView)
    # and libtvm_ffi.so for symbol resolution. Used by tvm_ffi_bridge.h /
    # applyTVMFunction. Python registers AOT-compiled CuTeDSL kernels into
    # the global registry; TE C++ looks them up via Function::GetGlobalRequired.
    tvm_ffi_spec = importlib.util.find_spec("tvm_ffi")
    if tvm_ffi_spec is None or not tvm_ffi_spec.submodule_search_locations:
        raise RuntimeError(
            "apache-tvm-ffi package not found; install it (e.g. "
            "`pip install apache-tvm-ffi`) — required for the TVM FFI bridge."
        )
    tvm_ffi_root = Path(tvm_ffi_spec.submodule_search_locations[0])
    tvm_ffi_include = tvm_ffi_root / "include"
    tvm_ffi_lib_dir = tvm_ffi_root / "lib"
    if not tvm_ffi_include.is_dir() or not (tvm_ffi_lib_dir / "libtvm_ffi.so").exists():
        raise RuntimeError(
            f"apache-tvm-ffi assets missing at {tvm_ffi_root} (need include/ and lib/libtvm_ffi.so)"
        )
    include_dirs.append(tvm_ffi_include)

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

    setup_mpi_flags(include_dirs, cxx_flags)

    library_dirs = [tvm_ffi_lib_dir]
    libraries = ["tvm_ffi"]
    # rpath pinned to the pip install dir so the loader finds libtvm_ffi.so
    # without LD_LIBRARY_PATH at runtime.
    extra_link_args = [f"-Wl,-rpath,{tvm_ffi_lib_dir}"]
    if bool(int(os.getenv("NVTE_ENABLE_NVSHMEM", 0))):
        assert (
            os.getenv("NVSHMEM_HOME") is not None
        ), "NVSHMEM_HOME must be set when compiling with NVTE_ENABLE_NVSHMEM=1"
        nvshmem_home = Path(os.getenv("NVSHMEM_HOME"))
        include_dirs.append(nvshmem_home / "include")
        library_dirs.append(nvshmem_home / "lib")
        libraries.append("nvshmem_host")
        cxx_flags.append("-DNVTE_ENABLE_NVSHMEM")

    if bool(int(os.getenv("NVTE_WITH_CUBLASMP", 0))):
        cxx_flags.append("-DNVTE_WITH_CUBLASMP")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CppExtension

    return CppExtension(
        name="transformer_engine_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={"cxx": cxx_flags},
        extra_link_args=extra_link_args,
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
