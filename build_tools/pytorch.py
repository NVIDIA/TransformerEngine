# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""
import os
from pathlib import Path

import setuptools

from .utils import all_files_in_dir


def setup_pytorch_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup MUSA extension for PyTorch support"""

    # Source files
    csrc_source_files = Path(csrc_source_files)
    extensions_dir = csrc_source_files / "extensions"
    sources = [
        csrc_source_files / "common.cpp",
        csrc_source_files / "type_converters.cpp",
        csrc_source_files / "quantizer.cpp",
    ] + all_files_in_dir(extensions_dir)

    import torch, torch_musa

    torch_musa_dir = Path(torch_musa.__file__).parent

    # Header files
    include_dirs = [
        torch_musa_dir / "share" / "torch_musa_codegen",
        "/home/torch_musa",  # some *.muh not installed!
        common_header_files,
        common_header_files / "common",
        common_header_files / "common" / "include",
        csrc_header_files,
    ]

    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
        "-std=c++17",
        "-Wno-reorder",
        "-march=native",
        "force_mcc",
    ]
    mcc_flags = [
        "-O3",
        "-march=native",
    ]

    # Libraries
    library_dirs = [torch_musa_dir / "lib"]
    libraries = [
        "musa_kernels",
        "musa_python",
    ]

    if bool(int(os.getenv("NVTE_UB_WITH_MPI", "0"))):
        assert (
            os.getenv("MPI_HOME") is not None
        ), "MPI_HOME=/path/to/mpi must be set when compiling with NVTE_UB_WITH_MPI=1!"
        mpi_path = Path(os.getenv("MPI_HOME"))
        include_dirs.append(mpi_path / "include")
        cxx_flags.append("-DNVTE_UB_WITH_MPI")
        mcc_flags.append("-DNVTE_UB_WITH_MPI")
        library_dirs.append(mpi_path / "lib")
        libraries.append("mpi")

    cxx_flags.append("-DNVTE_SKIP_MUSA_UNCOMPATIBLE")
    mcc_flags.append("-DNVTE_SKIP_MUSA_UNCOMPATIBLE")

    cxx_flags.append("-DNVTE_USE_MUSA=1")
    mcc_flags.append("-DNVTE_USE_MUSA=1")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch_musa.utils.musa_extension import MUSAExtension

    return MUSAExtension(
        name="transformer_engine_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={
            "cxx": cxx_flags,
            "mcc": mcc_flags,
        },
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
