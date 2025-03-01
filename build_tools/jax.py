# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX related extensions."""
import os
from pathlib import Path

import setuptools
from glob import glob

from .utils import cuda_path, all_files_in_dir
from typing import List


def xla_path() -> str:
    """XLA root path lookup.
    Throws FileNotFoundError if XLA source is not found."""

    try:
        from jax.extend import ffi
    except ImportError:
        if os.getenv("XLA_HOME"):
            xla_home = Path(os.getenv("XLA_HOME"))
        else:
            xla_home = "/opt/xla"
    else:
        xla_home = ffi.include_dir()

    if not os.path.isdir(xla_home):
        raise FileNotFoundError("Could not find xla source.")
    return xla_home


def setup_jax_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup PyBind11 extension for JAX support"""
    # Source files
    csrc_source_files = Path(csrc_source_files)
    extensions_dir = csrc_source_files / "extensions"
    sources = [
        csrc_source_files / "utils.cu",
    ] + all_files_in_dir(extensions_dir, ".cpp")

    # Header files
    cuda_home, _ = cuda_path()
    xla_home = xla_path()
    include_dirs = [
        cuda_home / "include",
        common_header_files,
        common_header_files / "common",
        common_header_files / "common" / "include",
        csrc_header_files,
        xla_home,
    ]

    # Compile flags
    cxx_flags = ["-O3"]
    nvcc_flags = ["-O3"]

    # Define TE/JAX as a Pybind11Extension
    from pybind11.setup_helpers import Pybind11Extension

    class Pybind11CUDAExtension(Pybind11Extension):
        """Modified Pybind11Extension to allow combined CXX + NVCC compile flags."""

        def _add_cflags(self, flags: List[str]) -> None:
            if isinstance(self.extra_compile_args, dict):
                cxx_flags = self.extra_compile_args.pop("cxx", [])
                cxx_flags += flags
                self.extra_compile_args["cxx"] = cxx_flags
            else:
                self.extra_compile_args[:0] = flags

    return Pybind11CUDAExtension(
        "transformer_engine_jax",
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )
