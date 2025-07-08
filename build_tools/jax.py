# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX related extensions."""
import os
from pathlib import Path
from packaging import version

import setuptools

from .utils import get_cuda_include_dirs, all_files_in_dir, debug_build_enabled
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/JAX extensions."""
    return ["jax", "flax>=0.7.1"]


def test_requirements() -> List[str]:
    """Test dependencies for TE/JAX extensions."""
    return ["numpy"]


def xla_path() -> str:
    """XLA root path lookup.
    Throws FileNotFoundError if XLA source is not found."""

    try:
        import jax

        if version.parse(jax.__version__) >= version.parse("0.5.0"):
            from jax import ffi  # pylint: disable=ungrouped-imports
        else:
            from jax.extend import ffi  # pylint: disable=ungrouped-imports

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
    sources = all_files_in_dir(extensions_dir, name_extension="cpp")

    # Header files
    include_dirs = get_cuda_include_dirs()
    include_dirs.extend(
        [
            common_header_files,
            common_header_files / "common",
            common_header_files / "common" / "include",
            csrc_header_files,
            xla_path(),
        ]
    )

    # Compile flags
    cxx_flags = ["-O3"]
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")

    # Define TE/JAX as a Pybind11Extension
    from pybind11.setup_helpers import Pybind11Extension

    return Pybind11Extension(
        "transformer_engine_jax",
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        extra_compile_args=cxx_flags,
    )
