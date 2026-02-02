# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    """Test dependencies for TE/JAX extensions.

    Triton Package Selection:
        The triton package is selected based on NVTE_USE_PYTORCH_TRITON environment variable:

        Default (NVTE_USE_PYTORCH_TRITON unset or "0"):
            Returns 'triton' - OpenAI's standard package from PyPI.
            Install with: pip install triton

        NVTE_USE_PYTORCH_TRITON=1:
            Returns 'pytorch-triton' - for mixed JAX+PyTorch environments.
            Install with: pip install pytorch-triton --index-url https://download.pytorch.org/whl/cu121

            Note: Do NOT install pytorch-triton from PyPI directly - that's a placeholder.
    """
    use_pytorch_triton = bool(int(os.environ.get("NVTE_USE_PYTORCH_TRITON", "0")))

    triton_package = "pytorch-triton" if use_pytorch_triton else "triton"

    return [
        "numpy",
        triton_package,
    ]


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
    print("Includ dirs for JAX extension:", include_dirs)

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
        libraries=["nccl"],
    )

_compiled = False

def compile_extension():
    import os
    import shutil

    global _compiled
    if _compiled:
        return

    base_dir = Path(os.path.dirname(__file__)).parent.parent.parent
    te_jax_build_dir = base_dir / "build" / "te_jax"
    # if os.path.exists(te_jax_build_dir):
    #     shutil.rmtree(te_jax_build_dir)

    ext = setup_jax_extension(
        Path(__file__).resolve().parent.parent / "csrc",
        Path(__file__).resolve().parent.parent / "csrc",
        Path(__file__).resolve().parent.parent.parent,
    )
    from pybind11.setup_helpers import build_ext as BuildExtension
    from setuptools import Distribution
    import subprocess

    dist = Distribution()
    dist.ext_modules = [ext]
    cmd = BuildExtension(dist)
    cmd.initialize_options()
    cmd.parallel = os.cpu_count()  # Enable parallel compilation
    cmd.finalize_options()
    cmd.build_temp = os.path.join(te_jax_build_dir, "temp")
    cmd.build_lib = os.path.join(te_jax_build_dir, "lib")
    os.makedirs(cmd.build_temp, exist_ok=True)
    os.makedirs(cmd.build_lib, exist_ok=True)
    cmd.run()

    subprocess.call([
        "cp",
        os.path.join(cmd.build_lib, "transformer_engine_jax" + cmd.get_ext_filename(fullname="")),
        base_dir,
    ])

    _compiled = True