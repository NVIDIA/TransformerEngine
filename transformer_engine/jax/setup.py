# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script for Transformer Engine JAX extensions.

This module handles the build and installation of the JAX-specific components
of Transformer Engine. It manages:
- JAX extension compilation with pybind11
- Common header file management
- Build tool dependencies
- Package metadata and dependencies

The script supports both development and release builds, with different
behaviors for:
- Build tool management
- Header file copying
- Extension compilation
- Package distribution
"""

# pylint: disable=wrong-import-position,wrong-import-order

import sys
import os
import shutil
from pathlib import Path

import setuptools

try:
    import jax  # pylint: disable=unused-import
except ImportError as e:
    raise RuntimeError("This package needs Jax to build.") from e


current_file_path = Path(__file__).parent.resolve()
build_tools_dir = current_file_path.parent.parent / "build_tools"
if bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))) or os.path.isdir(build_tools_dir):
    build_tools_copy = current_file_path / "build_tools"
    if build_tools_copy.exists():
        shutil.rmtree(build_tools_copy)
    shutil.copytree(build_tools_dir, build_tools_copy)


from build_tools.build_ext import get_build_ext
from build_tools.utils import copy_common_headers, install_and_import, cuda_toolkit_include_path
from build_tools.te_version import te_version
from build_tools.jax import setup_jax_extension

install_and_import("pybind11")
from pybind11.setup_helpers import build_ext as BuildExtension

os.environ["NVTE_PROJECT_BUILDING"] = "1"
CMakeBuildExtension = get_build_ext(BuildExtension, True)


if __name__ == "__main__":
    """Main entry point for JAX extension installation.

    This section handles:
    1. Common header file management
       - Creates a temporary directory for common headers
       - Copies necessary header files from the common library

    2. Extension module setup
       - Configures the JAX-specific C++ extension
       - Sets up build paths and dependencies

    3. Package configuration
       - Sets package metadata
       - Configures build and install requirements
       - Sets up extension modules

    4. Cleanup
       - Removes temporary directories after build
       - Cleans up build tools if not in release mode

    Environment variables:
    - NVTE_RELEASE_BUILD: Controls release build behavior
    - NVTE_PROJECT_BUILDING: Set to "1" during build

    Note:
        The script requires JAX to be installed for building.
        It will raise a RuntimeError if JAX is not available.
    """

    # Extensions
    common_headers_dir = "common_headers"
    copy_common_headers(current_file_path.parent, str(current_file_path / common_headers_dir))
    ext_modules = [
        setup_jax_extension(
            "csrc", current_file_path / "csrc", current_file_path / common_headers_dir
        )
    ]

    setup_requires = ["jax[cuda12]", "flax>=0.7.1"]
    if cuda_toolkit_include_path() is None:
        setup_requires.extend(
            [
                "nvidia-cuda-runtime-cu12",
                "nvidia-cublas-cu12",
                "nvidia-cudnn-cu12",
                "nvidia-cuda-cccl-cu12",
                "nvidia-cuda-nvcc-cu12",
                "nvidia-nvtx-cu12",
                "nvidia-cuda-nvrtc-cu12",
            ]
        )

    # Configure package
    setuptools.setup(
        name="transformer_engine_jax",
        version=te_version(),
        description="Transformer acceleration library - Jax Lib",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        setup_requires=setup_requires,
        install_requires=["jax", "flax>=0.7.1"],
        tests_require=["numpy"],
    )
    if any(x in sys.argv for x in (".", "sdist", "bdist_wheel")):
        shutil.rmtree(common_headers_dir)
        shutil.rmtree("build_tools")
