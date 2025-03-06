# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script for TE jax extensions."""

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
from build_tools.utils import copy_common_headers, install_and_import
from build_tools.te_version import te_version
from build_tools.jax import setup_jax_extension

install_and_import("pybind11")
from pybind11.setup_helpers import build_ext as BuildExtension

os.environ["NVTE_PROJECT_BUILDING"] = "1"
CMakeBuildExtension = get_build_ext(BuildExtension, True)


if __name__ == "__main__":
    # Extensions
    common_headers_dir = "common_headers"
    copy_common_headers(current_file_path.parent, str(current_file_path / common_headers_dir))
    ext_modules = [
        setup_jax_extension(
            "csrc", current_file_path / "csrc", current_file_path / common_headers_dir
        )
    ]

    # Configure package
    setuptools.setup(
        name="transformer_engine_jax",
        version=te_version(),
        description="Transformer acceleration library - Jax Lib",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        install_requires=["jax", "flax>=0.7.1"],
        tests_require=["numpy", "praxis"],
    )
    if any(x in sys.argv for x in (".", "sdist", "bdist_wheel")):
        shutil.rmtree(common_headers_dir)
        shutil.rmtree("build_tools")
