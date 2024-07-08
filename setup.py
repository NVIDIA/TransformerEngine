# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import os
from pathlib import Path
from typing import List, Tuple

import setuptools

from build_tools.build_ext import CMakeExtension, get_build_ext
from build_tools.utils import (
    found_cmake,
    found_ninja,
    found_pybind11,
    remove_dups,
    get_frameworks,
    install_and_import,
)
from build_tools.te_version import te_version


frameworks = get_frameworks()
current_file_path = Path(__file__).parent.resolve()


from setuptools.command.build_ext import build_ext as BuildExtension

if "pytorch" in frameworks:
    from torch.utils.cpp_extension import BuildExtension
elif "paddle" in frameworks:
    from paddle.utils.cpp_extension import BuildExtension
elif "jax" in frameworks:
    install_and_import("pybind11")
    from pybind11.setup_helpers import build_ext as BuildExtension


CMakeBuildExtension = get_build_ext(BuildExtension)


def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library"""
    # Project directory root
    root_path = Path(__file__).resolve().parent
    return CMakeExtension(
        name="transformer_engine",
        cmake_path=root_path / Path("transformer_engine/common"),
        cmake_flags=[],
    )


def setup_requirements() -> Tuple[List[str], List[str], List[str]]:
    """Setup Python dependencies

    Returns dependencies for build, runtime, and testing.
    """

    # Common requirements
    setup_reqs: List[str] = []
    install_reqs: List[str] = [
        "pydantic",
        "importlib-metadata>=1.0; python_version<'3.8'",
        "packaging",
    ]
    test_reqs: List[str] = ["pytest>=8.2.1"]

    # Requirements that may be installed outside of Python
    if not found_cmake():
        setup_reqs.append("cmake>=3.18")
    if not found_ninja():
        setup_reqs.append("ninja")
    if not found_pybind11():
        setup_reqs.append("pybind11")

    return [remove_dups(reqs) for reqs in [setup_reqs, install_reqs, test_reqs]]


if __name__ == "__main__":
    # Dependencies
    setup_requires, install_requires, test_requires = setup_requirements()

    __version__ = te_version()

    ext_modules = [setup_common_extension()]
    if not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
        if "pytorch" in frameworks:
            from build_tools.pytorch import setup_pytorch_extension

            ext_modules.append(
                setup_pytorch_extension(
                    "transformer_engine/pytorch/csrc",
                    current_file_path / "transformer_engine" / "pytorch" / "csrc",
                    current_file_path / "transformer_engine",
                )
            )
        if "jax" in frameworks:
            from build_tools.jax import setup_jax_extension

            ext_modules.append(
                setup_jax_extension(
                    "transformer_engine/jax/csrc",
                    current_file_path / "transformer_engine" / "jax" / "csrc",
                    current_file_path / "transformer_engine",
                )
            )
        if "paddle" in frameworks:
            from build_tools.paddle import setup_paddle_extension

            ext_modules.append(
                setup_paddle_extension(
                    "transformer_engine/paddle/csrc",
                    current_file_path / "transformer_engine" / "paddle" / "csrc",
                    current_file_path / "transformer_engine",
                )
            )

    # Configure package
    setuptools.setup(
        name="transformer_engine",
        version=__version__,
        packages=setuptools.find_packages(
            include=[
                "transformer_engine",
                "transformer_engine.*",
                "transformer_engine/build_tools",
            ],
        ),
        extras_require={
            "test": test_requires,
        },
        description="Transformer acceleration library",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        setup_requires=setup_requires,
        install_requires=install_requires,
        license_files=("LICENSE",),
        include_package_data=True,
        package_data={"": ["VERSION.txt"]},
    )
