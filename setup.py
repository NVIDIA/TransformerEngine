# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import os
import time
from pathlib import Path
from typing import List, Tuple

import setuptools
from wheel.bdist_wheel import bdist_wheel

from build_tools.build_ext import CMakeExtension, get_build_ext
from build_tools.te_version import te_version
from build_tools.utils import (
    cuda_archs,
    found_cmake,
    found_ninja,
    found_pybind11,
    get_frameworks,
    install_and_import,
    remove_dups,
    uninstall_te_wheel_packages,
)

frameworks = get_frameworks()
current_file_path = Path(__file__).parent.resolve()


from setuptools.command.build_ext import build_ext as BuildExtension

os.environ["NVTE_PROJECT_BUILDING"] = "1"

if "pytorch" in frameworks:
    from torch.utils.cpp_extension import BuildExtension
elif "paddle" in frameworks:
    from paddle.utils.cpp_extension import BuildExtension
elif "jax" in frameworks:
    install_and_import("pybind11[global]")
    from pybind11.setup_helpers import build_ext as BuildExtension


CMakeBuildExtension = get_build_ext(BuildExtension)


class TimedBdist(bdist_wheel):
    """Helper class to measure build time"""

    def run(self):
        start_time = time.perf_counter()
        super().run()
        total_time = time.perf_counter() - start_time
        print(f"Total time for bdist_wheel: {total_time:.2f} seconds")


def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library"""
    # Project directory root
    root_path = Path(__file__).resolve().parent

    return CMakeExtension(
        name="transformer_engine",
        cmake_path=root_path / Path("transformer_engine/common"),
        cmake_flags=["-DCMAKE_CUDA_ARCHITECTURES={}".format(cuda_archs())],
    )


def setup_requirements() -> Tuple[List[str], List[str], List[str]]:
    """Setup Python dependencies

    Returns dependencies for build, runtime, and testing.
    """

    # Common requirements
    setup_reqs: List[str] = []
    install_reqs: List[str] = [
        "pydantic",
        "importlib-metadata>=1.0",
        "packaging",
    ]
    test_reqs: List[str] = ["pytest>=8.2.1"]

    # Requirements that may be installed outside of Python
    if not found_cmake():
        setup_reqs.append("cmake>=3.21")
    if not found_ninja():
        setup_reqs.append("ninja")
    if not found_pybind11():
        setup_reqs.append("pybind11")

    # Framework-specific requirements
    if not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
        if "pytorch" in frameworks:
            install_reqs.extend(["torch"])
            test_reqs.extend(["numpy", "onnxruntime", "torchvision", "prettytable"])
        if "jax" in frameworks:
            install_reqs.extend(["jax", "flax>=0.7.1"])
            test_reqs.extend(["numpy", "praxis"])
        if "paddle" in frameworks:
            install_reqs.append("paddlepaddle-gpu")
            test_reqs.append("numpy")

    return [remove_dups(reqs) for reqs in [setup_reqs, install_reqs, test_reqs]]


if __name__ == "__main__":
    __version__ = te_version()

    with open("README.rst", encoding="utf-8") as f:
        long_description = f.read()

    # Settings for building top level empty package for dependency management.
    if bool(int(os.getenv("NVTE_BUILD_METAPACKAGE", "0"))):
        assert bool(
            int(os.getenv("NVTE_RELEASE_BUILD", "0"))
        ), "NVTE_RELEASE_BUILD env must be set for metapackage build."
        ext_modules = []
        cmdclass = {}
        package_data = {}
        include_package_data = False
        setup_requires = []
        install_requires = ([f"transformer_engine_cu12=={__version__}"],)
        extras_require = {
            "pytorch": [f"transformer_engine_torch=={__version__}"],
            "jax": [f"transformer_engine_jax=={__version__}"],
            "paddle": [f"transformer_engine_paddle=={__version__}"],
        }
    else:
        setup_requires, install_requires, test_requires = setup_requirements()
        ext_modules = [setup_common_extension()]
        cmdclass = {"build_ext": CMakeBuildExtension, "bdist_wheel": TimedBdist}
        package_data = {"": ["VERSION.txt"]}
        include_package_data = True
        extras_require = {"test": test_requires}

        if not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
            # Remove residual FW packages since compiling from source
            # results in a single binary with FW extensions included.
            uninstall_te_wheel_packages()
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
        extras_require=extras_require,
        description="Transformer acceleration library",
        long_description=long_description,
        long_description_content_type="text/x-rst",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension, "bdist_wheel": TimedBdist},
        python_requires=">=3.8, <3.13",
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        setup_requires=setup_requires,
        install_requires=install_requires,
        license_files=("LICENSE",),
        include_package_data=include_package_data,
        package_data=package_data,
    )
