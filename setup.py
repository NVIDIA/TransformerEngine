# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

from importlib import metadata
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
    get_frameworks,
    remove_dups,
)

frameworks = get_frameworks()
current_file_path = Path(__file__).parent.resolve()


from setuptools.command.build_ext import build_ext as BuildExtension

os.environ["NVTE_PROJECT_BUILDING"] = "1"

if "pytorch" in frameworks:
    from torch.utils.cpp_extension import BuildExtension
elif "jax" in frameworks:
    from pybind11.setup_helpers import build_ext as BuildExtension


CMakeBuildExtension = get_build_ext(BuildExtension)
archs = cuda_archs()


class TimedBdist(bdist_wheel):
    """Helper class to measure build time"""

    def run(self):
        start_time = time.perf_counter()
        super().run()
        total_time = time.perf_counter() - start_time
        print(f"Total time for bdist_wheel: {total_time:.2f} seconds")


def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library"""
    cmake_flags = ["-DCMAKE_CUDA_ARCHITECTURES={}".format(archs)]
    if bool(int(os.getenv("NVTE_UB_WITH_MPI", "0"))):
        assert (
            os.getenv("MPI_HOME") is not None
        ), "MPI_HOME must be set when compiling with NVTE_UB_WITH_MPI=1"
        cmake_flags.append("-DNVTE_UB_WITH_MPI=ON")

    if bool(int(os.getenv("NVTE_ENABLE_NVSHMEM", "0"))):
        assert (
            os.getenv("NVSHMEM_HOME") is not None
        ), "NVSHMEM_HOME must be set when compiling with NVTE_ENABLE_NVSHMEM=1"
        cmake_flags.append("-DNVTE_ENABLE_NVSHMEM=ON")

    if bool(int(os.getenv("NVTE_BUILD_ACTIVATION_WITH_FAST_MATH", "0"))):
        cmake_flags.append("-DNVTE_BUILD_ACTIVATION_WITH_FAST_MATH=ON")

    if bool(int(os.getenv("NVTE_WITH_CUBLASMP", "0"))):
        cmake_flags.append("-DNVTE_WITH_CUBLASMP=ON")
        cublasmp_dir = os.getenv("CUBLASMP_HOME") or metadata.distribution(
            "nvidia-cublasmp-cu12"
        ).locate_file("nvidia/cublasmp/cu12")
        cmake_flags.append(f"-DCUBLASMP_DIR={cublasmp_dir}")
        nvshmem_dir = os.getenv("NVSHMEM_HOME") or metadata.distribution(
            "nvidia-nvshmem-cu12"
        ).locate_file("nvidia/nvshmem")
        cmake_flags.append(f"-DNVSHMEM_DIR={nvshmem_dir}")
        print("CMAKE_FLAGS:", cmake_flags[-2:])

    # Add custom CMake arguments from environment variable
    nvte_cmake_extra_args = os.getenv("NVTE_CMAKE_EXTRA_ARGS")
    if nvte_cmake_extra_args:
        cmake_flags.extend(nvte_cmake_extra_args.split())

    # Project directory root
    root_path = Path(__file__).resolve().parent

    return CMakeExtension(
        name="transformer_engine",
        cmake_path=root_path / Path("transformer_engine/common"),
        cmake_flags=cmake_flags,
    )


def setup_requirements() -> Tuple[List[str], List[str]]:
    """Setup Python dependencies

    Returns dependencies for runtime and testing.
    """

    # Common requirements
    install_reqs: List[str] = [
        "pydantic",
        "importlib-metadata>=1.0",
        "packaging",
    ]
    test_reqs: List[str] = ["pytest>=8.2.1"]

    # Framework-specific requirements
    if not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
        if "pytorch" in frameworks:
            from build_tools.pytorch import install_requirements, test_requirements

            install_reqs.extend(install_requirements())
            test_reqs.extend(test_requirements())
        if "jax" in frameworks:
            from build_tools.jax import install_requirements, test_requirements

            install_reqs.extend(install_requirements())
            test_reqs.extend(test_requirements())

    return [remove_dups(reqs) for reqs in [install_reqs, test_reqs]]


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
        package_data = {}
        include_package_data = False
        install_requires = ([f"transformer_engine_cu12=={__version__}"],)
        extras_require = {
            "pytorch": [f"transformer_engine_torch=={__version__}"],
            "jax": [f"transformer_engine_jax=={__version__}"],
        }
    else:
        install_requires, test_requires = setup_requirements()
        ext_modules = [setup_common_extension()]
        package_data = {"": ["VERSION.txt"]}
        include_package_data = True
        extras_require = {"test": test_requires}

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
        python_requires=">=3.8",
        classifiers=["Programming Language :: Python :: 3"],
        install_requires=install_requires,
        license_files=("LICENSE",),
        include_package_data=include_package_data,
        package_data=package_data,
    )
