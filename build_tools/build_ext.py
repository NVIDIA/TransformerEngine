# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import ctypes
import os
import subprocess
import sys
import sysconfig
import copy
import time

from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Type

import setuptools

from .utils import (
    cmake_bin,
    debug_build_enabled,
    found_ninja,
    get_frameworks,
    cuda_path,
    get_max_jobs_for_parallel_build,
)


class CMakeExtension(setuptools.Extension):
    """CMake extension module"""

    def __init__(
        self,
        name: str,
        cmake_path: Path,
        cmake_flags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name, sources=[])  # No work for base class
        self.cmake_path: Path = cmake_path
        self.cmake_flags: List[str] = [] if cmake_flags is None else cmake_flags

    def _build_cmake(self, build_dir: Path, install_dir: Path) -> None:
        # Make sure paths are str
        _cmake_bin = str(cmake_bin())
        cmake_path = str(self.cmake_path)
        build_dir = str(build_dir)
        install_dir = str(install_dir)

        # CMake configure command
        build_type = "Debug" if debug_build_enabled() else "Release"
        configure_command = [
            _cmake_bin,
            "-S",
            cmake_path,
            "-B",
            build_dir,
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        configure_command += self.cmake_flags

        import pybind11

        pybind11_dir = Path(pybind11.__file__).resolve().parent
        pybind11_dir = pybind11_dir / "share" / "cmake" / "pybind11"
        configure_command.append(f"-Dpybind11_DIR={pybind11_dir}")

        # CMake build and install commands
        build_command = [_cmake_bin, "--build", build_dir, "--verbose"]
        install_command = [_cmake_bin, "--install", build_dir, "--verbose"]

        # Check whether parallel build is restricted
        max_jobs = get_max_jobs_for_parallel_build()
        if found_ninja():
            configure_command.append("-GNinja")
        build_command.append("--parallel")
        if max_jobs > 0:
            build_command.append(str(max_jobs))

        # Run CMake commands
        start_time = time.perf_counter()
        for command in [configure_command, build_command, install_command]:
            print(f"Running command {' '.join(command)}")
            try:
                subprocess.run(command, cwd=build_dir, check=True)
            except (CalledProcessError, OSError) as e:
                raise RuntimeError(f"Error when running CMake: {e}")

        total_time = time.perf_counter() - start_time
        print(f"Time for build_ext: {total_time:.2f} seconds")


def get_build_ext(extension_cls: Type[setuptools.Extension], install_so_in_wheel_lib: bool = False):
    class _CMakeBuildExtension(extension_cls):
        """Setuptools command with support for CMake extension modules"""

        def run(self) -> None:
            # Build CMake extensions
            for ext in self.extensions:
                package_path = Path(self.get_ext_fullpath(ext.name))
                install_dir = package_path.resolve().parent
                if isinstance(ext, CMakeExtension):
                    print(f"Building CMake extension {ext.name}")
                    # Set up incremental builds for CMake extensions
                    build_dir = os.getenv("NVTE_CMAKE_BUILD_DIR")
                    if build_dir:
                        build_dir = Path(build_dir).resolve()
                    else:
                        root_dir = Path(__file__).resolve().parent.parent
                        build_dir = root_dir / "build" / "cmake"

                    # Ensure the directory exists
                    build_dir.mkdir(parents=True, exist_ok=True)

                    ext._build_cmake(
                        build_dir=build_dir,
                        install_dir=install_dir,
                    )

            # Build non-CMake extensions as usual
            all_extensions = self.extensions
            self.extensions = [
                ext for ext in self.extensions if not isinstance(ext, CMakeExtension)
            ]
            super().run()
            self.extensions = all_extensions

            # Ensure that binaries are not in global package space.
            lib_dir = (
                "wheel_lib"
                if bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))) or install_so_in_wheel_lib
                else ""
            )
            target_dir = install_dir / "transformer_engine" / lib_dir
            target_dir.mkdir(exist_ok=True, parents=True)

            for ext in Path(self.build_lib).glob("*.so"):
                self.copy_file(ext, target_dir)
                os.remove(ext)

        def build_extensions(self):
            # BuildExtensions from PyTorch already handle CUDA files correctly
            # so we don't need to modify their compiler. Only the pybind11 build_ext needs to be fixed.
            if "pytorch" not in get_frameworks():
                # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
                # extra_compile_args is a dict.
                for ext in self.extensions:
                    if isinstance(ext.extra_compile_args, dict):
                        for target in ["cxx", "nvcc"]:
                            if target not in ext.extra_compile_args.keys():
                                ext.extra_compile_args[target] = []

                # Define new _compile method that redirects to NVCC for .cu and .cuh files.
                original_compile_fn = self.compiler._compile
                self.compiler.src_extensions += [".cu", ".cuh"]

                def _compile_fn(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
                    # Copy before we make any modifications.
                    cflags = copy.deepcopy(extra_postargs)
                    original_compiler = self.compiler.compiler_so
                    try:
                        _, nvcc_bin = cuda_path()
                        original_compiler = self.compiler.compiler_so

                        if os.path.splitext(src)[1] in [".cu", ".cuh"]:
                            self.compiler.set_executable("compiler_so", str(nvcc_bin))
                            if isinstance(cflags, dict):
                                cflags = cflags["nvcc"]

                            # Add -fPIC if not already specified
                            if not any("-fPIC" in flag for flag in cflags):
                                cflags.extend(["--compiler-options", "'-fPIC'"])

                            # Forward unknown options
                            if not any("--forward-unknown-opts" in flag for flag in cflags):
                                cflags.append("--forward-unknown-opts")

                        elif isinstance(cflags, dict):
                            cflags = cflags["cxx"]

                        # Append -std=c++17 if not already in flags
                        if not any(flag.startswith("-std=") for flag in cflags):
                            cflags.append("-std=c++17")

                        return original_compile_fn(obj, src, ext, cc_args, cflags, pp_opts)

                    finally:
                        # Put the original compiler back in place.
                        self.compiler.set_executable("compiler_so", original_compiler)

                self.compiler._compile = _compile_fn

            super().build_extensions()

    return _CMakeBuildExtension
