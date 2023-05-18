# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
from subprocess import CalledProcessError
import sys
import tempfile
from typing import List, Optional, Tuple, Union

import setuptools
from setuptools.command.build_ext import build_ext

# Project directory root
root_path: Path = Path(__file__).resolve().parent

@lru_cache(maxsize=1)
def with_debug_build() -> bool:
    """Whether to build with a debug configuration"""
    for arg in sys.argv:
        if arg == "--debug":
            sys.argv.remove(arg)
            return True
    return False

# Call once in global scope since this function manipulates the
# command-line arguments. Future calls will use a cached value.
with_debug_build()

def found_cmake() -> bool:
    """"Check if valid CMake is available

    CMake 3.18 or newer is required.

    """

    # Check if CMake is available
    try:
        _cmake_bin = cmake_bin()
    except FileNotFoundError:
        return False

    # Query CMake for version info
    output = subprocess.check_output(
        [_cmake_bin, "--version"],
        universal_newlines=True,
    )
    match = re.search(r"version\s*([\d.]+)", output)
    version = match.group(1).split('.')
    version = tuple(int(v) for v in version)
    return version >= (3, 18)

def cmake_bin() -> Path:
    """Get CMake executable

    Throws FileNotFoundError if not found.

    """

    # Search in CMake Python package
    _cmake_bin: Optional[Path] = None
    try:
        import cmake
    except ImportError:
        pass
    else:
        cmake_dir = Path(cmake.__file__).resolve().parent
        _cmake_bin = cmake_dir / "data" / "bin" / "cmake"
        if not _cmake_bin.is_file():
            _cmake_bin = None

    # Search in path
    if _cmake_bin is None:
        _cmake_bin = shutil.which("cmake")
        if _cmake_bin is not None:
            _cmake_bin = Path(_cmake_bin).resolve()

    # Return executable if found
    if _cmake_bin is None:
        raise FileNotFoundError("Could not find CMake executable")
    return _cmake_bin

def found_ninja() -> bool:
    """"Check if Ninja is available"""
    return shutil.which("ninja") is not None

def found_pybind11() -> bool:
    """"Check if pybind11 is available"""

    # Check if Python package is installed
    try:
        import pybind11
    except ImportError:
        pass
    else:
        return True

    # Check if CMake can find pybind11
    if not found_cmake():
        return False
    try:
        subprocess.run(
            [
                "cmake",
                "--find-package",
                "-DMODE=EXIST",
                "-DNAME=pybind11",
                "-DCOMPILER_ID=CXX",
                "-DLANGUAGE=CXX",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (CalledProcessError, OSError):
        pass
    else:
        return True
    return False

def cuda_version() -> Tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) tuple

    Throws FileNotFoundError if NVCC is not found.

    """

    # Try finding NVCC
    nvcc_bin: Optional[Path] = None
    if nvcc_bin is None and os.getenv("CUDA_HOME"):
        # Check in CUDA_HOME
        cuda_home = Path(os.getenv("CUDA_HOME"))
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if nvcc_bin is None:
        # Check if nvcc is in path
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin is not None:
            nvcc_bin = Path(nvcc_bin)
    if nvcc_bin is None:
        # Last-ditch guess in /usr/local/cuda
        cuda_home = Path("/usr/local/cuda")
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise FileNotFoundError(f"Could not find NVCC at {nvcc_bin}")

    # Query NVCC for version info
    output = subprocess.check_output(
        [nvcc_bin, "-V"],
        universal_newlines=True,
    )
    match = re.search(r"release\s*([\d.]+)", output)
    version = match.group(1).split('.')
    return tuple(int(v) for v in version)

@lru_cache(maxsize=1)
def with_userbuffers() -> bool:
    """Check if userbuffers support is enabled"""
    if int(os.getenv("NVTE_WITH_USERBUFFERS", "0")):
        assert os.getenv("MPI_HOME"), \
            "MPI_HOME must be set if NVTE_WITH_USERBUFFERS=1"
        return True
    return False

@lru_cache(maxsize=1)
def frameworks() -> List[str]:
    """DL frameworks to build support for"""
    _frameworks: List[str] = []
    supported_frameworks = ["pytorch", "jax", "tensorflow"]

    # Check environment variable
    if os.getenv("NVTE_FRAMEWORK"):
        for framework in os.getenv("NVTE_FRAMEWORK").split(","):
            _frameworks.append(framework)

    # Check command-line arguments
    for arg in sys.argv.copy():
        if arg.startswith("--framework="):
            framework = arg.replace("--framework=", "")
            _frameworks.append(framework)
            sys.argv.remove(arg)

    # Detect installed frameworks if not explicitly specified
    if not _frameworks:
        try:
            import torch
        except ImportError:
            pass
        else:
            _frameworks.append("pytorch")
        try:
            import jax
        except ImportError:
            pass
        else:
            _frameworks.append("jax")
        try:
            import tensorflow
        except ImportError:
            pass
        else:
            _frameworks.append("tensorflow")

    # Special framework names
    if "all" in _frameworks:
        _frameworks = supported_frameworks.copy()
    if "none" in _frameworks:
        _frameworks = []

    # Check that frameworks are valid
    _frameworks = [framework.lower() for framework in _frameworks]
    for framework in _frameworks:
        if framework not in supported_frameworks:
            raise ValueError(
                f"Transformer Engine does not support framework={framework}"
            )

    return _frameworks

# Call once in global scope since this function manipulates the
# command-line arguments. Future calls will use a cached value.
frameworks()

def setup_requirements() -> Tuple[List[str], List[str], List[str]]:
    """Setup Python dependencies

    Returns dependencies for build, runtime, and testing.

    """

    # Common requirements
    setup_reqs: List[str] = []
    install_reqs: List[str] = ["pydantic"]
    test_reqs: List[str] = ["pytest"]

    def add_unique(l: List[str], vals: Union[str, List[str]]) -> None:
        """Add entry to list if not already included"""
        if isinstance(vals, str):
            vals = [vals]
        for val in vals:
            if val not in l:
                l.append(val)

    # Requirements that may be installed outside of Python
    if not found_cmake():
        add_unique(setup_reqs, "cmake>=3.18")
    if not found_ninja():
        add_unique(setup_reqs, "ninja")

    # Framework-specific requirements
    if "pytorch" in frameworks():
        add_unique(install_reqs, ["torch", "flash-attn>=1.0.2"])
        add_unique(test_reqs, ["numpy", "onnxruntime", "torchvision"])
    if "jax" in frameworks():
        if not found_pybind11():
            add_unique(setup_reqs, "pybind11")
        add_unique(install_reqs, ["jax", "flax"])
        add_unique(test_reqs, ["numpy", "praxis"])
    if "tensorflow" in frameworks():
        if not found_pybind11():
            add_unique(setup_reqs, "pybind11")
        add_unique(install_reqs, "tensorflow")
        add_unique(test_reqs, ["keras", "tensorflow_datasets"])

    return setup_reqs, install_reqs, test_reqs


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
        build_type = "Debug" if with_debug_build() else "Release"
        configure_command = [
            _cmake_bin,
            "-S",
            cmake_path,
            "-B",
            build_dir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        configure_command += self.cmake_flags
        if found_ninja():
            configure_command.append("-GNinja")
        try:
            import pybind11
        except ImportError:
            pass
        else:
            pybind11_dir = Path(pybind11.__file__).resolve().parent
            pybind11_dir = pybind11_dir / "share" / "cmake" / "pybind11"
            configure_command.append(f"-Dpybind11_DIR={pybind11_dir}")

        # CMake build and install commands
        build_command = [_cmake_bin, "--build", build_dir]
        install_command = [_cmake_bin, "--install", build_dir]

        # Run CMake commands
        for command in [configure_command, build_command, install_command]:
            print(f"Running command {' '.join(command)}")
            try:
                subprocess.run(command, cwd=build_dir, check=True)
            except (CalledProcessError, OSError) as e:
                raise RuntimeError(f"Error when running CMake: {e}")


# PyTorch extension modules require special handling
if "pytorch" in frameworks():
    from torch.utils.cpp_extension import BuildExtension as BuildExtension
else:
    from setuptools.command.build_ext import build_ext as BuildExtension


class CMakeBuildExtension(BuildExtension):
    """Setuptools command with support for CMake extension modules"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def run(self) -> None:

        # Build CMake extensions
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                print(f"Building CMake extension {ext.name}")
                with tempfile.TemporaryDirectory() as build_dir:
                    build_dir = Path(build_dir)
                    package_path = Path(self.get_ext_fullpath(ext.name))
                    install_dir = package_path.resolve().parent
                    ext._build_cmake(
                        build_dir=build_dir,
                        install_dir=install_dir,
                    )

        # Build non-CMake extensions as usual
        all_extensions = self.extensions
        self.extensions = [
            ext for ext in self.extensions
            if not isinstance(ext, CMakeExtension)
        ]
        super().run()
        self.extensions = all_extensions

def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library

    Also builds JAX, TensorFlow, and userbuffers support if needed.

    """
    cmake_flags = []
    if "jax" in frameworks():
        cmake_flags.append("-DENABLE_JAX=ON")
    if "tensorflow" in frameworks():
        cmake_flags.append("-DENABLE_TENSORFLOW=ON")
    if with_userbuffers():
        cmake_flags.append("-DNVTE_WITH_USERBUFFERS=ON")
    return CMakeExtension(
        name="transformer_engine",
        cmake_path=root_path / "transformer_engine",
        cmake_flags=cmake_flags,
    )

def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    src_dir = root_path / "transformer_engine" / "pytorch" / "csrc"
    sources = [
        src_dir / "extensions.cu",
        src_dir / "common.cu",
        src_dir / "ts_fp8_op.cpp",
    ]

    # Header files
    include_dirs = [
        root_path / "transformer_engine" / "common" / "include",
        root_path / "pytorch" / "csrc",
        root_path / "3rdparty" / "cudnn-frontend" / "include",
    ]

    # Compiler flags
    cxx_flags = ["-O3"]
    nvcc_flags = [
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]

    # Version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version >= (11, 2):
            nvcc_flags.extend(["--threads", "4"])
        if version >= (11, 0):
            nvcc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
        if version >= (11, 8):
            nvcc_flags.extend(["-gencode", "arch=compute_90,code=sm_90"])

    # userbuffers support
    if with_userbuffers():
        if os.getenv("MPI_HOME"):
            mpi_home = Path(os.getenv("MPI_HOME"))
            include_dirs.append(mpi_home / "include")
        cxx_flags.append("-DNVTE_WITH_USERBUFFERS")
        nvcc_flags.append("-DNVTE_WITH_USERBUFFERS")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CUDAExtension
    return CUDAExtension(
        name="transformer_engine_extensions",
        sources=sources,
        include_dirs=include_dirs,
        # libraries=["transformer_engine"], ### TODO Debug linker errors
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )


def main():

    # Read package version from file
    with open(root_path / "VERSION", "r") as f:
        version = f.readline()

    # Setup dependencies
    setup_requires, install_requires, test_requires = setup_requirements()

    # Setup extensions
    ext_modules = [setup_common_extension()]
    if "pytorch" in frameworks():
        ext_modules.append(setup_pytorch_extension())

    # Configure package
    setuptools.setup(
        name="transformer_engine",
        version=version,
        packages=("transformer_engine",),
        description="Transformer acceleration library",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        setup_requires=setup_requires,
        install_requires=install_requires,
        extras_require={"test": test_requires},
        license_files=("LICENSE",),
    )


if __name__ == "__main__":
    main()
