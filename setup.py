from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
from subprocess import CalledProcessError
import sys
import tempfile
from typing import List, Optional, Tuple

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
    """"Check if CMake is available"""
    return shutil.which("cmake") is not None

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

    Throws RuntimeError if NVCC is not found.

    """

    # Try finding NVCC
    nvcc_bin = None
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
        raise RuntimeError(f"Could not find NVCC at {nvcc_bin}")

    # Get NVCC version info
    nvcc_output = subprocess.check_output(
        [nvcc_bin, "-V"],
        universal_newlines=True,
    )

    # Parse NVCC version info
    match = re.search(
        r"release (\d+)\.(\d+),",
        nvcc_output,
    )
    major = int(match.group(1))
    minor = int(match.group(2))
    return major, minor

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
    _frameworks = []
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
    for framework in _frameworks:
        if framework not in supported_frameworks:
            raise ValueError(
                f"Transformer Engine does not support framework={framework}"
            )

    return _frameworks

# Call once in global scope since this function manipulates the
# command-line arguments. Future calls will use a cached value.
frameworks()


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
        cmake_path = str(self.cmake_path)
        build_dir = str(build_dir)
        install_dir = str(install_dir)

        # Assume CMake is in path
        cmake_bin = shutil.which("cmake")

        # CMake commands
        build_type = "Debug" if with_debug_build() else "Release"
        configure_command = [
            cmake_bin,
            "-S",
            cmake_path,
            "-B",
            build_dir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_PREFIX_PATH={install_dir}",
        ]
        if found_ninja():
            configure_command.append("-GNinja")
        configure_command += self.cmake_flags
        build_command = [cmake_bin, "--build", build_dir]
        install_command = [cmake_bin, "--install", build_dir]

        # Run CMake commands
        for command in [configure_command, build_command, install_command]:
            print(f"Running CMake: {' '.join(command)}")
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
            if not isinstance(ext, CMakeExtension):
                continue
            print(f"Building CMake extension {ext.name}")

            # Create temporary build dir
            with tempfile.TemporaryDirectory() as build_dir:
                build_dir = Path(build_dir)

                # Determine install dir
                ext_path = Path(self.get_ext_fullpath(ext.name))
                install_dir = ext_path.parent.resolve() ### TODO Figure out

                # Build CMake project
                ext._build_cmake(build_dir, install_dir)

        # Build non-CMake extensions as usual
        self.extensions = [
            ext for ext in self.extensions
            if not isinstance(ext, CMakeExtension)
        ]
        super().run()


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
    except RuntimeError:
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
        libraries=["transformer_engine"],
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
    install_requires = ["pydantic"]
    if not found_cmake():
        install_requires.append("cmake")
    if not found_ninja():
        install_requires.append("ninja")
    if 'jax' in frameworks() or 'tensorflow' in frameworks():
        if not found_pybind11():
            install_requires.append("pybind11")
    framework_requires = {
        "pytorch": ["torch", "flash-attn>=1.0.2"],
        "jax": ["jax", "flax"],
        "tensorflow": ["tensorflow"],
    }
    for framework in frameworks():
        install_requires.extend(framework_requires[framework])

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
        install_requires=install_requires,
        extras_require={
            "test": ["pytest",
                     "tensorflow_datasets"],
            "test_pytest": ["onnxruntime",],
        },
        license_files=("LICENSE",),
    )


if __name__ == "__main__":
    main()
