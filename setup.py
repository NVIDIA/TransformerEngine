# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
from subprocess import CalledProcessError
import sys
import sysconfig
import platform
from typing import List, Optional, Tuple, Union
import copy

import setuptools

from te_version import te_version

# Project directory root
root_path: Path = Path(__file__).resolve().parent

@lru_cache(maxsize=1)
def with_debug_build() -> bool:
    """Whether to build with a debug configuration"""
    for arg in sys.argv:
        if arg == "--debug":
            sys.argv.remove(arg)
            return True
    if int(os.getenv("NVTE_BUILD_DEBUG", "0")):
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
    output = subprocess.run(
        [_cmake_bin, "--version"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"version\s*([\d.]+)", output.stdout)
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
        import pybind11  # pylint: disable=unused-import
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
                str(cmake_bin()),
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

# Find the library file extension for the OS type
@lru_cache(maxsize=1)
def lib_extension():
    """Get the library file extension for the operating system"""
    system = platform.system()
    if system == "Linux":
        lib_ext = "so"
    elif system == "Darwin":
        lib_ext = "dylib"
    elif system == "Windows":
        lib_ext = "dll"
    else:
        raise RuntimeError(f"Unsupported operating system ({system})")
    return lib_ext

@lru_cache
def cuda_path() -> Tuple[str, str]:
    """CUDA root path and NVCC binary path as a tuple.

    Throws FileNotFoundError if NVCC is not found."""
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
            cuda_home = Path(nvcc_bin.rstrip("/bin/nvcc"))
            nvcc_bin = Path(nvcc_bin)
    if nvcc_bin is None:
        # Last-ditch guess in /usr/local/cuda
        cuda_home = Path("/usr/local/cuda")
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise FileNotFoundError(f"Could not find NVCC at {nvcc_bin}")

    return cuda_home, nvcc_bin

def cuda_version() -> Tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) tuple."""
    # Query NVCC for version info
    _, nvcc_bin = cuda_path()
    output = subprocess.run(
        [nvcc_bin, "-V"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    version = match.group(1).split('.')
    return tuple(int(v) for v in version)

@lru_cache
def cudnn_path() -> Tuple[str, str]:
    """cuDNN include and library paths.

    Throws FileNotFoundError if libcudnn.so is not found."""
    assert os.path.exists(root_path / "3rdparty" / "cudnn-frontend"), (
        "Could not find cuDNN frontend API. Try running 'git submodule update --init --recursive' "
        "within the Transformer Engine source.")
    try:
        shutil.copy2(root_path / "3rdparty" / "cudnn-frontend" / "cmake" / "cuDNN.cmake",
                     root_path / "FindCUDNN.cmake")
        find_cudnn = subprocess.run(
            [
                str(cmake_bin()),
                "--find-package",
                f"-DCMAKE_MODULE_PATH={str(root_path)}",
                "-DCMAKE_FIND_DEBUG_MODE=ON",
                "-DMODE=EXIST",
                "-DCOMPILER_ID=CXX",
                "-DLANGUAGE=CXX",
                "-DNAME=CUDNN",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        os.remove(root_path / "FindCUDNN.cmake")
    except subprocess.CalledProcessError as e:
        raise FileNotFoundError("Could not find a cuDNN installation.") from e

    cudnn_include = None
    cudnn_link = None
    lib_ext = lib_extension()
    for line in find_cudnn.stderr.splitlines():
        if "cudnn.h" in line:
            cudnn_include = Path(line.lstrip()).parent
        elif "libcudnn." + lib_ext in line:
            cudnn_link = Path(line.lstrip()).parent
        if cudnn_include is not None and cudnn_link is not None:
            break
    if cudnn_include is None or cudnn_link is None:
        raise FileNotFoundError("Could not find a cuDNN installation.")

    return cudnn_include, cudnn_link

@lru_cache(maxsize=1)
def frameworks() -> List[str]:
    """DL frameworks to build support for"""
    _frameworks: List[str] = []
    supported_frameworks = ["pytorch", "jax", "paddle"]

    # Check environment variable
    if os.getenv("NVTE_FRAMEWORK"):
        _frameworks.extend(os.getenv("NVTE_FRAMEWORK").split(","))

    # Check command-line arguments
    for arg in sys.argv.copy():
        if arg.startswith("--framework="):
            _frameworks.extend(arg.replace("--framework=", "").split(","))
            sys.argv.remove(arg)

    # Detect installed frameworks if not explicitly specified
    if not _frameworks:
        try:
            import torch  # pylint: disable=unused-import
        except ImportError:
            pass
        else:
            _frameworks.append("pytorch")
        try:
            import jax  # pylint: disable=unused-import
        except ImportError:
            pass
        else:
            _frameworks.append("jax")
        try:
            import paddle  # pylint: disable=unused-import
        except ImportError:
            pass
        else:
            _frameworks.append("paddle")

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

def install_and_import(package):
    """Install a package via pip (if not already installed) and import into globals."""
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    finally:
        globals()[package] = importlib.import_module(package)

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
        add_unique(install_reqs, ["torch", "flash-attn>=2.0.6,<=2.5.8,!=2.0.9,!=2.1.0"])
        add_unique(test_reqs, ["numpy", "onnxruntime", "torchvision"])
    if "jax" in frameworks():
        add_unique(install_reqs, ["jax", "flax>=0.7.1"])
        add_unique(test_reqs, ["numpy", "praxis"])
    if "paddle" in frameworks():
        add_unique(install_reqs, "paddlepaddle-gpu")
        add_unique(test_reqs, "numpy")

    return setup_reqs, install_reqs, test_reqs


# PyTorch extension modules require special handling
if "pytorch" in frameworks():
    from torch.utils.cpp_extension import BuildExtension  # pylint: disable=import-error
elif "paddle" in frameworks():
    from paddle.utils.cpp_extension import BuildExtension  # pylint: disable=import-error
else:
    install_and_import('pybind11')
    from pybind11.setup_helpers import build_ext as BuildExtension


class CMakeBuildExtension(BuildExtension):
    """Setuptools command with support for CMake extension modules"""

    def run(self) -> None:
        # Build CMake extensions
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                print(f"Building CMake extension {ext.name}")
                # Set up incremental builds for CMake extensions
                setup_dir = Path(__file__).resolve().parent
                build_dir = setup_dir / "build" / "cmake"
                build_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
                package_path = Path(self.get_ext_fullpath(ext.name))
                install_dir = package_path.resolve().parent
                ext._build_cmake(
                    build_dir=build_dir,
                    install_dir=install_dir,
                )

        # Paddle requires linker search path for libtransformer_engine.so
        paddle_ext = None
        if "paddle" in frameworks():
            for ext in self.extensions:
                if "paddle" in ext.name:
                    ext.library_dirs.append(self.build_lib)
                    paddle_ext = ext
                    break

        # Build non-CMake extensions as usual
        all_extensions = self.extensions
        self.extensions = [
            ext for ext in self.extensions
            if not isinstance(ext, CMakeExtension)
        ]
        super().run()
        self.extensions = all_extensions

        # Manually write stub file for Paddle extension
        if paddle_ext is not None:

            # Load libtransformer_engine.so to avoid linker errors
            for path in Path(self.build_lib).iterdir():
                if path.name.startswith("libtransformer_engine."):
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

            # Figure out stub file path
            module_name = paddle_ext.name
            assert module_name.endswith("_pd_"), \
                "Expected Paddle extension module to end with '_pd_'"
            stub_name = module_name[:-4]  # remove '_pd_'
            stub_path = os.path.join(self.build_lib, stub_name + ".py")

            # Figure out library name
            # Note: This library doesn't actually exist. Paddle
            # internally reinserts the '_pd_' suffix.
            so_path = self.get_ext_fullpath(module_name)
            _, so_ext = os.path.splitext(so_path)
            lib_name = stub_name + so_ext

            # Write stub file
            print(f"Writing Paddle stub for {lib_name} into file {stub_path}")
            from paddle.utils.cpp_extension.extension_utils import custom_write_stub
            custom_write_stub(lib_name, stub_path)

    def build_extensions(self):
        # BuildExtensions from PyTorch and PaddlePaddle already handle CUDA files correctly
        # so we don't need to modify their compiler. Only the pybind11 build_ext needs to be fixed.
        if "pytorch" not in frameworks() and "paddle" not in frameworks():
            # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
            # extra_compile_args is a dict.
            for ext in self.extensions:
                if isinstance(ext.extra_compile_args, dict):
                    for target in ['cxx', 'nvcc']:
                        if target not in ext.extra_compile_args.keys():
                            ext.extra_compile_args[target] = []

            # Define new _compile method that redirects to NVCC for .cu and .cuh files.
            original_compile_fn = self.compiler._compile
            self.compiler.src_extensions += ['.cu', '.cuh']
            def _compile_fn(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
                # Copy before we make any modifications.
                cflags = copy.deepcopy(extra_postargs)
                original_compiler = self.compiler.compiler_so
                try:
                    _, nvcc_bin = cuda_path()
                    original_compiler = self.compiler.compiler_so

                    if os.path.splitext(src)[1] in ['.cu', '.cuh']:
                        self.compiler.set_executable('compiler_so', str(nvcc_bin))
                        if isinstance(cflags, dict):
                            cflags = cflags['nvcc']

                        # Add -fPIC if not already specified
                        if not any('-fPIC' in flag for flag in cflags):
                            cflags.extend(['--compiler-options', "'-fPIC'"])

                        # Forward unknown options
                        if not any('--forward-unknown-opts' in flag for flag in cflags):
                            cflags.append('--forward-unknown-opts')

                    elif isinstance(cflags, dict):
                        cflags = cflags['cxx']

                    # Append -std=c++17 if not already in flags
                    if not any(flag.startswith('-std=') for flag in cflags):
                        cflags.append('-std=c++17')

                    return original_compile_fn(obj, src, ext, cc_args, cflags, pp_opts)

                finally:
                    # Put the original compiler back in place.
                    self.compiler.set_executable('compiler_so', original_compiler)

            self.compiler._compile = _compile_fn

        super().build_extensions()


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
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        configure_command += self.cmake_flags
        if found_ninja():
            configure_command.append("-GNinja")

        import pybind11
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
                raise RuntimeError(f"Error when running CMake: {e}")  # pylint: disable=raise-missing-from

GLIBCXX_USECXX11_ABI = None

def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library

    Also builds JAX or userbuffers support if needed.

    """
    # FindPythonInterp and FindPythonLibs are deprecated in newer CMake versions,
    # but PyBind11 still tries to use them unless we set PYBIND11_FINDPYTHON=ON.
    cmake_flags = [ "-DPYBIND11_FINDPYTHON=ON" ]

    # Optionally switch userbuffers bootstrapping to the old MPI method.
    # NOTE: This requires launching PyTorch distributed runs with
    #       `mpiexec -np <N> -x MASTER_ADDR=<host addr> -x MASTER_PORT=<host port> -x PATH ...`
    #       instead of `torchrun --nproc-per-node=<N> ...`
    if int(os.getenv("UB_MPI_BOOTSTRAP", "0")):
        assert os.getenv("MPI_HOME"), \
            "MPI_HOME must be set if UB_MPI_BOOTSTRAP=1"
        cmake_flags += [ "-DUB_MPI_BOOTSTRAP=ON" ]

    # If we need to build TE/PyTorch extensions later, we need to compile core TE library with
    # the same C++ ABI version as PyTorch.
    if "pytorch" in frameworks():
        import torch
        global GLIBCXX_USECXX11_ABI
        GLIBCXX_USECXX11_ABI = torch.compiled_with_cxx11_abi()
        cmake_flags.append(f"-DUSE_CXX11_ABI={int(GLIBCXX_USECXX11_ABI)}")

    return CMakeExtension(
        name="transformer_engine",
        cmake_path=root_path / "transformer_engine",
        cmake_flags=cmake_flags,
    )

def _all_files_in_dir(path):
    return list(path.iterdir())

def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    src_dir = root_path / "transformer_engine" / "pytorch" / "csrc"
    extensions_dir = src_dir / "extensions"
    sources = [
        src_dir / "common.cu",
        src_dir / "ts_fp8_op.cpp",
    ] + _all_files_in_dir(extensions_dir)

    # Header files
    include_dirs = [
        root_path / "transformer_engine" / "common" / "include",
        root_path / "transformer_engine" / "pytorch" / "csrc",
        root_path / "transformer_engine",
        root_path / "3rdparty" / "cudnn-frontend" / "include",
    ]

    # Compiler flags
    cxx_flags = [
        "-O3",
        "-Wno-return-local-addr",
    ]
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

    # Add PyBind flags
    import pybind11
    include_dirs.append(pybind11.get_include())
    if lib_extension() == "dll":
        cxx_flags += [ "/EHsc", "/bigobj" ]
    else:
        cxx_flags += [ "-fvisibility=hidden" ]

    # Link core TE library
    lib_kwargs = { 'libraries' : [ 'transformer_engine' ] }
    if lib_extension() == 'dll':
        # Windows can dynamically load from the same folder as the extension library but needs full
        # DLL path to link correctly at compile-time.
        lib_kwargs['libraries'] = [
            str(root_path / 'build' / 'cmake' / 'common' / 'libtransformer_engine.dll')
        ]
    else:
        # We don't know the pip install path for libtransformer_engine.so, but we can link
        # against it at compile time using the fixed CMake build path.
        lib_kwargs['library_dirs'] = [ str(root_path / 'build' / 'cmake' / 'common') ]
        # Pip will install the framework extension libraries into the same directory as
        # libtransformer_engine.so, which means we can dynamically load from the framework
        # extension's $ORIGIN path at runtime.
        lib_kwargs['extra_link_args'] = [ '-Wl,-rpath,$ORIGIN' ]

    # Construct PyTorch CUDA extension
    from torch.utils.cpp_extension import CUDAExtension
    return CUDAExtension(
        name="transformer_engine_extensions",
        sources=[ str(path) for path in sources ],
        include_dirs=[ str(path) for path in include_dirs ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        **lib_kwargs,
    )


def setup_jax_extension() -> setuptools.Extension:
    """Setup PyBind11 extension for JAX support"""
    # Source files
    src_dir = root_path / "transformer_engine" / "jax" / "csrc"
    sources = [
        src_dir / "extensions.cpp",
        src_dir / "modules.cpp",
        src_dir / "utils.cu",
    ]

    # Header files
    cuda_home, _ = cuda_path()
    cudnn_include, cudnn_link = cudnn_path()
    include_dirs = [
        cuda_home / "include",
        cudnn_include,
        root_path / "transformer_engine" / "common" / "include",
        root_path / "transformer_engine" / "jax" / "csrc",
        root_path / "transformer_engine",
    ]

    # Compile flags
    cxx_flags = [ "-O3" ]
    nvcc_flags = [ "-O3" ]
    if GLIBCXX_USECXX11_ABI is not None:
        # Compile JAX extensions with same ABI as core library
        flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(GLIBCXX_USECXX11_ABI)}"
        cxx_flags.append(flag)
        nvcc_flags.append(flag)

    # Linked libraries
    libraries = [
        'cudart',
        'cublas',
        'cublasLt',
        'cudnn',
    ]
    lib_kwargs = {}
    if lib_extension() == 'dll':
        # Windows can dynamically load from the same folder as the extension library but needs full
        # DLL path to link correctly at compile-time.
        cuda_lib_dir = cuda_home / 'lib' / 'x64'
        libraries = [ cuda_lib_dir / f'lib{name}.dll' for name in libraries ]
        lib_kwargs['libraries'] = [ str(path) for path in libraries ] + [
            str(root_path / 'build' / 'cmake' / 'common' /'libtransformer_engine.dll')
        ]
    else:
        # Set link and runtime paths for CUDA libraries.
        cuda_lib_dir = cuda_home / 'lib64'
        if (not cuda_lib_dir.exists() and (cuda_home / 'lib').exists()):
            cuda_lib_dir = cuda_home / 'lib'
        lib_kwargs['libraries'] = libraries
        lib_kwargs['library_dirs'] = [
            str(cuda_lib_dir),
            str(cudnn_link),
        ]
        lib_kwargs['extra_link_args'] = [
            f"-Wl,-rpath,{path}" for path in lib_kwargs['library_dirs']
        ]
        lib_kwargs['libraries'].append('transformer_engine')
        # We don't know the pip install path for libtransformer_engine.so, but we can link
        # against it at compile time using the fixed CMake build path.
        lib_kwargs['library_dirs'].append(str(root_path / 'build' / 'cmake' / 'common'))
        # Pip will install the framework extension libraries into the same directory as
        # libtransformer_engine.so, which means we can dynamically load from the framework
        # extension's $ORIGIN path at runtime.
        lib_kwargs['extra_link_args'].append('-Wl,-rpath,$ORIGIN')

    # Add PyBind11 to the extension
    from pybind11.setup_helpers import Pybind11Extension
    class Pybind11CUDAExtension(Pybind11Extension):
        """Modified Pybind11Extension to allow combined CXX + NVCC compile flags."""

        def _add_cflags(self, flags: List[str]) -> None:
            if isinstance(self.extra_compile_args, dict):
                cxx_flags = self.extra_compile_args.pop('cxx', [])
                cxx_flags += flags
                self.extra_compile_args['cxx'] = cxx_flags
            else:
                self.extra_compile_args[:0] = flags

    return Pybind11CUDAExtension(
        "transformer_engine_jax",
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags
        },
        **lib_kwargs,
    )


def setup_paddle_extension() -> setuptools.Extension:
    """Setup CUDA extension for Paddle support"""

    # Source files
    src_dir = root_path / "transformer_engine" / "paddle" / "csrc"
    sources = [
        src_dir / "extensions.cu",
        src_dir / "common.cpp",
        src_dir / "custom_ops.cu",
    ]

    # Header files
    include_dirs = [
        root_path / "transformer_engine" / "common" / "include",
        root_path / "transformer_engine" / "paddle" / "csrc",
        root_path / "transformer_engine",
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
    if GLIBCXX_USECXX11_ABI is not None:
        # Compile JAX extensions with same ABI as core library
        flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(GLIBCXX_USECXX11_ABI)}"
        cxx_flags.append(flag)
        nvcc_flags.append(flag)

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

    # Link core TE library
    lib_kwargs = { 'libraries' : [ 'transformer_engine' ] }
    if lib_extension() == 'dll':
        # Windows can dynamically load from the same folder as the extension library but needs full
        # DLL path to link correctly at compile-time.
        lib_kwargs['libraries'] = [
            str(root_path / 'build' / 'cmake' / 'common' / 'libtransformer_engine.dll')
        ]
    else:
        # We don't know the pip install path for libtransformer_engine.so, but we can link
        # against it at compile time using the fixed CMake build path.
        lib_kwargs['library_dirs'] += [ str(root_path / 'build' / 'cmake' / 'common') ]
        # Pip will install the framework extension libraries into the same directory as
        # libtransformer_engine.so, which means we can dynamically load from the framework
        # extension's $ORIGIN path at runtime.
        lib_kwargs['extra_link_args'] = [ '-Wl,-rpath,$ORIGIN' ]

    # Construct Paddle CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from paddle.utils.cpp_extension import CUDAExtension  # pylint: disable=import-error
    ext = CUDAExtension(
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        **lib_kwargs
    )
    ext.name = "transformer_engine_paddle_pd_"
    return ext


if __name__ == "__main__":
    # Submodules to install
    packages = setuptools.find_packages(
        include=["transformer_engine", "transformer_engine.*"],
    )

    # Dependencies
    setup_requires, install_requires, test_requires = setup_requirements()

    # Extensions
    ext_modules = [setup_common_extension()]

    if "pytorch" in frameworks():
        ext_modules.append(setup_pytorch_extension())

    if "jax" in frameworks():
        ext_modules.append(setup_jax_extension())

    if "paddle" in frameworks():
        ext_modules.append(setup_paddle_extension())

    # Configure package
    setuptools.setup(
        name="transformer_engine",
        version=te_version(),
        packages=packages,
        description="Transformer acceleration library",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        setup_requires=setup_requires,
        install_requires=install_requires,
        extras_require={"test": test_requires},
        license_files=("LICENSE",),
    )
