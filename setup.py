# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

from importlib import metadata
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import setuptools
from wheel.bdist_wheel import bdist_wheel

from build_tools.build_ext import CMakeExtension, get_build_ext
from build_tools.te_version import te_version
from build_tools.utils import (
    cuda_archs,
    cuda_version,
    get_frameworks,
    remove_dups,
    min_python_version_str,
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
            f"nvidia-cublasmp-cu{cuda_version()[0]}"
        ).locate_file(f"nvidia/cublasmp/cu{cuda_version()[0]}")
        cmake_flags.append(f"-DCUBLASMP_DIR={cublasmp_dir}")

    if bool(int(os.getenv("NVTE_WITH_CUSOLVERMP", "0"))):
        cmake_flags.append("-DNVTE_WITH_CUSOLVERMP=ON")
        cusolvermp_dir = os.getenv("CUSOLVERMP_HOME", "/usr")
        cmake_flags.append(f"-DCUSOLVERMP_DIR={cusolvermp_dir}")

    # NCCL EP: on by default; auto-disabled if no arch >= 90.
    # Set NVTE_BUILD_WITH_NCCL_EP=0/1 to force off/on.
    nccl_ep_env = os.getenv("NVTE_BUILD_WITH_NCCL_EP")
    explicit_nccl_ep = nccl_ep_env is not None
    build_with_nccl_ep = bool(int(nccl_ep_env)) if explicit_nccl_ep else True

    if build_with_nccl_ep:
        arch_tokens = [a.strip() for a in str(archs or "").split(";") if a.strip()]
        has_hopper_or_newer = any(t.lower() == "native" for t in arch_tokens) or any(
            int(t.rstrip("af")) >= 90 for t in arch_tokens if t.rstrip("af").isdigit()
        )
        if not has_hopper_or_newer:
            if explicit_nccl_ep:
                raise RuntimeError(
                    "NVTE_BUILD_WITH_NCCL_EP=1 requires at least one CUDA arch >= 90 in "
                    f"NVTE_CUDA_ARCHS (got '{archs}'). Add '90' or unset NVTE_BUILD_WITH_NCCL_EP."
                )
            print(
                "[NCCL EP] No CUDA arch >= 90 in NVTE_CUDA_ARCHS"
                f" ('{archs}'); auto-disabling NCCL EP (nvte_ep_* will throw at runtime)."
            )
            build_with_nccl_ep = False

    if build_with_nccl_ep:
        build_nccl_ep_submodule()
    else:
        cmake_flags.append("-DNVTE_WITH_NCCL_EP=OFF")

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


def _discover_nccl_home() -> str:
    """Resolve NCCL_HOME: honor env var, else probe well-known prefixes, else ldconfig."""
    env_home = os.environ.get("NCCL_HOME")
    if env_home:
        if (Path(env_home) / "include" / "nccl.h").exists():
            return env_home
        print(
            f"[NCCL EP] WARNING: NCCL_HOME='{env_home}' is set but "
            f"'{env_home}/include/nccl.h' was not found; falling back to system probes."
        )

    lib_names = ("libnccl.so", "libnccl.so.2")
    # Include Debian/Ubuntu multiarch subdirs (e.g. lib/aarch64-linux-gnu).
    lib_subdirs = ("lib", "lib64", "lib/aarch64-linux-gnu", "lib/x86_64-linux-gnu")
    for cand in ("/opt/nvidia/nccl", "/usr/local/nccl", "/usr"):
        p = Path(cand)
        if (p / "include" / "nccl.h").exists() and any(
            (p / sub / name).exists() for sub in lib_subdirs for name in lib_names
        ):
            return str(p)

    try:
        out = subprocess.check_output(["ldconfig", "-p"], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if "libnccl.so" in line and "=>" in line:
                lib_path = Path(line.split("=>")[-1].strip())
                # Walk upward so multiarch layouts (.../lib/<triplet>/libnccl.so)
                # resolve to the prefix that contains include/nccl.h.
                for root in (lib_path.parent.parent, lib_path.parent.parent.parent):
                    if (root / "include" / "nccl.h").exists():
                        return str(root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        "Could not locate NCCL core (nccl.h + libnccl.so). Set NCCL_HOME to the install prefix."
    )


def build_nccl_ep_submodule() -> str:
    """Build libnccl_ep.so from the 3rdparty/nccl submodule.

    NCCL EP is on by default; the system NCCL core (libnccl.so) supplies the
    headers and runtime symbols. Returns the submodule build directory.
    """
    nccl_root = current_file_path / "3rdparty" / "nccl"
    if not (nccl_root / "Makefile").exists():
        raise RuntimeError(
            f"NCCL submodule not found at {nccl_root}. "
            "Run `git submodule update --init --recursive`."
        )

    build_dir = nccl_root / "build"
    nccl_ep_lib = build_dir / "lib" / "libnccl_ep.so"

    archs = cuda_archs() or "90"
    arch_list = []
    for a in str(archs).split(";"):
        a = a.strip().rstrip("af")
        if a and a.isdigit() and int(a) >= 90:
            arch_list.append(a)
    if not arch_list:
        arch_list = ["90"]
    gencode = " ".join(f"-gencode=arch=compute_{a},code=sm_{a}" for a in arch_list)

    nproc = os.cpu_count() or 8
    env = os.environ.copy()
    env["NVCC_GENCODE"] = gencode
    # NCCL EP needs the core NCCL headers + libnccl.so; write NCCL EP build
    # outputs to the submodule's local build/ tree.
    nccl_home = _discover_nccl_home()
    env["NCCL_HOME"] = nccl_home
    env["NCCL_EP_BUILDDIR"] = str(build_dir)

    if not nccl_ep_lib.exists():
        print(f"[NCCL EP] Building libnccl_ep.so (gencode='{gencode}')")
        subprocess.check_call(
            ["make", "-j", str(nproc), "-C", "contrib/nccl_ep", "lib"],
            cwd=str(nccl_root),
            env=env,
        )

    # Stage libnccl_ep.so.0 alongside libtransformer_engine.so so $ORIGIN-rpath
    # finds it in the installed wheel.
    soname = "libnccl_ep.so.0"
    src = (build_dir / "lib" / soname).resolve()
    dst = current_file_path / "transformer_engine" / soname
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    print(f"[NCCL EP] Bundled {dst} ({src.stat().st_size // (1 << 20)} MB)")

    # TE's CMake expects nccl.h under 3rdparty/nccl/build/include/ for its
    # version check. Mirror the top-level host headers from the system NCCL
    # install — DON'T mirror nccl_device/ because the submodule ships its own
    # newer copy at src/include/nccl_device/ with device-side templates that
    # conflict with older system versions, and the JIT include path picks the
    # submodule's.
    nccl_include = build_dir / "include"
    nccl_include.mkdir(parents=True, exist_ok=True)
    for cand in (Path(nccl_home) / "include", Path("/usr/include")):
        p = Path(cand)
        if (p / "nccl.h").exists():
            for name in ("nccl.h", "nccl_net.h", "nccl_tuner.h"):
                src = p / name
                dst = nccl_include / name
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)
            break

    return str(build_dir)


def git_check_submodules() -> None:
    """
    Attempt to checkout git submodules automatically during setup.

    This runs successfully only if the submodules are
    either in the correct or uninitialized state.

    Note to devs: With this, any updates to the submodules itself, e.g. moving to a newer
    commit, must be commited before build. This also ensures that stale submodules aren't
    being silently used by developers.
    """

    # Provide an option to skip these checks for development.
    if bool(int(os.getenv("NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD", "0"))):
        return

    # Require git executable.
    if shutil.which("git") is None:
        return

    # Require a .gitmodules file.
    if not (current_file_path / ".gitmodules").exists():
        return

    try:
        submodules = subprocess.check_output(
            ["git", "submodule", "status", "--recursive"],
            cwd=str(current_file_path),
            text=True,
        ).splitlines()

        for submodule in submodules:
            # '-' start is for an uninitialized submodule.
            # ' ' start is for a submodule on the correct commit.
            assert submodule[0] in (
                " ",
                "-",
            ), (
                "Submodules are initialized incorrectly. If this is intended, set the "
                "environment variable `NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD` to a "
                "non-zero value to skip these checks during development. Otherwise, "
                "run `git submodule update --init --recursive` to checkout the correct"
                " submodule commits."
            )

        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(current_file_path),
        )
    except subprocess.CalledProcessError:
        return


if __name__ == "__main__":
    __version__ = te_version()

    git_check_submodules()

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
        install_requires = []
        extras_require = {
            "core": [f"transformer_engine_cu12=={__version__}"],
            "core_cu12": [f"transformer_engine_cu12=={__version__}"],
            "core_cu13": [f"transformer_engine_cu13=={__version__}"],
            "pytorch": [f"transformer_engine_torch=={__version__}"],
            "jax": [f"transformer_engine_jax=={__version__}"],
        }
    else:
        install_requires, test_requires = setup_requirements()
        ext_modules = [setup_common_extension()]
        # libnccl_ep.so.0 is staged by build_nccl_ep_submodule(); ship it.
        package_data = {"": ["VERSION.txt"], "transformer_engine": ["libnccl_ep.so*"]}
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
        python_requires=f">={min_python_version_str()}",
        classifiers=["Programming Language :: Python :: 3"],
        install_requires=install_requires,
        license_files=("LICENSE",),
        include_package_data=include_package_data,
        package_data=package_data,
    )
