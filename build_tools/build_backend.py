# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PEP 517 backend with CUDA-version-, framework-, and feature-specific build requirements."""

from contextlib import contextmanager
import os
import re
import subprocess
from typing import Iterator, List, NamedTuple, Optional, Tuple

from build_tools.build_backend_utils import (
    ConfigSettings,
    config_value,
    normalize_cuda_version,
    requested_cuda_version,
    setuptools_hooks,
)
from build_tools.utils import MIN_CUDNN_VERSION, nvcc_path

# cuda-toolkit pins all components to the same CUDA release. The nvcc extra includes the
# compiler's crt and nvvm components.
_CUDA_TOOLKIT_EXTRAS = ("nvcc", "cudart", "cccl", "cublas", "nvml", "profiler")
_FRAMEWORK_BUILD_PACKAGES = {
    "pytorch": ["torch>=2.1"],
    "jax": ["jax>=0.5.0", "flax>=0.7.1"],
}
_SUPPORTED_FRAMEWORKS = tuple(_FRAMEWORK_BUILD_PACKAGES)


class _Feature(NamedTuple):
    """Optional feature that can be enabled with an environment variable or config setting."""

    environment_variable: str
    # Selects an existing installation instead of installing the requirements.
    home_variable: Optional[str]
    # Formatted with the CUDA major version.
    requirements: Tuple[str, ...]


# Keyed by config setting, e.g. --config-settings with-cublasmp=1.
_FEATURES = {
    "with-cublasmp": _Feature(
        "NVTE_WITH_CUBLASMP",
        "CUBLASMP_HOME",
        # cuBLASMp's overlap algorithms require NCCL 2.30.
        ("nvidia-cublasmp-cu{cuda_major}>=0.8.1", "nvidia-nccl-cu{cuda_major}>=2.30"),
    ),
    "with-cusolvermp": _Feature(
        "NVTE_WITH_CUSOLVERMP", "CUSOLVERMP_HOME", ("nvidia-cusolvermp-cu{cuda_major}",)
    ),
    "enable-nvshmem": _Feature(
        "NVTE_ENABLE_NVSHMEM", "NVSHMEM_HOME", ("nvidia-nvshmem-cu{cuda_major}",)
    ),
    "ub-with-mpi": _Feature("NVTE_UB_WITH_MPI", "MPI_HOME", ("openmpi",)),
    "with-cutedsl": _Feature("NVTE_WITH_CUTEDSL", None, ("apache-tvm-ffi>=0.1.12",)),
}
_PRIVATE_SETTINGS = ("cuda-version", "framework", *_FEATURES)


def _normalize_frameworks(value: str) -> List[str]:
    """Validate and normalize a framework selection."""
    frameworks = list(
        dict.fromkeys(
            framework.strip().lower() for framework in value.split(",") if framework.strip()
        )
    )
    if not frameworks:
        raise ValueError("Framework selection cannot be empty")

    special_frameworks = {"all", "none"}.intersection(frameworks)
    if special_frameworks:
        if len(frameworks) != 1:
            raise ValueError("'all' and 'none' cannot be combined with other frameworks")
        return list(_SUPPORTED_FRAMEWORKS) if frameworks[0] == "all" else []

    unsupported = [framework for framework in frameworks if framework not in _SUPPORTED_FRAMEWORKS]
    if unsupported:
        supported = ", ".join((*_SUPPORTED_FRAMEWORKS, "all", "none"))
        raise ValueError(
            f"Unsupported framework {unsupported[0]!r}; expected one or more of: {supported}"
        )

    return frameworks


def _config_frameworks(config_settings: ConfigSettings) -> Optional[List[str]]:
    """Get the target frameworks from PEP 517 config settings."""
    value = config_value(config_settings, "framework")
    return _normalize_frameworks(value) if value is not None else None


def _environment_frameworks() -> Optional[List[str]]:
    """Get the target frameworks from the environment."""
    value = os.getenv("NVTE_FRAMEWORK")
    return _normalize_frameworks(value) if value is not None else None


def _requested_frameworks(config_settings: ConfigSettings) -> Optional[List[str]]:
    """Resolve an explicit framework selection in descending order of precedence."""
    configured = _config_frameworks(config_settings)
    return configured if configured is not None else _environment_frameworks()


def _config_feature(config_settings: ConfigSettings, setting: str) -> Optional[bool]:
    """Get whether PEP 517 config settings enable or disable an optional feature."""
    value = config_value(config_settings, setting)
    if value is None:
        return None

    if value not in ("0", "1"):
        raise ValueError(f"Invalid value {value!r} for {setting}; expected 0 or 1")
    return value == "1"


def _feature_enabled(config_settings: ConfigSettings, setting: str) -> bool:
    """Resolve whether an optional feature is enabled, preferring config settings."""
    configured = _config_feature(config_settings, setting)
    if configured is not None:
        return configured
    return bool(int(os.getenv(_FEATURES[setting].environment_variable, "0")))


@contextmanager
def _build_environment(config_settings: ConfigSettings) -> Iterator[None]:
    """Apply config-setting framework and feature selections while running setuptools."""
    overrides = {}
    frameworks = _config_frameworks(config_settings)
    if frameworks is not None:
        overrides["NVTE_FRAMEWORK"] = ",".join(frameworks) if frameworks else "none"
    for setting, feature in _FEATURES.items():
        enabled = _config_feature(config_settings, setting)
        if enabled is not None:
            overrides[feature.environment_variable] = "1" if enabled else "0"

    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _nvcc_cuda_version() -> Optional[str]:
    """Get the version reported by the selected NVCC."""
    if (cuda_compiler := nvcc_path()) is None:
        return None

    try:
        result = subprocess.run(
            [cuda_compiler, "--version"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
    return normalize_cuda_version(match.group(1)) if match is not None else None


def _torch_cuda_version() -> Optional[str]:
    """Get the CUDA version required by PyTorch, if available."""
    try:
        import torch
    except (ImportError, OSError):
        return None
    value = getattr(getattr(torch, "version", None), "cuda", None)
    return normalize_cuda_version(value) if value else None


def _framework_cuda_version(config_settings: ConfigSettings) -> Optional[str]:
    """Get a CUDA version required by the selected framework."""
    frameworks = _requested_frameworks(config_settings)

    if frameworks is None or "pytorch" in frameworks:
        if cuda_version := _torch_cuda_version():
            return cuda_version

    # JAX CUDA plugin names identify the CUDA major version, but not the
    # major/minor toolkit release needed to select component packages.
    return None


def _cuda_version(config_settings: ConfigSettings) -> str:
    """Resolve the CUDA version in descending order of precedence."""
    cuda_version = (
        requested_cuda_version(config_settings)
        or _nvcc_cuda_version()
        or _framework_cuda_version(config_settings)
    )
    if cuda_version is None:
        raise ValueError(
            "Could not determine the CUDA version for an isolated build; install a CUDA "
            "toolkit, set CUDA_HOME, or pass --config-settings cuda-version=<major>.<minor> "
            "(or set NVTE_CUDA_VERSION)"
        )
    return cuda_version


def _cuda_build_requirements(config_settings: ConfigSettings) -> List[str]:
    """Get build requirements for the requested CUDA version.

    The CUDA Toolkit and cuDNN are installed from Python packages unless CUDA_HOME selects an
    installed CUDA Toolkit, in which case the installed cuDNN is used as well.
    """
    cuda_version = _cuda_version(config_settings)
    cuda_major = cuda_version.split(".", maxsplit=1)[0]
    requirements = [f"nvidia-nccl-cu{cuda_major}>=2"]
    if not os.getenv("CUDA_HOME"):
        requirements += [
            f"cuda-toolkit[{','.join(_CUDA_TOOLKIT_EXTRAS)}]=={cuda_version}.*",
            f"nvidia-cudnn-cu{cuda_major}>={MIN_CUDNN_VERSION}",
        ]
    return requirements


def _framework_build_requirements(config_settings: ConfigSettings) -> List[str]:
    """Get build requirements for the requested frameworks."""
    frameworks = _requested_frameworks(config_settings)
    if frameworks is None:
        # Without a selection, setup.py detects frameworks in the isolated build
        # environment, which has none, so only the core library is built.
        return []

    return [
        requirement
        for framework in frameworks
        for requirement in _FRAMEWORK_BUILD_PACKAGES[framework]
    ]


def _feature_build_requirements(config_settings: ConfigSettings) -> List[str]:
    """Get build requirements for the enabled optional features."""
    cuda_major = _cuda_version(config_settings).split(".", maxsplit=1)[0]
    return [
        requirement.format(cuda_major=cuda_major)
        for setting, feature in _FEATURES.items()
        if _feature_enabled(config_settings, setting)
        and not (feature.home_variable and os.getenv(feature.home_variable))
        for requirement in feature.requirements
    ]


def _build_requirements(config_settings: ConfigSettings) -> List[str]:
    """Get CUDA, framework, and optional feature build requirements."""
    framework_requirements = _framework_build_requirements(config_settings)
    return (
        _cuda_build_requirements(config_settings)
        + framework_requirements
        + _feature_build_requirements(config_settings)
    )


# PEP 517 and PEP 660 hooks. setup.py always depends on the build requirements, so they are
# also needed for sdists.
globals().update(
    setuptools_hooks(
        requirements=_build_requirements,
        private_settings=_PRIVATE_SETTINGS,
        environment=_build_environment,
    )
)
