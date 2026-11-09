# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PEP 517 backend with CUDA-version-, framework-, and feature-specific build requirements."""

from contextlib import contextmanager
import os
import re
import subprocess
from typing import Iterator, List, Mapping, NamedTuple, Optional, Tuple, Union

from setuptools import build_meta

from build_tools.utils import nvcc_path

_SETUPTOOLS_BACKEND = build_meta.__legacy__
_CUDA_BUILD_PACKAGES = {
    "12": [
        # Also contains nvvm and crt
        "nvidia-cuda-nvcc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-cccl-cu12",
        "nvidia-cuda-profiler-api-cu12",
        "nvidia-nvml-dev-cu12",
    ],
    "13": [
        "nvidia-cuda-nvcc",
        "nvidia-cuda-runtime",
        "nvidia-cuda-crt",
        "nvidia-cuda-cccl",
        "nvidia-cuda-profiler-api",
        "nvidia-nvml-dev",
        "nvidia-nvvm",
    ],
}
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

ConfigValue = Union[str, List[str]]
ConfigSettings = Optional[Mapping[str, ConfigValue]]


def _setuptools_config_settings(config_settings: ConfigSettings) -> ConfigSettings:
    """Remove settings that are private to this backend."""
    if config_settings is None:
        return None

    settings = dict(config_settings)
    for setting in _PRIVATE_SETTINGS:
        settings.pop(setting, None)
    return settings or None


def _normalize_cuda_version(value: str) -> str:
    """Validate and normalize a CUDA major/minor version."""
    value = value.lower().removeprefix("cu")
    if re.fullmatch(r"\d+\.\d+", value) is None:
        raise ValueError(
            f"Invalid CUDA version {value!r}; expected <major>.<minor>, for example 12.8"
        )

    major = value.split(".", maxsplit=1)[0]
    if major not in _CUDA_BUILD_PACKAGES:
        supported = ", ".join(sorted(_CUDA_BUILD_PACKAGES))
        raise ValueError(f"Unsupported CUDA major version {major!r}; expected one of: {supported}")

    return value


def _config_cuda_version(config_settings: ConfigSettings) -> Optional[str]:
    """Get the CUDA version from PEP 517 config settings."""
    settings: Mapping[str, ConfigValue] = config_settings or {}
    value = settings.get("cuda-version")
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            raise ValueError("CUDA package version cannot be empty")
        value = value[-1]

    return _normalize_cuda_version(str(value))


def _environment_cuda_version() -> Optional[str]:
    """Get the CUDA version from the environment."""
    value = os.getenv("NVTE_CUDA_VERSION")
    return _normalize_cuda_version(value) if value is not None else None


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
    settings: Mapping[str, ConfigValue] = config_settings or {}
    value = settings.get("framework")
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            raise ValueError("Framework selection cannot be empty")
        value = value[-1]

    return _normalize_frameworks(str(value))


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
    settings: Mapping[str, ConfigValue] = config_settings or {}
    value = settings.get(setting)
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            raise ValueError(f"{setting} cannot be empty")
        value = value[-1]

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
    return _normalize_cuda_version(match.group(1)) if match is not None else None


def _torch_cuda_version() -> Optional[str]:
    """Get the CUDA version required by PyTorch, if available."""
    try:
        import torch
    except (ImportError, OSError):
        return None
    value = getattr(getattr(torch, "version", None), "cuda", None)
    return _normalize_cuda_version(value) if value else None


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
        _config_cuda_version(config_settings)
        or _environment_cuda_version()
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
    """Get build requirements for the requested CUDA package version."""
    cuda_version = _cuda_version(config_settings)
    cuda_major = cuda_version.split(".", maxsplit=1)[0]
    component_constraint = f"=={cuda_version}.*"

    packages = _CUDA_BUILD_PACKAGES[cuda_major]
    requirements = [f"{package}{component_constraint}" for package in packages]
    requirements.append(f"nvidia-nccl-cu{cuda_major}>=2")
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


###################################################################################################
# PEP 517 and PEP 660 defined functions
###################################################################################################


# Defined by PEP 517
def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> List[str]:
    """Get requirements for building a wheel."""
    return _build_requirements(config_settings)


# Defined by PEP 517
def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> List[str]:
    """Get requirements needed while evaluating setup.py for an sdist."""
    # Build requirements are needed even for sdist because setup.py currently always depends
    # on them. It could be refactored to only depend on them when building a wheel, but that
    # would require a more invasive refactor.
    return _build_requirements(config_settings)


# Defined by PEP 517
def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """Build a wheel with setuptools."""
    with _build_environment(config_settings):
        return _SETUPTOOLS_BACKEND.build_wheel(
            wheel_directory,
            _setuptools_config_settings(config_settings),
            metadata_directory,
        )


# Defined by PEP 517
def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: ConfigSettings = None,
) -> str:
    """Prepare wheel metadata with setuptools."""
    with _build_environment(config_settings):
        return _SETUPTOOLS_BACKEND.prepare_metadata_for_build_wheel(
            metadata_directory,
            _setuptools_config_settings(config_settings),
        )


# Defined by PEP 517
def build_sdist(
    sdist_directory: str,
    config_settings: ConfigSettings = None,
) -> str:
    """Build an sdist with setuptools."""
    with _build_environment(config_settings):
        return _SETUPTOOLS_BACKEND.build_sdist(
            sdist_directory,
            _setuptools_config_settings(config_settings),
        )


if hasattr(_SETUPTOOLS_BACKEND, "build_editable"):

    # Defined by PEP 660
    def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> List[str]:
        """Get requirements for building an editable wheel."""
        return _build_requirements(config_settings)

    # Defined by PEP 660
    def build_editable(
        wheel_directory: str,
        config_settings: ConfigSettings = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        """Build an editable wheel with setuptools."""
        with _build_environment(config_settings):
            return _SETUPTOOLS_BACKEND.build_editable(
                wheel_directory,
                _setuptools_config_settings(config_settings),
                metadata_directory,
            )

    # Defined by PEP 660
    def prepare_metadata_for_build_editable(
        metadata_directory: str,
        config_settings: ConfigSettings = None,
    ) -> str:
        """Prepare editable-wheel metadata with setuptools."""
        with _build_environment(config_settings):
            return _SETUPTOOLS_BACKEND.prepare_metadata_for_build_editable(
                metadata_directory,
                _setuptools_config_settings(config_settings),
            )
