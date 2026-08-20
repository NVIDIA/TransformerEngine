# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PEP 517 backend with CUDA-version-specific build requirements."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Mapping, Optional, Union

from setuptools import build_meta


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
_DEFAULT_CUDA_VERSION = "13.3"

ConfigValue = Union[str, List[str]]
ConfigSettings = Optional[Mapping[str, ConfigValue]]


def _setuptools_config_settings(config_settings: ConfigSettings) -> ConfigSettings:
    """Remove settings that are private to this backend."""
    if config_settings is None:
        return None

    settings = dict(config_settings)
    settings.pop("cuda-version", None)
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


def _system_cuda_version() -> Optional[str]:
    """Get the version reported by a system-installed NVCC."""
    candidates = []
    if cuda_compiler := os.getenv("CUDACXX"):
        candidates.append(Path(cuda_compiler))
    if cuda_home := os.getenv("CUDA_HOME"):
        candidates.append(Path(cuda_home) / "bin" / "nvcc")
    if nvcc := shutil.which("nvcc"):
        candidates.append(Path(nvcc))
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))

    checked = set()
    for candidate in candidates:
        if candidate in checked or not candidate.is_file():
            continue

        checked.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                check=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue

        match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
        if match is not None:
            return _normalize_cuda_version(match.group(1))

    return None


def _torch_cuda_version() -> Optional[str]:
    """Get the CUDA version required by PyTorch, if available."""
    try:
        import torch
    except (ImportError, OSError):
        return None
    value = getattr(getattr(torch, "version", None), "cuda", None)
    return _normalize_cuda_version(value) if value else None


def _framework_cuda_version() -> Optional[str]:
    """Get a CUDA version required by the selected framework."""
    frameworks = {
        framework.strip().lower()
        for framework in os.getenv("NVTE_FRAMEWORK", "").split(",")
        if framework.strip()
    }

    if not frameworks or frameworks.intersection({"all", "pytorch"}):
        if cuda_version := _torch_cuda_version():
            return cuda_version

    # JAX CUDA plugin names identify the CUDA major version, but not the
    # major/minor toolkit release needed to select component packages.
    return None


def _cuda_version(config_settings: ConfigSettings) -> str:
    """Resolve the CUDA version in descending order of precedence."""
    return (
        _config_cuda_version(config_settings)
        or _environment_cuda_version()
        or _system_cuda_version()
        or _framework_cuda_version()
        or _DEFAULT_CUDA_VERSION
    )


def _cuda_build_requirements(config_settings: ConfigSettings) -> List[str]:
    """Get build requirements for the requested CUDA package version."""
    cuda_version = _cuda_version(config_settings)
    cuda_major = cuda_version.split(".", maxsplit=1)[0]
    component_constraint = f"=={cuda_version}.*"

    packages = _CUDA_BUILD_PACKAGES[cuda_major]
    requirements = [f"{package}{component_constraint}" for package in packages]
    requirements.append(f"nvidia-nccl-cu{cuda_major}>=2")
    return requirements


###################################################################################################
# PEP 517 and PEP 660 defined functions
###################################################################################################


# Defined by PEP 517
def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> List[str]:
    """Get CUDA requirements for building a wheel."""
    return _cuda_build_requirements(config_settings)


# Defined by PEP 517
def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> List[str]:
    """Get CUDA requirements needed while evaluating setup.py for an sdist."""
    # CUDA build requirements are needed even for sdist because setup.py currently always
    # depends on these packages. It could be refactored to only depend on them when building
    # a wheel, but that would require a more invasive refactor.
    return _cuda_build_requirements(config_settings)


# Defined by PEP 517
def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """Build a wheel with setuptools."""
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
    return _SETUPTOOLS_BACKEND.build_sdist(
        sdist_directory,
        _setuptools_config_settings(config_settings),
    )


if hasattr(_SETUPTOOLS_BACKEND, "build_editable"):

    # Defined by PEP 660
    def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> List[str]:
        """Get CUDA requirements for building an editable wheel."""
        return _cuda_build_requirements(config_settings)

    # Defined by PEP 660
    def build_editable(
        wheel_directory: str,
        config_settings: ConfigSettings = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        """Build an editable wheel with setuptools."""
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
        return _SETUPTOOLS_BACKEND.prepare_metadata_for_build_editable(
            metadata_directory,
            _setuptools_config_settings(config_settings),
        )
