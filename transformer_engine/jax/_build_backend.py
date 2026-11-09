# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PEP 517 backend that builds against JAX with support for the requested CUDA version."""

import os
import re
from typing import List, Mapping, Optional, Union

from setuptools import build_meta

_SETUPTOOLS_BACKEND = build_meta.__legacy__
_SUPPORTED_CUDA_MAJOR_VERSIONS = ("12", "13")
# Isolated builds have always used JAX with CUDA 12 support.
_DEFAULT_CUDA_MAJOR_VERSION = "12"

ConfigValue = Union[str, List[str]]
ConfigSettings = Optional[Mapping[str, ConfigValue]]


def _setuptools_config_settings(config_settings: ConfigSettings) -> ConfigSettings:
    """Remove settings that are private to this backend."""
    if config_settings is None:
        return None

    settings = dict(config_settings)
    settings.pop("cuda-version", None)
    return settings or None


def _cuda_major_version(config_settings: ConfigSettings) -> str:
    """Get the CUDA major version from config settings or NVTE_CUDA_VERSION."""
    settings: Mapping[str, ConfigValue] = config_settings or {}
    value = settings.get("cuda-version")
    if isinstance(value, list):
        if not value:
            raise ValueError("CUDA version cannot be empty")
        value = value[-1]
    if value is None:
        value = os.getenv("NVTE_CUDA_VERSION")
    if value is None:
        return _DEFAULT_CUDA_MAJOR_VERSION

    value = str(value).lower().removeprefix("cu")
    if re.fullmatch(r"\d+\.\d+", value) is None:
        raise ValueError(
            f"Invalid CUDA version {value!r}; expected <major>.<minor>, for example 13.0"
        )

    major = value.split(".", maxsplit=1)[0]
    if major not in _SUPPORTED_CUDA_MAJOR_VERSIONS:
        supported = ", ".join(_SUPPORTED_CUDA_MAJOR_VERSIONS)
        raise ValueError(f"Unsupported CUDA major version {major!r}; expected one of: {supported}")

    return major


def _build_requirements(config_settings: ConfigSettings) -> List[str]:
    """Get JAX with support for the requested CUDA version."""
    # setup.py detects the CUDA version of the installed JAX to select the core library.
    return [f"jax[cuda{_cuda_major_version(config_settings)}]"]


###################################################################################################
# PEP 517 and PEP 660 defined functions
###################################################################################################

# The setuptools hooks for build requirements are not used because they run setup.py, which
# imports JAX before it is installed.


# Defined by PEP 517
def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> List[str]:
    """Get requirements for building a wheel."""
    return _build_requirements(config_settings)


# Defined by PEP 517
def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> List[str]:
    """Get requirements needed while evaluating setup.py for an sdist."""
    return _build_requirements(config_settings)


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
        """Get requirements for building an editable wheel."""
        return _build_requirements(config_settings)

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
