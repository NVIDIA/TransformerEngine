# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared helpers for Transformer Engine's PEP 517 build backends.

Only the standard library and setuptools may be imported here, since the backends import
this module before any other build requirements are installed.
"""

from contextlib import nullcontext
import os
import re
from typing import Callable, ContextManager, Dict, Iterable, List, Mapping, Optional, Union

from setuptools import build_meta

SUPPORTED_CUDA_MAJOR_VERSIONS = ("12", "13")

ConfigValue = Union[str, List[str]]
ConfigSettings = Optional[Mapping[str, ConfigValue]]


def config_value(config_settings: ConfigSettings, setting: str) -> Optional[str]:
    """Get a PEP 517 config setting, using the last value if it was given more than once."""
    settings: Mapping[str, ConfigValue] = config_settings or {}
    value = settings.get(setting)
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            raise ValueError(f"{setting} cannot be empty")
        value = value[-1]

    return str(value)


def normalize_cuda_version(value: str) -> str:
    """Validate and normalize a CUDA major/minor version."""
    value = value.lower().removeprefix("cu")
    if re.fullmatch(r"\d+\.\d+", value) is None:
        raise ValueError(
            f"Invalid CUDA version {value!r}; expected <major>.<minor>, for example 12.8"
        )

    major = value.split(".", maxsplit=1)[0]
    if major not in SUPPORTED_CUDA_MAJOR_VERSIONS:
        supported = ", ".join(SUPPORTED_CUDA_MAJOR_VERSIONS)
        raise ValueError(f"Unsupported CUDA major version {major!r}; expected one of: {supported}")

    return value


def requested_cuda_version(config_settings: ConfigSettings) -> Optional[str]:
    """Get the CUDA version from the cuda-version config setting or NVTE_CUDA_VERSION."""
    value = config_value(config_settings, "cuda-version")
    if value is None:
        value = os.getenv("NVTE_CUDA_VERSION")
    return normalize_cuda_version(value) if value is not None else None


def setuptools_hooks(
    requirements: Callable[[ConfigSettings], List[str]],
    private_settings: Iterable[str],
    environment: Optional[Callable[[ConfigSettings], ContextManager]] = None,
) -> Dict[str, Callable]:
    """Create PEP 517 and PEP 660 hooks that build with the legacy setuptools backend.

    Build requirements come only from `requirements`. The setuptools hooks for build
    requirements are not used because they run setup.py, which can import build
    requirements before they are installed. Private settings are removed before the
    config settings are passed to setuptools, and builds run inside `environment`.
    """
    backend = build_meta.__legacy__
    private_settings = frozenset(private_settings)
    if environment is None:
        environment = lambda config_settings: nullcontext()

    def setuptools_config_settings(config_settings: ConfigSettings) -> ConfigSettings:
        if config_settings is None:
            return None
        settings = {
            setting: value
            for setting, value in config_settings.items()
            if setting not in private_settings
        }
        return settings or None

    def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> List[str]:
        """Get requirements for building a wheel."""
        return requirements(config_settings)

    def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> List[str]:
        """Get requirements needed while evaluating setup.py for an sdist."""
        return requirements(config_settings)

    def build_wheel(
        wheel_directory: str,
        config_settings: ConfigSettings = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        """Build a wheel with setuptools."""
        with environment(config_settings):
            return backend.build_wheel(
                wheel_directory, setuptools_config_settings(config_settings), metadata_directory
            )

    def prepare_metadata_for_build_wheel(
        metadata_directory: str, config_settings: ConfigSettings = None
    ) -> str:
        """Prepare wheel metadata with setuptools."""
        with environment(config_settings):
            return backend.prepare_metadata_for_build_wheel(
                metadata_directory, setuptools_config_settings(config_settings)
            )

    def build_sdist(sdist_directory: str, config_settings: ConfigSettings = None) -> str:
        """Build an sdist with setuptools."""
        with environment(config_settings):
            return backend.build_sdist(sdist_directory, setuptools_config_settings(config_settings))

    hooks = {
        "get_requires_for_build_wheel": get_requires_for_build_wheel,
        "get_requires_for_build_sdist": get_requires_for_build_sdist,
        "build_wheel": build_wheel,
        "prepare_metadata_for_build_wheel": prepare_metadata_for_build_wheel,
        "build_sdist": build_sdist,
    }

    if hasattr(backend, "build_editable"):

        def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> List[str]:
            """Get requirements for building an editable wheel."""
            return requirements(config_settings)

        def build_editable(
            wheel_directory: str,
            config_settings: ConfigSettings = None,
            metadata_directory: Optional[str] = None,
        ) -> str:
            """Build an editable wheel with setuptools."""
            with environment(config_settings):
                return backend.build_editable(
                    wheel_directory, setuptools_config_settings(config_settings), metadata_directory
                )

        def prepare_metadata_for_build_editable(
            metadata_directory: str, config_settings: ConfigSettings = None
        ) -> str:
            """Prepare editable-wheel metadata with setuptools."""
            with environment(config_settings):
                return backend.prepare_metadata_for_build_editable(
                    metadata_directory, setuptools_config_settings(config_settings)
                )

        hooks.update(
            get_requires_for_build_editable=get_requires_for_build_editable,
            build_editable=build_editable,
            prepare_metadata_for_build_editable=prepare_metadata_for_build_editable,
        )

    return hooks
