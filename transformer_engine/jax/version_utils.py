# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
JAX version helpers.

Provides version checks for JAX that can be used across TE JAX (quantize, triton
extensions, etc.) without pulling in feature-specific code.
"""

from functools import lru_cache
from importlib.metadata import version as get_pkg_version

from packaging.version import Version as PkgVersion


@lru_cache(maxsize=None)
def jax_version_meet_requirement(version: str):
    """Return True if the installed JAX version is >= the required version."""
    jax_version = PkgVersion(get_pkg_version("jax"))
    jax_version_required = PkgVersion(version)
    return jax_version >= jax_version_required


# Minimum JAX version required for Triton kernel dispatch (jaxlib < 0.8.0 segfaults).
TRITON_EXTENSION_MIN_JAX_VERSION = "0.8.0"


def is_triton_extension_supported() -> bool:
    """Return True if the current JAX version supports Triton kernel dispatch.

    JAX/jaxlib >= 0.8.0 is required. Older versions segfault when dispatching
    Triton kernels. Use this to skip tests or gate features without importing
    triton_extensions (which would raise immediately on old jax).
    """
    return jax_version_meet_requirement(TRITON_EXTENSION_MIN_JAX_VERSION)


__all__ = [
    "jax_version_meet_requirement",
    "is_triton_extension_supported",
    "TRITON_EXTENSION_MIN_JAX_VERSION",
]
