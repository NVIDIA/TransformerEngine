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

# Nightly and stable floors for safe input_output_aliases in TritonAutotunedKernelCall.
# jaxlib/gpu/triton_kernels.cc had a bug in the autotuning save/restore loop:
# it iterated over all declared aliases unconditionally, but input_copies only
# contains entries for aliases where XLA actually shared buffers at runtime.
# Accessing a missing entry produced a null vector → CUDA_ERROR_INVALID_VALUE.
# Fixed by: https://github.com/jax-ml/jax/pull/35218 (committed 2026-03-10 on jax-ml/jax main;
# first published nightly container: jax-2026-03-17). Ships in JAX 0.9.3 (stable).
#
# Two separate floors are required because packaging.version always ranks a stable
# release above any pre-release of the same series: PkgVersion("0.9.2") >
# PkgVersion("0.9.2.dev20260317"), so a single ">= 0.9.2.dev20260317" check would
# incorrectly accept 0.9.2 stable, which does NOT contain the fix.
#
#   nightly build  (v.dev is not None): safe if >= 0.9.2.dev20260317
#   stable release (v.dev is None):     safe if >= 0.9.3
_TRITON_AUTOTUNED_ALIAS_NIGHTLY_FLOOR = "0.9.2.dev20260317"
_TRITON_AUTOTUNED_ALIAS_STABLE_FLOOR = "0.9.3"

# Legacy single-constant kept for external callers; reflects the stable floor.
TRITON_AUTOTUNED_INPUT_OUTPUT_ALIAS_MIN_JAX_VERSION = _TRITON_AUTOTUNED_ALIAS_STABLE_FLOOR


@lru_cache(maxsize=None)
def is_triton_autotuned_alias_safe() -> bool:
    """Return True if the installed JAX safely supports input_output_aliases on autotuned calls.

    Uses two separate floors (jax-ml/jax#35218):
    - nightly builds: >= 0.9.2.dev20260317 (first container with the fix)
    - stable releases: >= 0.9.3 (0.9.2 stable does not contain the fix)
    """
    v = PkgVersion(get_pkg_version("jax"))
    if v.dev is not None:
        return v >= PkgVersion(_TRITON_AUTOTUNED_ALIAS_NIGHTLY_FLOOR)
    return v >= PkgVersion(_TRITON_AUTOTUNED_ALIAS_STABLE_FLOOR)


def is_triton_extension_supported() -> bool:
    """Return True if the current JAX version supports Triton kernel dispatch.

    JAX/jaxlib >= 0.8.0 is required. Older versions segfault when dispatching
    Triton kernels. Use this to skip tests or gate features without importing
    triton_extensions (which would raise immediately on old jax).
    """
    return jax_version_meet_requirement(TRITON_EXTENSION_MIN_JAX_VERSION)


__all__ = [
    "jax_version_meet_requirement",
    "is_triton_autotuned_alias_safe",
    "is_triton_extension_supported",
    "TRITON_EXTENSION_MIN_JAX_VERSION",
    "TRITON_AUTOTUNED_INPUT_OUTPUT_ALIAS_MIN_JAX_VERSION",
]
