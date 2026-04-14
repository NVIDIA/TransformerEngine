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

# Minimum JAX version for safe input_output_aliases in TritonAutotunedKernelCall.
# jaxlib/gpu/triton_kernels.cc had a bug in the autotuning save/restore loop:
# it iterated over all declared aliases unconditionally, but input_copies only
# contains entries for aliases where XLA actually shared buffers at runtime.
# Accessing a missing entry produced a null vector → CUDA_ERROR_INVALID_VALUE.
# Fixed by: https://github.com/jax-ml/jax/pull/35218 (committed 2026-03-10 on jax-ml/jax main;
# first published nightly container: jax-2026-03-17). Ships in JAX 0.9.3 (stable).
# Nightly containers report "0.9.2devYYYYMMDD", which packaging.version treats as a
# pre-release of 0.9.2 (less than 0.9.2 stable, greater than any 0.9.2.devN).
# Using "0.9.2.dev20260317" as the floor accepts post-fix nightlies while correctly
# rejecting pre-fix builds (0.9.2.dev20260316 and earlier).
TRITON_AUTOTUNED_INPUT_OUTPUT_ALIAS_MIN_JAX_VERSION = "0.9.2.dev20260317"


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
    "TRITON_AUTOTUNED_INPUT_OUTPUT_ALIAS_MIN_JAX_VERSION",
]
