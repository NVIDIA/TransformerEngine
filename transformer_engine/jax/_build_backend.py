# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PEP 517 backend that builds against JAX with support for the requested CUDA version."""

from pathlib import Path
import sys
from typing import List

# Like setup.py, prefer the repository's build_tools in a source checkout. The sdist contains
# a copy next to this file.
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
if (_REPOSITORY_ROOT / "build_tools").is_dir():
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# pylint: disable=wrong-import-position
from build_tools.build_backend_utils import (
    ConfigSettings,
    requested_cuda_version,
    setuptools_hooks,
)

# Isolated builds have always used JAX with CUDA 12 support.
_DEFAULT_CUDA_MAJOR_VERSION = "12"


def _build_requirements(config_settings: ConfigSettings) -> List[str]:
    """Get JAX with support for the requested CUDA version, and Flax."""
    # setup.py detects the CUDA version of the installed JAX to select the core library.
    cuda_version = requested_cuda_version(config_settings)
    cuda_major = (
        cuda_version.split(".", maxsplit=1)[0] if cuda_version else _DEFAULT_CUDA_MAJOR_VERSION
    )
    # Flax depends on JAX, so it must not be a static requirement: pip skips a dynamic
    # requirement whose package is already installed, even if the requirement adds extras.
    return [f"jax[cuda{cuda_major}]", "flax>=0.7.1"]


# PEP 517 and PEP 660 hooks
globals().update(
    setuptools_hooks(requirements=_build_requirements, private_settings=("cuda-version",))
)
