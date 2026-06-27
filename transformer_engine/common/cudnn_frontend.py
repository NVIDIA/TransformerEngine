# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utilities for using the vendored cuDNN frontend Python bindings."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Optional

from packaging.version import Version as PkgVersion


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vendored_python_path() -> Path:
    """Path to the vendored cuDNN frontend Python package in source checkouts."""
    return _repo_root() / "3rdparty" / "cudnn-frontend" / "python"


def _vendored_cudnn_package() -> Path:
    return vendored_python_path() / "cudnn"


def vendored_python_package_is_built() -> bool:
    """Whether the vendored Python package has a built extension module."""
    cudnn_package = _vendored_cudnn_package()
    return cudnn_package.is_dir() and any(cudnn_package.glob("_compiled_module*"))


def import_cudnn_frontend() -> ModuleType:
    """Import cuDNN frontend, preferring the built vendored submodule package."""
    if vendored_python_package_is_built():
        path = str(vendored_python_path())
        if path not in sys.path:
            sys.path.insert(0, path)

    if importlib.util.find_spec("cudnn") is not None:
        cudnn = importlib.import_module("cudnn")
        check_cudnn_frontend_vendored_version_match(cudnn)
        return cudnn

    raise ImportError(
        "cuDNN frontend Python package not found. Build it from Transformer Engine's vendored "
        "submodule with: python -m build_tools.cudnn_frontend install"
    )


def cudnn_frontend_version(cudnn: Optional[ModuleType] = None) -> PkgVersion:
    """Return the imported cuDNN frontend Python package version."""
    module = import_cudnn_frontend() if cudnn is None else cudnn
    version = getattr(module, "__version__", None)
    if version is None:
        raise RuntimeError("cuDNN frontend Python package does not expose __version__.")
    return PkgVersion(str(version))


def cudnn_frontend_version_at_least(min_version: str) -> bool:
    """Whether the imported cuDNN frontend Python package is at least ``min_version``."""
    try:
        return cudnn_frontend_version() >= PkgVersion(min_version)
    except ImportError:
        return False


def check_cudnn_frontend_vendored_version_match(cudnn: ModuleType) -> None:
    """Validate imported Python package version against vendored source when available."""
    vendored_version = vendored_source_version()
    if vendored_version is None:
        return
    python_version = cudnn_frontend_version(cudnn)
    if python_version != vendored_version:
        raise RuntimeError(
            "cuDNN frontend Python package version mismatch: "
            f"imported cudnn.__version__={python_version}, but Transformer Engine vendors "
            f"cuDNN frontend {vendored_version}. Install cuDNN frontend from Transformer "
            "Engine's vendored submodule."
        )


def encode_cudnn_frontend_version(version: str) -> int:
    """Encode a cuDNN frontend Python version as CUDNN_FRONTEND_VERSION."""
    public_version = version.split("+", 1)[0].split("-", 1)[0]
    parts = public_version.split(".")
    if len(parts) < 3:
        raise RuntimeError(f"Could not parse cuDNN frontend Python version: {version!r}.")
    major, minor, patch = (int(part) for part in parts[:3])
    return major * 10000 + minor * 100 + patch


def check_cudnn_frontend_version_match(cudnn: ModuleType, cpp_version: int) -> int:
    """Validate that Python cuDNN frontend and C++ headers have matching versions."""
    python_version_string = getattr(cudnn, "__version__", None)
    if python_version_string is None:
        raise RuntimeError("cuDNN frontend Python package does not expose __version__.")
    python_version = encode_cudnn_frontend_version(str(python_version_string))
    if python_version != int(cpp_version):
        raise RuntimeError(
            "cuDNN frontend Python/C++ version mismatch: "
            f"Python cudnn.__version__={python_version_string!r} encodes to {python_version}, "
            f"but Transformer Engine C++ was built with CUDNN_FRONTEND_VERSION={cpp_version}. "
            "Install cuDNN frontend from Transformer Engine's vendored submodule."
        )
    return python_version


def vendored_source_version() -> Optional[PkgVersion]:
    """Read the vendored source version when running from a source checkout."""
    init_py = _vendored_cudnn_package() / "__init__.py"
    if not init_py.is_file():
        return None
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py.read_text(), re.M)
    if match is None:
        return None
    return PkgVersion(match.group(1))
