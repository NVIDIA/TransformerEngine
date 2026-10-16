# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level package"""

# pylint: disable=unused-import

import ctypes
import functools
import os
from ctypes.util import find_library
from importlib import metadata
from pathlib import Path
from typing import Optional, Tuple
import transformer_engine.common

# Minimum NCCL version for the runtime-loaded NCCL EP backend.
_NCCL_EP_MIN_VERSION = (2, 30, 4)


@functools.lru_cache(maxsize=1)
def _nccl_runtime_version() -> Optional[Tuple[int, int, int]]:
    """Return runtime (major, minor, patch) from libnccl.so.2, or None if unavailable."""
    try:
        libnccl = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_LOCAL)
        ncclGetVersion = libnccl.ncclGetVersion
    except (OSError, AttributeError):
        return None
    ver = ctypes.c_int(0)
    if ncclGetVersion(ctypes.byref(ver)) != 0:
        return None
    v = ver.value
    return (v // 10000, (v // 100) % 100, v % 100)


def _nccl_ep_library_installed() -> bool:
    if nccl_ep_home := os.getenv("NCCL_EP_HOME"):
        home = Path(nccl_ep_home)
        if any(
            library.is_file()
            for lib_dir in ("lib", "lib64")
            for library in (home / lib_dir).glob("libnccl_ep.so*")
        ):
            return True

    try:
        transformer_engine.common._get_shared_object_file("nccl_ep")
    except FileNotFoundError:
        return find_library("nccl_ep") is not None
    else:
        return True


def is_nccl_ep_available() -> bool:
    """Return True if the NCCL EP library and a compatible NCCL runtime are available."""
    if not _nccl_ep_library_installed():
        return False
    cur = _nccl_runtime_version()
    return cur is not None and cur >= _NCCL_EP_MIN_VERSION


def require_nccl_ep() -> None:
    """Raise RuntimeError if NCCL EP cannot run on the current libnccl."""
    if not _nccl_ep_library_installed():
        raise RuntimeError(
            "NCCL EP library libnccl_ep.so is not installed. Build Transformer Engine "
            "with NVTE_WITH_NCCL_EP=1."
        )
    mn = ".".join(str(x) for x in _NCCL_EP_MIN_VERSION)
    cur = _nccl_runtime_version()
    if cur is None:
        raise RuntimeError(
            f"NCCL EP requires NCCL >= {mn}; could not load libnccl.so.2 or query its "
            "version. Install NCCL or ensure libnccl.so.2 is on the loader path."
        )
    if cur < _NCCL_EP_MIN_VERSION:
        raise RuntimeError(
            f"NCCL EP requires NCCL >= {mn} at runtime; found "
            f"{'.'.join(str(x) for x in cur)}. Upgrade NCCL to a compatible version."
        )


try:
    from . import pytorch
except ImportError:
    pass
except FileNotFoundError as e:
    if "Could not find shared object file" not in str(e):
        raise e  # Unexpected error
    else:
        if os.getenv("NVTE_FRAMEWORK"):
            frameworks = os.getenv("NVTE_FRAMEWORK").split(",")
            if "pytorch" in frameworks or "all" in frameworks:
                raise e
        else:
            # If we got here, we could import `torch` but could not load the framework extension.
            # This can happen when a user wants to work only with `transformer_engine.jax` on a system that
            # also has a PyTorch installation. In order to enable that use case, we issue a warning here
            # about the missing PyTorch extension in case the user hasn't set NVTE_FRAMEWORK.
            import warnings

            warnings.warn(
                "Detected a PyTorch installation but could not find the shared object file for the "
                "Transformer Engine PyTorch extension library. If this is not intentional, please "
                "reinstall Transformer Engine with `pip install transformer_engine[pytorch]` or "
                "build from source with `NVTE_FRAMEWORK=pytorch`.",
                category=RuntimeWarning,
            )

try:
    from . import jax
except ImportError:
    pass
except FileNotFoundError as e:
    if "Could not find shared object file" not in str(e):
        raise e  # Unexpected error
    else:
        if os.getenv("NVTE_FRAMEWORK"):
            frameworks = os.getenv("NVTE_FRAMEWORK").split(",")
            if "jax" in frameworks or "all" in frameworks:
                raise e
        else:
            # If we got here, we could import `jax` but could not load the framework extension.
            # This can happen when a user wants to work only with `transformer_engine.pytorch` on a system
            # that also has a Jax installation. In order to enable that use case, we issue a warning here
            # about the missing Jax extension in case the user hasn't set NVTE_FRAMEWORK.
            import warnings

            warnings.warn(
                "Detected a Jax installation but could not find the shared object file for the "
                "Transformer Engine Jax extension library. If this is not intentional, please "
                "reinstall Transformer Engine with `pip install transformer_engine[jax]` or "
                "build from source with `NVTE_FRAMEWORK=jax`.",
                category=RuntimeWarning,
            )

__version__ = str(metadata.version("transformer_engine"))
