# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level package"""

# pylint: disable=unused-import

import ctypes
import functools
import os
from importlib import metadata
import transformer_engine.common

# Minimum NCCL version for the statically-linked NCCL EP backend.
_NCCL_EP_MIN_VERSION = (2, 30, 4)


@functools.lru_cache(maxsize=1)
def is_nccl_ep_available() -> bool:
    """Return True if the runtime libnccl.so is new enough for NCCL EP."""
    try:
        libnccl = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
        ver = ctypes.c_int(0)
        libnccl.ncclGetVersion(ctypes.byref(ver))
    except (OSError, AttributeError):
        return False
    v = ver.value
    cur = (v // 10000, (v // 100) % 100, v % 100)
    return cur >= _NCCL_EP_MIN_VERSION


def require_nccl_ep() -> None:
    """Raise RuntimeError if NCCL EP cannot run on the current libnccl."""
    if not is_nccl_ep_available():
        mn = ".".join(str(x) for x in _NCCL_EP_MIN_VERSION)
        raise RuntimeError(
            f"NCCL EP requires NCCL >= {mn} at runtime; upgrade libnccl.so or "
            "rebuild Transformer Engine with NVTE_BUILD_WITH_NCCL_EP=0."
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
