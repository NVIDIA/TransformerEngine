# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Vendor-specific backend implementations.

This package contains hardware vendor-specific backend implementations
for TransformerEngine-FL. Each vendor subdirectory should contain its
own backend implementation.
"""

from __future__ import annotations

import os

_vendor_loading_errors = []

try:
    from ..._build_config import SKIP_CUDA_BUILD as _SKIP_CUDA_BUILD_CONFIG
except ImportError:
    _SKIP_CUDA_BUILD_CONFIG = bool(int(os.environ.get("TE_FL_SKIP_CUDA", "0")))
    print(f"Build config not found, using env var: SKIP_CUDA_BUILD={_SKIP_CUDA_BUILD_CONFIG}")

if os.environ.get("TE_FL_SKIP_CUDA"):
    _SKIP_CUDA_BUILD = bool(int(os.environ.get("TE_FL_SKIP_CUDA", "0")))
else:
    _SKIP_CUDA_BUILD = _SKIP_CUDA_BUILD_CONFIG

if not _SKIP_CUDA_BUILD:
    try:
        from .cuda import CUDABackend
    except ImportError as e:
        _vendor_loading_errors.append(("cuda", "ImportError", str(e)))
        print(f"Failed to import CUDA vendor backend: {e}")
    except Exception as e:
        _vendor_loading_errors.append(("cuda", type(e).__name__, str(e)))
        print(f"Error loading CUDA vendor backend: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("CUDA vendor backend skipped (CUDA build was disabled at build time)")
    _vendor_loading_errors.append(("cuda", "Skipped", "CUDA build was disabled at build time"))


def get_vendor_loading_errors():
    """Get errors that occurred during vendor backend loading."""
    return _vendor_loading_errors.copy()


__all__ = ["get_vendor_loading_errors"]
