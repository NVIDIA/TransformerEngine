# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Version information"""
import sys
from packaging.version import Version

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

def _version_str() -> str:
    """Transformer Engine version string"""

    # Try getting version from package metadata
    version_str = None
    try:
        version_str = metadata.version("transformer_engine")
    except:
        pass
    if version_str:
        return version_str

    # Try getting version from Git root directory
    try:
        from te_version import te_version
        version_str = te_version()
    except:
        pass
    if version_str:
        return version_str

    # Could not deduce version
    return "0.dev0+unknown"

# Transformer Engine version
__version__: Version = Version(_version_str())
