# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Minimum versions of Transformer Engine dependencies.

The build loads this file directly, without importing the transformer_engine package, so it
must not import anything.
"""

# Minimum cuDNN version supported by Transformer Engine, as (major, minor, patch).
# Keep in sync with kMinCudnnVersion in transformer_engine/common/cudnn_min_version.h.
MIN_CUDNN_VERSION = (9, 12, 0)
