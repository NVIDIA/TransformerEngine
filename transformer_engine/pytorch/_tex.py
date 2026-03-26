# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unified routing module for transformer_engine_torch.

Imports from pybind11 .so as the base, then selectively overrides with
stable ABI implementations for ops that have been validated.

Usage: import transformer_engine.pytorch._tex as tex
"""

# Base: everything from pybind11 .so
from transformer_engine_torch import *  # noqa: F401,F403

# The quantizer classes (float8_tensor.py, etc.) have been patched directly
# to call stable ABI ops via _quantize_stable.py. No overrides needed here.

# Future: as more ops are validated, they can be overridden here.
# For now, the pybind11 .so provides all functions and the stable ABI
# is used internally by the quantizer classes.
