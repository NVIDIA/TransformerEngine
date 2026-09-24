# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unified correctness and benchmarking test utilities."""

from .case import Case, CaseSkip

__all__ = ["Case", "CaseSkip", "benchmark"]  # pylint: disable=undefined-all-variable


def __getattr__(name):
    """Resolve ``benchmark`` lazily so importing this package does not require pytest."""
    if name == "benchmark":
        from .decorator import benchmark as _benchmark

        return _benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
