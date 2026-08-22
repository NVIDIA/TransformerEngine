# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Storage for benchmark axis declarations, kept free of pytest imports."""

from __future__ import annotations

from typing import Any

DECLARATION_ATTR = "_te_benchmark_axes"

_PLUGIN_ACTIVE = False
"""Set true by ``plugin.py``'s ``pytest_configure`` once the plugin is registered."""


def set_plugin_active(active: bool) -> None:
    """Record whether the benchmarkable pytest plugin is registered in this session."""
    global _PLUGIN_ACTIVE  # pylint: disable=global-statement
    _PLUGIN_ACTIVE = bool(active)


def plugin_active() -> bool:
    """Return whether the benchmarkable pytest plugin registered itself this session."""
    return _PLUGIN_ACTIVE


def normalize_argnames(argnames: Any) -> str:
    """Return a canonical comma-joined key with no spaces, so a declaration and a
    ``pytest.mark.parametrize`` mark written with different spacing still match."""
    if isinstance(argnames, str):
        names = [name.strip() for name in argnames.split(",") if name.strip()]
    else:
        names = [str(name).strip() for name in argnames]
    return ",".join(names)


def record_axis(holder: Any, argnames: Any, values: Any) -> None:
    """Attach one benchmark axis declaration to a function, method, or class."""
    declarations = holder.__dict__.get(DECLARATION_ATTR)
    if declarations is None:
        declarations = {}
        setattr(holder, DECLARATION_ATTR, declarations)
    declarations[normalize_argnames(argnames)] = list(values)


def declared_axes(function: Any) -> dict[str, list]:
    """Return the benchmark axis declarations attached to one test function."""
    return dict(getattr(function, DECLARATION_ATTR, None) or {})
