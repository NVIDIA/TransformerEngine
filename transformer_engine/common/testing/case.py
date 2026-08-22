# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""The Case contract returned by a benchmarkable test, and its axis_value renderer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import enum
import os
from typing import Any

from .declaration import plugin_active


def _stable_name(value: Any) -> str | None:
    """Return ``value``'s ``__qualname__``, else its ``__name__``, else ``None``."""
    for attribute in ("__qualname__", "__name__"):
        name = getattr(value, attribute, None)
        if isinstance(name, str):
            return name
    return None


def axis_value(value: Any) -> Any:
    """Render one parametrize value canonically for artifact identity."""
    # The enum check must precede the int check: IntEnum is an int.
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, type):
        return value.__qualname__
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    name = _stable_name(value)
    if name is not None:
        return name
    rendered = str(value)
    # An address-bearing rendering is not stable across processes: reject it.
    if " at 0x" in rendered:
        raise ValueError(
            f"Benchmark axis value {rendered!r} has no stable rendering: its repr embeds "
            "a memory address, so no two runs would agree on its case_id. Parametrize "
            "this axis with an enum or a named function/class instead of a bare object."
        )
    return rendered


class CaseSkip(Exception):
    """Raised by a Case callable when a feature is unavailable; the plugin skips."""


def _require_plugin() -> None:
    """Refuse to build a Case inside a pytest session that has no plugin to run it."""
    # pytest sets PYTEST_CURRENT_TEST only while a test is executing, so this fires
    # exactly when a Case would be discarded and its test pass having verified nothing.
    # It lives here, not in the decorators, because a Case-bearing test need not be
    # decorated at all, and a class-level decorator never wraps its methods.
    current = os.environ.get("PYTEST_CURRENT_TEST")
    if not current or plugin_active():
        return
    raise RuntimeError(
        f"{current} built a Case, but the nvte-benchmark pytest plugin is not registered,"
        " so nothing would run it and this test asserts nothing. It autoloads from"
        " transformer_engine's entry point, so reinstall transformer_engine (nvte-setup)"
        " if pytest is not picking it up, and drop any '-p no:nvte-benchmark'."
    )


@dataclass
class Case:
    """What a benchmarkable test returns: ``setup`` builds an opaque state object,
    ``evaluate`` is the Transformer Engine path, and ``reference`` is the comparison
    target in correctness mode and a timed baseline in benchmark mode.
    """

    setup: Callable[[], Any]
    evaluate: Callable[[Any], Any]
    reference: Callable[[Any], Any] | None = None
    verify: Callable[[Any, Any], None] | None = None
    reset: Callable[[Any], None] | None = None
    batchable: bool = True
    time_reference: bool = True
    bytes_moved: int | None = None
    flops: int | None = None

    # Setting `reset` implies not batchable: ``runner.py`` pins inner_iterations to 1.

    def __post_init__(self) -> None:
        """Validate the plugin is present to run this Case, and that ``reference`` and
        ``verify`` are set together."""
        _require_plugin()
        if self.reference is None and self.verify is not None:
            raise ValueError("Case defines verify but no reference to compare against.")
        if self.reference is not None and self.verify is None:
            raise ValueError(
                "Case defines a reference but no verify, and there is no default "
                "comparator. Pass a verify built on the framework's tolerance helper "
                "(tests/pytorch/utils.py::dtype_tols, tests/jax/utils.py::assert_allclose)."
            )

    def run_verify(self, actual: Any, expected: Any) -> None:
        """Compare one evaluate output against one reference output."""
        self.verify(actual, expected)
