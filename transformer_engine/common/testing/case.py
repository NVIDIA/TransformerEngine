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


#: Budget for a whole multi-rank launch -- child startup, import, rendezvous, every
#: iteration and teardown -- when a Case does not set one. Deliberately generous: waiting
#: on a slow test costs time, cutting a valid one short costs a debugging session.
DEFAULT_TIMEOUT_S = 1800.0


@dataclass
class Case:
    """What a benchmarkable test returns: ``setup`` builds an opaque state object,
    ``evaluate`` is the Transformer Engine path, and ``reference`` is the comparison
    target in correctness mode and a timed baseline in benchmark mode.

    One ``state`` threads through every callable. ``dist_init`` seeds it, ``setup``
    fills in the test data, and ``dist_clean`` tears the communicator down. ``state`` is
    ``None`` throughout for a serial Case.
    """

    setup: Callable[[Any], Any]
    evaluate: Callable[[Any], Any]
    reference: Callable[[Any], Any] | None = None
    verify: Callable[[Any, Any], None] | None = None
    reset: Callable[[Any], None] | None = None
    dist_init: Callable[[], Any] | None = None
    dist_clean: Callable[[Any], None] | None = None
    #: Must block the host, not merely order the stream: a stream-ordered barrier folds
    #: rank skew back into the measured interval.
    barrier: Callable[[Any], None] | None = None
    num_gpus: int = 1
    timeout: float = DEFAULT_TIMEOUT_S
    batchable: bool = True
    time_reference: bool = True
    bytes_moved: int | None = None
    flops: int | None = None

    # Setting `reset` implies not batchable: ``runner.py`` pins inner_iterations to 1.

    def __post_init__(self) -> None:
        """Validate the plugin is present to run this Case, that ``reference`` and
        ``verify`` are set together, and that a multi-rank Case can form and align a
        process group."""
        _require_plugin()
        if self.reference is None and self.verify is not None:
            raise ValueError("Case defines verify but no reference to compare against.")
        if self.reference is not None and self.verify is None:
            raise ValueError(
                "Case defines a reference but no verify, and there is no default "
                "comparator. Pass a verify built on the framework's tolerance helper "
                "(tests/pytorch/utils.py::dtype_tols, tests/jax/utils.py::assert_allclose)."
            )
        if self.num_gpus < 1:
            raise ValueError(f"Case num_gpus must be at least 1, got {self.num_gpus}.")
        if self.num_gpus > 1:
            if self.dist_init is None:
                raise ValueError(
                    f"Case with num_gpus={self.num_gpus} must define dist_init to build the"
                    " process group its ranks share."
                )
            if self.barrier is None:
                raise ValueError(
                    f"Case with num_gpus={self.num_gpus} must define barrier so the harness"
                    " can align ranks before each timed sample."
                )
        if self.timeout <= 0:
            raise ValueError(f"Case timeout must be positive, got {self.timeout}.")
