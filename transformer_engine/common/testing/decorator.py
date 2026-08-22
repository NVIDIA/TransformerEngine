# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Benchmarkable mode constants, plugin markers, and the ``benchmark`` decorator."""

import pytest

from .declaration import record_axis

MODE_CORRECTNESS = "correctness"
"""Run the correctness matrix, verified, with nothing timed."""

MODE_BENCHMARK = "benchmark"
"""Run only the benchmark matrix, gated by a one-time correctness check per point."""

CASE_MARKER = "nvte_case"
"""Case-bearing: the test returns a Case, so the plugin runs it instead of pytest."""

BENCHMARK_MARKER = "nvte_benchmark"
"""Benchmark-eligible: the test declared benchmark axes with ``@benchmark(...)``."""

SUPPRESS_MARKER = "nvte_no_benchmark"
"""Benchmarking suppressed by ``@benchmark.skip`` or a true ``@benchmark.skipif``."""

MARKERS = (
    (CASE_MARKER, "test returns a Case for the nvte-benchmark plugin to run."),
    (BENCHMARK_MARKER, "Case-bearing test that declared benchmark axes."),
    (SUPPRESS_MARKER, "Case-bearing test that is never a benchmark point."),
)


def _apply(holder, *names, **kwargs):
    """Apply plugin markers by name to a function, a method, or a class."""
    for name in names:
        holder = getattr(pytest.mark, name)(**kwargs)(holder)
    return holder


class _Benchmark:
    """Implements the ``benchmark`` decorator.

    An instance, not a module-level function carrying attributes, so ``skip`` and
    ``skipif`` are real bound methods that ``help()``, ``inspect`` and pylint resolve.
    """

    def __call__(self, argnames, values):
        """Mark a test Case-bearing and benchmark-eligible, and declare the values one axis
        takes in benchmark mode.

        ``argnames`` accepts the same forms as ``pytest.mark.parametrize``. The values are
        substituted into an existing ``parametrize`` mark on the test, its class, or its
        module; undeclared axes keep their correctness values. Applies to a function, a
        method, or a class.
        """

        def decorator(holder):
            record_axis(holder, argnames, values)
            return _apply(holder, CASE_MARKER, BENCHMARK_MARKER)

        return decorator

    def skip(self, holder=None, *, reason=""):
        """Mark a test Case-bearing but never benchmarked. Usable bare or called with a
        ``reason``, on a function, a method, or a class."""

        def decorator(target):
            target = _apply(target, CASE_MARKER)
            return _apply(target, SUPPRESS_MARKER, reason=reason)

        return decorator if holder is None else decorator(holder)

    def skipif(self, condition, *, reason=""):
        """Mark a test Case-bearing, and suppress benchmarking when ``condition`` is true.

        A false condition declares nothing: skip and skipif only ever subtract benchmarking.
        """

        def decorator(target):
            target = _apply(target, CASE_MARKER)
            return _apply(target, SUPPRESS_MARKER, reason=reason) if condition else target

        return decorator


benchmark = _Benchmark()
