# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Timing backends for benchmarkable cases."""

from __future__ import annotations

import math
import statistics
import time

from .device import synchronize


TIMING_METHOD = "wall-clock-with-device-sync"


class WallClockSampler:
    """Time ``inner_iterations`` calls with the host clock and one device sync."""

    def __init__(self, inner_iterations: int) -> None:
        self.inner_iterations = inner_iterations

    def __call__(self, function, state) -> float:
        """Return the mean milliseconds per call across the inner iterations."""
        output = None
        start = time.perf_counter()
        for _ in range(self.inner_iterations):
            output = function(state)
        synchronize(output)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms / self.inner_iterations


def timing_stats(samples_ms: list[float]) -> dict[str, float]:
    """Compute timing statistics in milliseconds."""
    if not samples_ms:
        return {
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stddev_ms": 0.0,
            "p95_ms": 0.0,
        }

    ordered = sorted(samples_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stddev_ms": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "p95_ms": ordered[p95_index],
    }
