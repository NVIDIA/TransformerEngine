# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Case-execution engine: drives a Case's setup/evaluate/reference/verify cycle."""

from __future__ import annotations

import os
import time

from .case import Case, axis_value
from .device import synchronize
from .timing import TIMING_METHOD, WallClockSampler, timing_stats


def _framework_for(pyfuncitem) -> str:
    """Infer the framework label for the report from the test's location under tests/.

    This is a reporting label only; nothing in the execution path branches on it.
    """
    path = str(pyfuncitem.path)
    if f"{os.sep}jax{os.sep}" in path or path.endswith(f"{os.sep}jax"):
        return "jax"
    return "pytorch"


def _run_correctness(case: Case) -> None:
    """Run one setup/evaluate/reference/verify cycle with no timing."""
    state = case.setup()
    actual = case.evaluate(state)
    if case.reference is None:
        return
    # ``synchronize(actual)`` must precede ``reset(state)``: ``actual`` may alias
    # ``state``, so an eager ``reset`` would race with in-flight ``evaluate`` work.
    # ``reset`` must precede ``reference`` so it sees unmutated state.
    synchronize(actual)
    if case.reset is not None:
        case.reset(state)
    expected = case.reference(state)
    synchronize(expected)
    case.run_verify(actual, expected)


def _run_benchmark_point(case, settings, pyfuncitem):
    """Gate once on correctness, then time evaluate and optionally reference."""
    state = case.setup()

    precondition_verified = False
    if case.reference is not None:
        actual = case.evaluate(state)
        # Same ordering rule as ``_run_correctness``: synchronize before reset.
        synchronize(actual)
        if case.reset is not None:
            case.reset(state)
        expected = case.reference(state)
        synchronize(expected)
        case.run_verify(actual, expected)
        precondition_verified = True

    variants = [("evaluation", case.evaluate)]
    if case.reference is not None and case.time_reference and not settings["no_reference"]:
        variants.insert(0, ("reference", case.reference))

    # A case needing reset between calls cannot be batched: inner iterations are
    # submitted back to back with no chance to reset, so timings would average over
    # drifting state.
    batchable = case.batchable and case.reset is None
    inner = settings["inner_iterations"] if batchable else 1
    records = []
    for name, function in variants:
        records.append(
            _time_variant(
                case, settings, pyfuncitem, name, function, inner, batchable, precondition_verified
            )
        )
    return records


def _time_variant(
    case, settings, pyfuncitem, variant, function, inner, batchable, precondition_verified
):
    """Warm up, then time ``function``, with no verification inside the timed loop.

    A ``CaseSkip`` from ``setup()`` here means ``setup()`` is non-deterministic, which
    the ``Case`` contract forbids, so it is left to propagate rather than caught.
    """
    record = _base_record(pyfuncitem, variant, precondition_verified)
    state = case.setup()
    for _ in range(settings["warmup"]):
        output = function(state)
        synchronize(output)
        if case.reset is not None:
            case.reset(state)
    synchronize()

    sampler = WallClockSampler(inner)
    samples_ms = []
    start = time.perf_counter()
    while (
        len(samples_ms) < settings["iterations"]
        or time.perf_counter() - start < settings["min_run_time"]
    ):
        samples_ms.append(sampler(function, state))
        if case.reset is not None:
            case.reset(state)
    synchronize()

    stats = timing_stats(samples_ms)
    record.update(
        {
            "status": "completed",
            "warmup_iterations": settings["warmup"],
            "iterations": len(samples_ms),
            "inner_iterations": inner,
            "batchable": batchable,
            "timing_method": TIMING_METHOD,
            "samples_ms": samples_ms,
            "timing": stats,
            "metrics": _metrics(case, stats["median_ms"]),
        }
    )
    return record


def _metrics(case, median_ms):
    """Derive bandwidth and FLOPs metrics from the median timing, when available."""
    metrics = {}
    if median_ms > 0:
        if case.bytes_moved is not None:
            metrics["bandwidth_GBps"] = case.bytes_moved / (median_ms / 1.0e3) / 1.0e9
        if case.flops is not None:
            metrics["tflops"] = case.flops / (median_ms / 1.0e3) / 1.0e12
    return metrics


def _base_record(pyfuncitem, variant, precondition_verified):
    """Build the record fields that are known before timing runs."""
    params = {name: axis_value(v) for name, v in pyfuncitem.callspec.params.items()}
    return {
        "schema_version": "benchmark_record/v1",
        "status": "pending",
        "variant": variant,
        "framework": _framework_for(pyfuncitem),
        "case_id": case_id_for(pyfuncitem, params),
        "component": pyfuncitem.module.__name__.rsplit(".", maxsplit=1)[-1],
        "operation": pyfuncitem.originalname,
        "params": params,
        "node_id": pyfuncitem.nodeid,
        "precondition_verified": precondition_verified,
        "tags": [],
        "unit_test": pyfuncitem.nodeid,
        "source": pyfuncitem.module.__name__,
        "regression_threshold": None,
    }


def case_id_for(pyfuncitem, params) -> str:
    """Build the stable identity for a benchmark record.

    Keyed on module, test function and sorted named axis values rather than the
    positional pytest node ID, which would rename cases when an axis is reordered.
    ``params`` is already rendered by the caller; re-applying ``axis_value`` here is safe
    only because it is idempotent on the primitives that rendering produces.
    """
    axes = ".".join(f"{name}{axis_value(params[name])}" for name in sorted(params))
    return f"{pyfuncitem.module.__name__}.{pyfuncitem.originalname}.{axes}"
