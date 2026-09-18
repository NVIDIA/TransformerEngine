# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Case-execution engine: drives a Case's setup/evaluate/reference/verify cycle."""

from __future__ import annotations

import os
import time

from .case import Case, axis_value
from .device import synchronize
from .distributed import launch
from .timing import TIMING_METHOD, WallClockSampler, timing_stats


def _framework_for(pyfuncitem) -> str:
    """Infer the report's framework label from the test's location under tests/."""
    path = str(pyfuncitem.path)
    if f"{os.sep}jax{os.sep}" in path or path.endswith(f"{os.sep}jax"):
        return "jax"
    return "pytorch"


def _run_correctness(case: Case) -> None:
    """Run one setup/evaluate/reference/verify cycle with no timing."""
    current = launch()
    state = (
        case.dist_init(
            current.rank, current.world_size, current.coordinator_addr, current.coordinator_port
        )
        if case.dist_init is not None
        else None
    )
    try:
        state = case.setup(state)
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
        case.verify(actual, expected)
    finally:
        # The harness owns teardown so a Case callable never has to be defensive: a
        # CaseSkip from setup or a failing verify still releases the communicator.
        if case.dist_clean is not None:
            case.dist_clean(state)


def _run_benchmark_point(case, settings, pyfuncitem):
    """Gate once on correctness, then time evaluate and optionally reference."""
    current = launch()
    state = (
        case.dist_init(
            current.rank, current.world_size, current.coordinator_addr, current.coordinator_port
        )
        if case.dist_init is not None
        else None
    )
    try:
        state = case.setup(state)

        precondition_verified = False
        if case.reference is not None:
            actual = case.evaluate(state)
            # Same ordering rule as ``_run_correctness``: synchronize before reset.
            synchronize(actual)
            if case.reset is not None:
                case.reset(state)
            expected = case.reference(state)
            synchronize(expected)
            case.verify(actual, expected)
            precondition_verified = True

        # Invariant: setup() is never called twice without a reset() in between. The
        # next setup happens inside _time_variant, so this is the boundary that would
        # otherwise leave the gate's test data unreleased.
        if case.reset is not None:
            case.reset(state)

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
            record, state = _time_variant(
                case,
                settings,
                pyfuncitem,
                name,
                function,
                inner,
                batchable,
                precondition_verified,
                state,
            )
            records.append(record)
        return records
    finally:
        # The harness owns teardown so a Case callable never has to be defensive: a
        # CaseSkip from setup or a failing verify still releases the communicator.
        if case.dist_clean is not None:
            case.dist_clean(state)


def _time_variant(
    case,
    settings,
    pyfuncitem,
    variant,
    function,
    inner,
    batchable,
    precondition_verified,
    state,
):
    """Warm up, then time ``function``, and return its record alongside the state."""
    record = _base_record(pyfuncitem, variant, precondition_verified)
    state = case.setup(state)
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
        # Align ranks outside the measured interval: a rank arriving late would
        # otherwise have its wait recorded as this operation's cost on every rank that
        # arrived on time.
        if case.barrier is not None:
            case.barrier(state)
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
    return record, state


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
    # world_size is workload identity, so it rides in params and therefore reaches both
    # case_id and record_key: a 4-rank run is a different workload from an 8-rank one.
    # rank is not -- it is a separate field, keyed on only so N ranks' records cannot
    # collapse into one another.
    current = launch()
    world = None if current is None else current.world_size
    if world is not None:
        declared = params.get("world_size")
        if declared is not None and declared != world:
            raise ValueError(
                f"{pyfuncitem.nodeid} parametrizes world_size={declared} but is running on"
                f" {world} ranks, so its case_id would not describe what ran."
            )
        params["world_size"] = world
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
        "rank": None if current is None else current.rank,
        "world_size": world,
        "precondition_verified": precondition_verified,
        "tags": [],
        "unit_test": pyfuncitem.nodeid,
        "source": pyfuncitem.module.__name__,
        "regression_threshold": None,
    }


def case_id_for(pyfuncitem, params) -> str:
    """Build a record's stable identity from its module, test function and sorted axes."""
    axes = ".".join(f"{name}{axis_value(params[name])}" for name in sorted(params))
    return f"{pyfuncitem.module.__name__}.{pyfuncitem.originalname}.{axes}"
