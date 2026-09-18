# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Distributed execution context for a rank spawned by the benchmarkable harness."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

from .decorator import MODE_BENCHMARK

#: Bounded so a failing rank's context reaches pytest and JUnit without burying it.
_OUTPUT_TAIL_CHARS = 4000
_POLL_INTERVAL_S = 0.05
_KILL_GRACE_S = 10.0

RANK_ENV = "NVTE_BENCHMARK_DIST_RANK"
WORLD_SIZE_ENV = "NVTE_BENCHMARK_DIST_WORLD_SIZE"
RENDEZVOUS_ENV = "NVTE_BENCHMARK_DIST_RENDEZVOUS"
NODE_ID_ENV = "NVTE_BENCHMARK_DIST_NODE_ID"


def is_child() -> bool:
    """Whether this process is a rank the harness spawned, rather than the parent.

    The parent never sets these variables on itself, so their presence is what stops a
    child from launching ranks of its own for the same test.
    """
    return RANK_ENV in os.environ


def rank() -> int | None:
    """This process's rank, or ``None`` when it is not a spawned rank."""
    value = os.environ.get(RANK_ENV)
    return None if value is None else int(value)


def world_size() -> int | None:
    """The launch's world size, or ``None`` when this is not a spawned rank."""
    value = os.environ.get(WORLD_SIZE_ENV)
    return None if value is None else int(value)


def rendezvous() -> str | None:
    """The rendezvous location for this launch, or ``None`` outside one.

    Unique per launch: two configs at the same world size in one session must not share
    it, and a stale location left by a crashed run must not be reused.
    """
    return os.environ.get(RENDEZVOUS_ENV)


def expected_node_id() -> str | None:
    """The pytest node ID this rank was spawned to run, or ``None`` outside a launch.

    Checked against the test actually running so a stale ``NVTE_BENCHMARK_DIST_*``
    environment -- an old export, or a command line copied out of a log -- fails loudly
    instead of silently running one rank where the test asked for several.
    """
    return os.environ.get(NODE_ID_ENV)


def _child_command(pyfuncitem, settings, report_dir) -> list[str]:
    """Build the pytest invocation one rank runs: exactly this test, nothing else."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        pyfuncitem.nodeid,
        "-q",
        # Ranks run concurrently against one cache directory otherwise.
        "-p",
        "no:cacheprovider",
    ]
    if settings["mode"] == MODE_BENCHMARK:
        cmd += [
            "--nvte-benchmark",
            "--nvte-benchmark-iterations",
            str(settings["iterations"]),
            "--nvte-benchmark-warmup",
            str(settings["warmup"]),
            "--nvte-benchmark-inner-iterations",
            str(settings["inner_iterations"]),
            "--nvte-benchmark-min-run-time",
            str(settings["min_run_time"]),
            "--nvte-benchmark-report-dir",
            str(report_dir),
        ]
        if settings["no_reference"]:
            cmd.append("--nvte-benchmark-no-reference")
    return cmd


def _child_env(parent_env, rank_index, world, rendezvous_path, nodeid) -> dict[str, str]:
    """Build a rank's environment as a copy, never by mutating the parent's.

    Mutating ``os.environ`` and unsetting afterwards leaks values into later launches in
    the same session, which silently changes the workload being measured.
    """
    env = dict(parent_env)
    env[RANK_ENV] = str(rank_index)
    env[WORLD_SIZE_ENV] = str(world)
    env[RENDEZVOUS_ENV] = str(rendezvous_path)
    env[NODE_ID_ENV] = nodeid
    return env


def _harvest(procs, results) -> None:
    """Record the outcome of every rank that has exited, reading its pipes exactly once."""
    for index, proc in enumerate(procs):
        if index in results or proc.poll() is None:
            continue
        out, err = proc.communicate()
        results[index] = (proc.returncode, out, err)


def _stop_remaining(procs, results) -> None:
    """Stop ranks still running and record what they produced: ask first, then insist.

    A rank blocked inside a collective will not run a Python signal handler, so the kill
    is the backstop the grace period exists to avoid needing.
    """
    pending = [i for i in range(len(procs)) if i not in results]
    for index in pending:
        procs[index].terminate()
    for index in pending:
        proc = procs[index]
        try:
            out, err = proc.communicate(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        results[index] = (proc.returncode, out, err)


def _supervise(procs, timeout) -> list[tuple[int, str, str]]:
    """Wait for every rank, stopping the rest as soon as one fails or the budget expires.

    The parent runs no collective of its own, so it cannot wedge alongside the ranks it
    is watching -- which is what lets it enforce this at all.
    """
    deadline = time.monotonic() + timeout
    results: dict[int, tuple[int, str, str]] = {}
    timed_out = False
    while len(results) < len(procs):
        _harvest(procs, results)
        if any(returncode != 0 for returncode, _, _ in results.values()):
            # Cascade: stop the survivors so their dist_clean runs, rather than leaving
            # them blocked in a collective the failed rank will never join.
            break
        if time.monotonic() > deadline:
            timed_out = True
            break
        time.sleep(_POLL_INTERVAL_S)
    _stop_remaining(procs, results)
    if timed_out:
        raise AssertionError(
            f"Distributed test exceeded its {timeout:g}s budget; all {len(procs)} ranks were"
            " stopped. Raise Case(timeout=...) if the workload is legitimately this long."
        )
    return [results[i] for i in range(len(procs))]


def _assert_all_ranks_passed(outcomes) -> None:
    """Raise with each failing rank's output, bounded, naming the rank it came from."""
    failed = [(i, rc, out, err) for i, (rc, out, err) in enumerate(outcomes) if rc != 0]
    if not failed:
        return
    detail = []
    for index, returncode, out, err in failed:
        detail.append(
            f"\n--- rank {index} exited {returncode} ---"
            f"\n--- stdout (last {_OUTPUT_TAIL_CHARS} chars) ---\n{out[-_OUTPUT_TAIL_CHARS:]}"
            f"\n--- stderr (last {_OUTPUT_TAIL_CHARS} chars) ---\n{err[-_OUTPUT_TAIL_CHARS:]}"
        )
    raise AssertionError(f"{len(failed)} of {len(outcomes)} ranks failed." + "".join(detail))


def _collect_records(report_root) -> list:
    """Read every rank's report. Records already carry rank, so they stay distinguishable."""
    records = []
    for report in sorted(report_root.glob("rank*/benchmark_report.json")):
        with report.open("r", encoding="utf-8") as handle:
            records.extend(json.load(handle).get("records", []))
    return records


def run_across_ranks(pyfuncitem, case, settings) -> list:
    """Run one test across ``case.num_gpus`` ranks and return their benchmark records.

    Each rank is a full pytest session collecting exactly this node ID, so fixtures,
    parametrization and Case construction all work the way they do serially.
    """
    prefix = f"nvte-benchmark-{os.getpid()}-"
    with tempfile.TemporaryDirectory(prefix=prefix) as workdir:
        work = pathlib.Path(workdir)
        reports = work / "reports"
        reports.mkdir()
        # Unique per launch, not merely per world size: two configs at the same size in
        # one session would otherwise share a store, and a stale one is read as a
        # confusing NCCL bootstrap failure rather than a rendezvous error.
        rendezvous = work / "rdzv"
        procs = []
        for index in range(case.num_gpus):
            procs.append(
                subprocess.Popen(  # pylint: disable=consider-using-with
                    _child_command(pyfuncitem, settings, reports / f"rank{index}"),
                    env=_child_env(os.environ, index, case.num_gpus, rendezvous, pyfuncitem.nodeid),
                    cwd=str(pyfuncitem.config.rootpath),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        try:
            outcomes = _supervise(procs, case.timeout)
        finally:
            # Safety net only: _supervise has already harvested every rank, so this must
            # not touch the pipes again.
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
        _assert_all_ranks_passed(outcomes)
        return _collect_records(reports)
