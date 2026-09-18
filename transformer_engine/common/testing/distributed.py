# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Distributed execution context for a rank spawned by the benchmarkable harness."""

from __future__ import annotations

import json
import os
import pathlib
import signal
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


def rank() -> int | None:
    """This process's rank, or ``None`` when it is not a rank the harness spawned."""
    value = os.environ.get(RANK_ENV)
    return None if value is None else int(value)


def world_size() -> int | None:
    """The launch's world size, or ``None`` when this is not a spawned rank."""
    value = os.environ.get(WORLD_SIZE_ENV)
    return None if value is None else int(value)


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
    """Record the outcome of every rank that has exited."""
    for index, proc in enumerate(procs):
        if index not in results and proc.poll() is not None:
            results[index] = proc.returncode


def _stop_remaining(procs, results) -> None:
    """Stop ranks still running: ask, then insist.

    ``terminate`` is a kill, not an unwind -- CPython installs a handler for SIGINT only,
    so a rank does not run its ``finally`` blocks. Process death is what releases the
    communicator and the device.
    """
    pending = [i for i in range(len(procs)) if i not in results]
    for index in pending:
        procs[index].terminate()
    for index in pending:
        proc = procs[index]
        try:
            proc.wait(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        results[index] = proc.returncode


def _supervise(procs, results, timeout) -> bool:
    """Wait for every rank, stopping the rest once one fails or the budget expires.

    Returns whether the budget expired. The parent runs no collective of its own, and
    writes rank output to files rather than pipes, so it has nothing to block on.
    """
    deadline = time.monotonic() + timeout
    timed_out = False
    while len(results) < len(procs):
        _harvest(procs, results)
        if any(returncode != 0 for returncode in results.values()):
            break
        if time.monotonic() > deadline:
            timed_out = True
            break
        time.sleep(_POLL_INTERVAL_S)
    _stop_remaining(procs, results)
    return timed_out


def _rank_output(log_dir, index) -> str:
    """One rank's captured streams, tail-bounded so a cascade stays readable."""
    text = []
    for stream in ("stdout", "stderr"):
        path = log_dir / f"rank{index}.{stream}"
        body = path.read_text(errors="replace") if path.exists() else ""
        text.append(f"\n--- rank {index} {stream} (last {_OUTPUT_TAIL_CHARS} chars) ---\n")
        text.append(body[-_OUTPUT_TAIL_CHARS:])
    return "".join(text)


def _report_outcomes(results, log_dir, timed_out, timeout) -> None:
    """Raise if the launch did not succeed, with the output of the ranks that explain it.

    A rank the harness stopped exits -SIGTERM/-SIGKILL. Counting those as failures would
    blame the survivors for the one rank that actually failed, and bury its output under
    theirs.
    """
    stopped = {-signal.SIGTERM, -signal.SIGKILL}
    failed = [i for i, rc in sorted(results.items()) if rc != 0 and rc not in stopped]
    if timed_out:
        # Every rank's output is kept: a hang is the case where it is most needed, and
        # the cause is usually visible in whichever rank stopped making progress.
        detail = "".join(_rank_output(log_dir, i) for i in sorted(results))
        raise AssertionError(
            f"Distributed test hit its {timeout:g}s budget with"
            f" {len(results) - len(failed)} rank(s) still running.{detail}"
        )
    if failed:
        detail = "".join(_rank_output(log_dir, i) for i in failed)
        raise AssertionError(f"rank(s) {failed} failed.{detail}")


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
    parametrization and Case construction work the way they do serially.
    """
    with tempfile.TemporaryDirectory(prefix=f"nvte-benchmark-{os.getpid()}-") as workdir:
        work = pathlib.Path(workdir)
        reports = work / "reports"
        reports.mkdir()
        # Unique per launch, not merely per world size: two configs at the same size in
        # one session would otherwise share a store, and a stale one surfaces as an NCCL
        # bootstrap failure rather than a rendezvous error.
        rendezvous_path = work / "rdzv"
        procs, streams, results = [], [], {}
        try:
            for index in range(case.num_gpus):
                # Files, not pipes: nothing drains a running rank, and a rank that wrote
                # past the pipe buffer would block in write() until the budget expired.
                out = (work / f"rank{index}.stdout").open("w", encoding="utf-8")
                err = (work / f"rank{index}.stderr").open("w", encoding="utf-8")
                streams += [out, err]
                procs.append(
                    subprocess.Popen(  # pylint: disable=consider-using-with
                        _child_command(pyfuncitem, settings, reports / f"rank{index}"),
                        env=_child_env(
                            os.environ, index, case.num_gpus, rendezvous_path, pyfuncitem.nodeid
                        ),
                        cwd=str(pyfuncitem.config.rootpath),
                        stdout=out,
                        stderr=err,
                    )
                )
            timed_out = _supervise(procs, results, case.timeout)
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            for stream in streams:
                stream.close()
        _report_outcomes(results, work, timed_out, case.timeout)
        return _collect_records(reports)
