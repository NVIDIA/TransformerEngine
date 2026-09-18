# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Distributed execution context for a rank spawned by the benchmarkable harness."""

from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
from typing import NamedTuple
from xml.etree import ElementTree

from .decorator import MODE_BENCHMARK

#: Characters kept from the end of each of a rank's captured streams.
_OUTPUT_TAIL_CHARS = 4000
_POLL_INTERVAL_S = 0.05
_KILL_GRACE_S = 10.0

#: Set by the harness on each rank it spawns, to the node ID that rank must run.
LAUNCH_ENV = "NVTE_BENCHMARK_DIST_LAUNCH"


class LaunchResult(NamedTuple):
    """What a multi-rank launch produced: benchmark records, and the reason every rank
    gave if they all skipped."""

    records: list
    skip_reason: str | None


class Launch(NamedTuple):
    """The launch a spawned rank belongs to."""

    node_id: str
    rank: int
    world_size: int
    coordinator_addr: str
    coordinator_port: int


def launch() -> Launch | None:
    """This rank's launch, or ``None`` when the harness did not spawn this process."""
    node_id = os.environ.get(LAUNCH_ENV)
    if node_id is None:
        return None
    return Launch(
        node_id,
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]),
    )


def _primary_address() -> str:
    """This host's outward-facing address, falling back to loopback."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _free_port() -> int:
    """A port the OS picks and has just confirmed free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def _child_command(pyfuncitem, settings, report_dir, junit_path) -> list[str]:
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
        # The parent's only evidence of what the rank did: ran, skipped, or never
        # collected the test at all. An exit code cannot tell those apart.
        f"--junitxml={junit_path}",
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


def _child_env(parent_env, rank_index, world, endpoint, nodeid) -> dict[str, str]:
    """Build a rank's environment: the launch marker plus the standard ``env://`` names."""
    address, port = endpoint
    env = dict(parent_env)
    env[LAUNCH_ENV] = nodeid
    env["MASTER_ADDR"] = address
    env["MASTER_PORT"] = str(port)
    env["RANK"] = str(rank_index)
    env["WORLD_SIZE"] = str(world)
    env["LOCAL_RANK"] = str(rank_index)
    env["LOCAL_WORLD_SIZE"] = str(world)
    return env


def _harvest(procs, results) -> None:
    """Record the outcome of every rank that has exited."""
    for index, proc in enumerate(procs):
        if index not in results and proc.poll() is not None:
            results[index] = proc.returncode


def _stop_remaining(procs, results) -> set[int]:
    """Stop ranks still running, returning their indices: ``terminate``, then ``kill``
    after a grace period."""
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
    return set(pending)


def _supervise(procs, results, timeout) -> tuple[bool, set[int]]:
    """Wait for every rank, stopping the rest once one fails or the budget expires, and
    return whether the budget expired alongside the ranks the harness stopped."""
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
    return timed_out, _stop_remaining(procs, results)


def _rank_output(log_dir, index) -> str:
    """One rank's captured streams, bounded to the last ``_OUTPUT_TAIL_CHARS`` of each."""
    text = []
    for stream in ("stdout", "stderr"):
        path = log_dir / f"rank{index}.{stream}"
        body = path.read_text(errors="replace") if path.exists() else ""
        text.append(f"\n--- rank {index} {stream} (last {_OUTPUT_TAIL_CHARS} chars) ---\n")
        text.append(body[-_OUTPUT_TAIL_CHARS:])
    return "".join(text)


def _report_outcomes(results, stopped, log_dir, timed_out, timeout) -> None:
    """Raise if any rank exited badly, with every rank's output."""
    # Membership in ``stopped``, not the signal number: a rank the OOM killer or the job
    # scheduler reaped also exits -SIGKILL/-SIGTERM, and that is a failure.
    failed = [i for i, rc in sorted(results.items()) if rc != 0 and i not in stopped]
    # Every rank's output either way: the rank that exits first is often not the one that
    # explains why, and a hang names no culprit at all.
    detail = "".join(_rank_output(log_dir, i) for i in sorted(results))
    if timed_out:
        raise AssertionError(
            f"Distributed test hit its {timeout:g}s budget with {len(stopped)} rank(s)"
            f" still running.{detail}"
        )
    if failed:
        raise AssertionError(f"rank(s) {failed} failed.{detail}")


#: JUnit child elements that mean the rank did not simply run the test.
_OUTCOME_ELEMENTS = {"skipped": "skipped", "failure": "failed", "error": "failed"}


def _rank_outcome(junit_path) -> tuple[str, str]:
    """One rank's pytest outcome and its reason, read from that rank's JUnit report."""
    if not junit_path.exists():
        return "missing", "wrote no JUnit report"
    cases = list(ElementTree.parse(junit_path).iter("testcase"))
    if not cases:
        return "missing", "collected no test"
    for element in cases[0]:
        outcome = _OUTCOME_ELEMENTS.get(element.tag)
        if outcome is not None:
            return outcome, element.get("message", "")
    return "passed", ""


def _rank_skip_reason(work, world) -> str | None:
    """Require every rank to have run this test, and return their reason if all skipped."""
    outcomes = [_rank_outcome(work / f"rank{index}.xml") for index in range(world)]
    broken = [i for i, (state, _) in enumerate(outcomes) if state in ("missing", "failed")]
    if broken:
        reasons = ", ".join(f"rank {i}: {outcomes[i][1] or outcomes[i][0]}" for i in broken)
        detail = "".join(_rank_output(work, i) for i in broken)
        raise AssertionError(f"rank(s) {broken} did not run this test ({reasons}).{detail}")
    skipped = [i for i, (state, _) in enumerate(outcomes) if state == "skipped"]
    if len(skipped) == world:
        return outcomes[0][1] or "every rank skipped"
    if skipped:
        raise AssertionError(
            f"rank(s) {skipped} skipped while the rest ran: a collective test must skip on"
            " every rank or none."
        )
    return None


def _collect_records(report_root, world) -> list:
    """Read every rank's benchmark records, requiring a report from each."""
    records = []
    for index in range(world):
        report = report_root / f"rank{index}" / "benchmark_report.json"
        if not report.exists():
            raise AssertionError(f"rank {index} ran this test but wrote no benchmark report.")
        with report.open("r", encoding="utf-8") as handle:
            records.extend(json.load(handle).get("records", []))
    return records


def run_across_ranks(pyfuncitem, case, settings) -> LaunchResult:
    """Run one test across ``case.num_gpus`` ranks.

    Each rank is a full pytest session collecting exactly this node ID.
    """
    with tempfile.TemporaryDirectory(prefix=f"nvte-benchmark-{os.getpid()}-") as workdir:
        work = pathlib.Path(workdir)
        reports = work / "reports"
        reports.mkdir()
        endpoint = (_primary_address(), _free_port())
        procs, streams, results = [], [], {}
        try:
            for index in range(case.num_gpus):
                # Files, not pipes: nothing drains a running rank.
                out = (work / f"rank{index}.stdout").open("w", encoding="utf-8")
                err = (work / f"rank{index}.stderr").open("w", encoding="utf-8")
                streams += [out, err]
                procs.append(
                    subprocess.Popen(  # pylint: disable=consider-using-with
                        _child_command(
                            pyfuncitem,
                            settings,
                            reports / f"rank{index}",
                            work / f"rank{index}.xml",
                        ),
                        env=_child_env(
                            os.environ, index, case.num_gpus, endpoint, pyfuncitem.nodeid
                        ),
                        cwd=str(pyfuncitem.config.rootpath),
                        stdout=out,
                        stderr=err,
                    )
                )
            timed_out, stopped = _supervise(procs, results, case.timeout)
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            for stream in streams:
                stream.close()
        _report_outcomes(results, stopped, work, timed_out, case.timeout)
        skip_reason = _rank_skip_reason(work, case.num_gpus)
        if skip_reason is not None:
            return LaunchResult([], skip_reason)
        if settings["mode"] != MODE_BENCHMARK:
            return LaunchResult([], None)
        return LaunchResult(_collect_records(reports, case.num_gpus), None)
