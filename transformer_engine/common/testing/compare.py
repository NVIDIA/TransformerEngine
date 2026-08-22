# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Compare benchmarkable reports against historical artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .artifacts import record_key


DEFAULT_RELATIVE_THRESHOLD = 0.05
DEFAULT_ABSOLUTE_THRESHOLD_MS = 0.01


def compare_reports(
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD,
    absolute_threshold_ms: float = DEFAULT_ABSOLUTE_THRESHOLD_MS,
) -> dict[str, Any]:
    """Compare completed records in two reports."""
    baseline_records = _completed_records_by_key(baseline_report)
    current_records = _completed_records_by_key(current_report)

    regressions = []
    improvements = []
    unchanged = []
    incompatible = []
    hardware_compatible, incompatible_reason = _hardware_compatible(baseline_report, current_report)

    for key, current in sorted(current_records.items()):
        baseline = baseline_records.get(key)
        if baseline is None:
            continue
        if not hardware_compatible:
            incompatible.append({"key": key, "reason": incompatible_reason})
            continue

        baseline_ms = _median_ms(baseline)
        current_ms = _median_ms(current)
        delta_ms = current_ms - baseline_ms
        relative_delta = delta_ms / baseline_ms if baseline_ms > 0 else 0.0
        threshold = _threshold_for_record(current, relative_threshold, absolute_threshold_ms)
        entry = {
            "key": key,
            "case_id": current.get("case_id"),
            "variant": current.get("variant"),
            "params": current.get("params", {}),
            "baseline_median_ms": baseline_ms,
            "current_median_ms": current_ms,
            "delta_ms": delta_ms,
            "relative_delta": relative_delta,
            "relative_threshold": threshold["relative"],
            "absolute_threshold_ms": threshold["absolute_ms"],
        }
        if delta_ms > threshold["absolute_ms"] and relative_delta > threshold["relative"]:
            regressions.append(entry)
        elif delta_ms < -threshold["absolute_ms"] and -relative_delta > threshold["relative"]:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    missing = sorted(key for key in baseline_records if key not in current_records)
    new = sorted(key for key in current_records if key not in baseline_records)
    return {
        "schema_version": "benchmark_comparison/v1",
        "summary": {
            "baseline_records": len(baseline_records),
            "current_records": len(current_records),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "unchanged": len(unchanged),
            "missing": len(missing),
            "new": len(new),
            "incompatible": len(incompatible),
        },
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "missing": missing,
        "new": new,
        "incompatible": incompatible,
    }


def main() -> int:
    """Compare two benchmark_report/v1 JSON files and write a comparison report."""
    parser = argparse.ArgumentParser(description="Compare benchmarkable JSON reports.")
    parser.add_argument("--baseline", required=True, help="Historical benchmark_report.json")
    parser.add_argument("--current", required=True, help="Current benchmark_report.json")
    parser.add_argument("--output", required=True, help="Comparison JSON output path")
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=DEFAULT_RELATIVE_THRESHOLD,
        help="Default relative regression threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--absolute-threshold-ms",
        type=float,
        default=DEFAULT_ABSOLUTE_THRESHOLD_MS,
        help="Default absolute regression threshold in ms (default: %(default)s)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit nonzero when any regression is detected.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help=(
            "Exit nonzero when a baseline record has no completed counterpart in the current "
            "report. A case that starts erroring disappears from the comparison entirely, so "
            "without this flag a broken case looks like a clean run."
        ),
    )
    parser.add_argument(
        "--fail-on-incompatible",
        action="store_true",
        help=(
            "Exit nonzero when any record is hardware-incompatible with its baseline "
            "counterpart. Every incompatible record short-circuits past the regression test "
            "and 'missing' stays zero because the key is still present, so without this flag "
            "a baseline from different hardware -- e.g. an H100 baseline against a B200 run -- "
            "compares as clean by default."
        ),
    )
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline))
    current = _read_json(Path(args.current))
    comparison = compare_reports(
        baseline,
        current,
        relative_threshold=args.relative_threshold,
        absolute_threshold_ms=args.absolute_threshold_ms,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if comparison["incompatible"]:
        reasons = sorted(
            {entry["reason"] for entry in comparison["incompatible"] if entry.get("reason")}
        )
        detail = f" ({'; '.join(reasons)})" if reasons else ""
        print(
            f"Warning: {len(comparison['incompatible'])} record(s) are hardware-incompatible "
            f"with their baseline counterpart and were skipped for regression comparison{detail}.",
            file=sys.stderr,
        )

    if args.fail_on_regression and comparison["regressions"]:
        print(
            f"Error: {len(comparison['regressions'])} regression(s) detected.",
            file=sys.stderr,
        )
        return 2
    if args.fail_on_missing and comparison["missing"]:
        print(
            f"Error: {len(comparison['missing'])} baseline record(s) missing from the current "
            "report.",
            file=sys.stderr,
        )
        return 3
    if args.fail_on_incompatible and comparison["incompatible"]:
        print(
            f"Error: {len(comparison['incompatible'])} record(s) are hardware-incompatible "
            "with their baseline counterpart.",
            file=sys.stderr,
        )
        return 4
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _completed_records_by_key(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for record in report.get("records", []):
        if record.get("status") == "completed":
            records[record_key(record)] = record
    return records


def _median_ms(record: dict[str, Any]) -> float:
    return float(record.get("timing", {}).get("median_ms", 0.0))


def _threshold_for_record(
    record: dict[str, Any],
    default_relative: float,
    default_absolute_ms: float,
) -> dict[str, float]:
    override = record.get("regression_threshold") or {}
    return {
        "relative": float(override.get("relative", default_relative)),
        "absolute_ms": float(override.get("absolute_ms", default_absolute_ms)),
    }


def _hardware_compatible(
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
) -> tuple[bool, str | None]:
    """Return whether the two reports were recorded on compatible hardware, plus a reason.

    Missing device metadata on either side means compatibility cannot be established, and is
    treated as incompatible rather than as a match.
    """
    baseline_device = _primary_device_identity(baseline_report)
    current_device = _primary_device_identity(current_report)
    if baseline_device is None and current_device is None:
        return (
            False,
            (
                "neither the baseline nor the current report carries device metadata -- "
                "hardware compatibility cannot be established"
            ),
        )
    if baseline_device is None:
        return (
            False,
            (
                "the baseline report carries no device metadata -- hardware "
                "compatibility cannot be established"
            ),
        )
    if current_device is None:
        return (
            False,
            (
                "the current report carries no device metadata -- hardware "
                "compatibility cannot be established"
            ),
        )
    if baseline_device != current_device:
        return False, "hardware metadata differs"
    return True, None


def _primary_device_identity(report: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return the primary CUDA device's ``(name, compute_capability)``.

    Returns ``None`` when the report carries no CUDA device metadata.
    """
    devices = report.get("environment", {}).get("devices", {}).get("cuda", {}).get("devices", [])
    if not devices:
        return None
    device = devices[0]
    return (device.get("name"), tuple(device.get("compute_capability", [])))


if __name__ == "__main__":
    raise SystemExit(main())
