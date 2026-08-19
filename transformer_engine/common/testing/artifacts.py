# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Artifact helpers for benchmarkable runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

from .case import axis_value
from .device import build_architectures, device_metadata


REPORT_SCHEMA_VERSION = "benchmark_report/v1"

# .../transformer_engine/common/testing/artifacts.py -> testing -> common
# -> transformer_engine -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def write_run_artifacts(
    output_dir: Path,
    records: list[dict[str, Any]],
    command: list[str],
    selection: dict[str, Any],
    sharding: dict[str, Any] | None = None,
    report_name: str = "benchmark_report.json",
    records_name: str = "benchmark_records.jsonl",
    summary_name: str = "benchmark_summary.csv",
) -> dict[str, Path]:
    """Write JSON, JSONL and CSV artifacts for one benchmark run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(records, command, selection, sharding=sharding)

    report_path = output_dir / report_name
    records_path = output_dir / records_name
    summary_path = output_dir / summary_name

    _write_json(report_path, report)
    _write_jsonl(records_path, records)
    _write_summary_csv(summary_path, records)
    return {"report": report_path, "records": records_path, "summary": summary_path}


def build_report(
    records: list[dict[str, Any]],
    command: list[str],
    selection: dict[str, Any],
    sharding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained machine-readable benchmark report."""
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "selection": selection,
        "environment": collect_environment(command),
        "sharding": sharding or {"enabled": False},
        "summary": {
            "record_count": len(records),
            "status_counts": status_counts,
        },
        "records": records,
    }


def collect_environment(command: list[str] | None = None) -> dict[str, Any]:
    """Collect stable environment metadata without copying arbitrary environment variables.

    Framework versions are read from already-imported modules; never import one here.
    ``build`` holds the architectures embedded in the loaded ``libtransformer_engine.so``.
    """
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "host": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
        },
        "git": _git_metadata(),
        "frameworks": _framework_versions(),
        "devices": {"cuda": device_metadata()},
        "build": build_architectures(),
        "scheduler": _scheduler_metadata(),
        "command": command or [],
    }


def merge_worker_reports(
    output_dir: Path,
    worker_report_paths: list[Path],
    command: list[str],
    selection: dict[str, Any],
    sharding: dict[str, Any],
) -> dict[str, Path]:
    """Merge worker reports into the standard top-level artifact set.

    A worker report that cannot be read is recorded in the output rather than raising.
    """
    merged_records: list[dict[str, Any]] = []
    worker_summaries = []
    unreadable_reports = []
    for report_path in sorted(worker_report_paths):
        try:
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            unreadable_reports.append({"path": str(report_path), "reason": str(exc)})
            continue
        worker_summaries.append(
            {
                "path": str(report_path),
                "summary": report.get("summary", {}),
                "selection": report.get("selection", {}),
            }
        )
        merged_records.extend(report.get("records", []))

    expected_records = _expected_records_from_worker_summaries(worker_summaries)
    sharding = dict(sharding)
    sharding["worker_reports"] = worker_summaries
    sharding["unreadable_worker_reports"] = unreadable_reports
    sharding["merge_validation"] = _validate_merged_records(
        merged_records,
        expected_records=expected_records,
        unreadable_reports=unreadable_reports,
    )
    return write_run_artifacts(
        output_dir,
        merged_records,
        command,
        selection,
        sharding=sharding,
    )


def record_key(record: dict[str, Any]) -> str:
    """Return a deterministic key for comparing benchmark records.

    ``params`` goes through ``axis_value``. Unlike the record writers, this ``json.dumps``
    has no ``default=str``, so a value it did not normalize raises here instead of becoming
    a key that differs every run.
    """
    key = {
        "case_id": record.get("case_id"),
        "framework": record.get("framework"),
        "operation": record.get("operation"),
        "variant": record.get("variant"),
        "params": {name: axis_value(value) for name, value in record.get("params", {}).items()},
    }
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def _validate_merged_records(
    records: list[dict[str, Any]],
    expected_records: list[dict[str, Any]] | None = None,
    unreadable_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        key = record_key(record)
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = []
    if expected_records is not None:
        expected_keys = sorted({record_key(record) for record in expected_records})
        missing = sorted(key for key in expected_keys if key not in counts)
    unreadable = unreadable_reports or []
    return {
        "record_count": len(records),
        "unique_record_count": len(counts),
        "duplicate_keys": duplicates,
        "missing_keys": missing,
        "unreadable_report_count": len(unreadable),
        "valid": not duplicates and not missing and not unreadable,
    }


def _expected_records_from_worker_summaries(
    worker_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_records = []
    for worker in worker_summaries:
        selection = worker.get("selection", {})
        include_reference = selection.get("include_reference", True)
        for case in selection.get("selected_cases", []):
            variants = []
            if include_reference and case.get("has_reference"):
                variants.append("reference")
            variants.append("evaluation")
            for variant in variants:
                expected_records.append(
                    {
                        "case_id": case.get("case_id"),
                        "framework": case.get("framework"),
                        "operation": case.get("operation"),
                        "variant": variant,
                        "params": case.get("params", {}),
                    }
                )
    return expected_records


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        # default=str: ``params`` hold pytest parametrize values, which are not always
        # JSON-native (a JAX dtype is a bare type object, not a serializable instance).
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str))
            handle.write("\n")


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "status",
        "framework",
        "case_id",
        "variant",
        "component",
        "operation",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            timing = record.get("timing", {})
            writer.writerow(
                {
                    "status": record.get("status"),
                    "framework": record.get("framework"),
                    "case_id": record.get("case_id"),
                    "variant": record.get("variant"),
                    "component": record.get("component"),
                    "operation": record.get("operation"),
                    "median_ms": timing.get("median_ms"),
                    "mean_ms": timing.get("mean_ms"),
                    "p95_ms": timing.get("p95_ms"),
                    "reason": record.get("reason"),
                }
            )


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run_git(["status", "--porcelain"])),
    }


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            encoding="utf-8",
            cwd=REPO_ROOT,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _framework_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "transformer_engine": _module_version("transformer_engine"),
        "torch": _module_version("torch"),
        "jax": _module_version("jax"),
        "jaxlib": _module_version("jaxlib"),
    }
    return versions


def _module_version(module_name: str) -> str | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    return getattr(module, "__version__", None)


def _scheduler_metadata() -> dict[str, Any]:
    names = [
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_STEP_GPUS",
    ]
    metadata = {name: os.environ.get(name) for name in names if os.environ.get(name) is not None}
    visible_devices = visible_cuda_devices()
    allocated_devices = scheduler_allocated_devices()
    metadata.update(
        {
            "visible_cuda_devices": visible_devices,
            "visible_gpu_count": len(visible_devices),
            "scheduler_allocated_devices": allocated_devices,
            "scheduler_allocated_gpu_count": len(allocated_devices),
        }
    )
    return metadata


def scheduler_allocated_devices() -> list[str]:
    """Return the device list the job scheduler allocated to this process."""
    for name in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
        devices = parse_device_list(os.environ.get(name))
        if devices:
            return devices
    return []


def visible_cuda_devices() -> list[str]:
    """Return the devices named by ``CUDA_VISIBLE_DEVICES``."""
    return parse_device_list(os.environ.get("CUDA_VISIBLE_DEVICES"))


def parse_device_list(raw: str | None) -> list[str]:
    """Split a comma-separated device list, dropping empty entries."""
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
