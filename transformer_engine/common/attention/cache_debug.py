# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused-attention graph-cache diagnostics for Python graph runtimes."""

from __future__ import annotations

import atexit
import os
import sys
import threading
from collections import defaultdict
from functools import lru_cache
from time import perf_counter_ns
from typing import Callable, Optional

_PREFIX = "[FUSED-ATTN-CACHE]"
_RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID")
_COUNTER_NAMES = ("hit", "miss", "create_graph", "cache_graph", "build_plans", "execute")
_BUILD_STAGES = (
    "validate",
    "build_operation_graph",
    "create_execution_plans",
    "check_support",
    "build_plans",
)

_lock = threading.Lock()
_counters = defaultdict(lambda: defaultdict(int))
_thread_counters = defaultdict(lambda: defaultdict(int))
_thread_devices = defaultdict(set)
_stage_timings = defaultdict(lambda: [0, 0])
_thread_ids: dict[int, int] = {}
_summary_registered = False


@lru_cache(maxsize=1)
def _configuration() -> tuple[int, bool, Optional[int]]:
    """Return ``(level, selected, rank)`` for the current process."""

    value = os.getenv("NVTE_FUSED_ATTN_CACHE_DEBUG", "0")
    level_text, separator, rank_text = value.partition(":")
    try:
        level = int(level_text)
    except ValueError:
        level = 1 if level_text else 0
    if level <= 0:
        return 0, False, None

    rank = None
    for variable in _RANK_ENV_VARS:
        rank_value = os.getenv(variable)
        if rank_value:
            try:
                rank = int(rank_value)
            except ValueError:
                rank = 0
            break
    if rank is None:
        return level, True, None
    if not separator:
        return level, rank == 0, rank
    if rank_text == "all":
        return level, True, rank
    selected_ranks = set()
    for token in rank_text.split(","):
        try:
            selected_ranks.add(int(token))
        except ValueError:
            continue
    return level, rank in selected_ranks, rank


def enabled(*, trace: bool = False) -> bool:
    """Whether cache diagnostics are enabled for this process and level."""

    level, selected, _ = _configuration()
    return selected and level >= (2 if trace else 1)


def _thread_id() -> int:
    native_id = threading.get_ident()
    with _lock:
        thread_id = _thread_ids.get(native_id)
        if thread_id is None:
            thread_id = len(_thread_ids)
            _thread_ids[native_id] = thread_id
    return thread_id


def _rank_tag() -> str:
    _, _, rank = _configuration()
    return "" if rank is None else f"rank={rank} | "


def _write(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


def _counter_line(
    thread_field: str,
    device_field: str,
    backend: str,
    direction: str,
    counters: dict[str, int],
    event: str = "",
) -> str:
    label = f"{backend} {direction}"
    if event:
        label += f" {event}"
    values = ", ".join(f"{name}={counters.get(name, 0):4d}" for name in _COUNTER_NAMES)
    return f"{_PREFIX} {_rank_tag()}{thread_field:<7} {device_field:<9} | {label:<24} | {values}\n"


def _register_summary() -> None:
    global _summary_registered
    with _lock:
        if _summary_registered:
            return
        atexit.register(print_summary)
        _summary_registered = True


def record_event(
    backend: str,
    direction: str,
    event: str,
    *,
    device: Optional[int] = None,
    key=None,
) -> None:
    """Record one graph cache event and optionally emit its level-2 trace."""

    if not enabled():
        return
    event = event.lower()
    if event not in _COUNTER_NAMES:
        raise ValueError(f"Unknown fused-attention cache event {event!r}.")
    _register_summary()
    thread_id = _thread_id()
    site = (backend, direction)
    thread_site = (thread_id, backend, direction)
    with _lock:
        _counters[site][event] += 1
        _thread_counters[thread_site][event] += 1
        if device is not None:
            _thread_devices[thread_id].add(device)
        snapshot = dict(_counters[site])
    if not enabled(trace=True):
        return
    if event in ("hit", "miss") and key is not None:
        _write(
            f"{_PREFIX} {_rank_tag()}tid={thread_id:<3} dev={str(device):<3} | "
            f"{backend} {direction} {event.upper():<12} | {key!r}\n"
        )
    else:
        _write(
            _counter_line(
                f"tid={thread_id}",
                f"dev={device}",
                backend,
                direction,
                snapshot,
                event.upper(),
            )
        )


def record_lookup(
    backend: str,
    direction: str,
    *,
    hit: bool,
    device: Optional[int] = None,
    key=None,
) -> None:
    """Record a cache hit or miss."""

    record_event(backend, direction, "hit" if hit else "miss", device=device, key=key)


def record_build_time(backend: str, direction: str, stage: str, elapsed_ns: int) -> None:
    """Accumulate CPU wall time for a cuDNN graph build stage."""

    if not enabled():
        return
    if stage not in _BUILD_STAGES:
        raise ValueError(f"Unknown fused-attention graph build stage {stage!r}.")
    _register_summary()
    with _lock:
        timing = _stage_timings[(backend, direction, stage)]
        timing[0] += 1
        timing[1] += elapsed_ns


def build_recorder(backend: str, direction: str) -> Callable[[str, int], None]:
    """Create the callback consumed by ``build_cudnn_graph``."""

    def record(name: str, elapsed_ns: int) -> None:
        if name in _BUILD_STAGES:
            record_build_time(backend, direction, name, elapsed_ns)
        else:
            record_event(backend, direction, name.lower())

    return record


def render_summary() -> str:
    """Render the current diagnostic summary without writing it."""

    if not enabled():
        return ""
    with _lock:
        counters = {site: dict(values) for site, values in _counters.items()}
        thread_counters = {site: dict(values) for site, values in _thread_counters.items()}
        thread_devices = {thread_id: set(values) for thread_id, values in _thread_devices.items()}
        stage_timings = {site: tuple(values) for site, values in _stage_timings.items()}

    marker = f"{_PREFIX} {_rank_tag()}===== summary"
    lines = [marker + " begin =====\n"]
    for (thread_id, backend, direction), values in sorted(thread_counters.items()):
        devices = thread_devices.get(thread_id, set())
        if not devices:
            device_field = "dev=None"
        elif len(devices) == 1:
            device_field = f"dev={next(iter(devices))}"
        else:
            device_field = "dev=mixed"
        lines.append(_counter_line(f"tid={thread_id}", device_field, backend, direction, values))
    for (backend, direction), values in sorted(counters.items()):
        lines.append(_counter_line("tid=all", "dev=all", backend, direction, values))
    for (backend, direction, stage), (calls, elapsed_ns) in sorted(stage_timings.items()):
        lines.append(
            f"{_PREFIX} {_rank_tag()}{backend:<3} {direction:<3} {stage:<22} | "
            f"calls={calls} | time={elapsed_ns / calls / 1e6:9.3f} ms/call\n"
        )
    lines.append(marker + " end =====\n")
    return "".join(lines)


def print_summary() -> None:
    """Write the current summary to stderr if diagnostics are enabled."""

    summary = render_summary()
    if summary:
        _write(summary)


def time_call(callback: Optional[Callable[[str, int], None]], stage: str, function):
    """Call a graph-build stage and report its elapsed CPU wall time."""

    if callback is None:
        return function()
    start = perf_counter_ns()
    try:
        return function()
    finally:
        callback(stage, perf_counter_ns() - start)


def _reset_for_tests() -> None:
    """Clear diagnostic state. Intended only for GPU-independent tests."""

    with _lock:
        _counters.clear()
        _thread_counters.clear()
        _thread_devices.clear()
        _stage_timings.clear()
        _thread_ids.clear()
    _configuration.cache_clear()
