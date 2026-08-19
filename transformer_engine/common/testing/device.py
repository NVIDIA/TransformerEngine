# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""CUDA device access for benchmarkable tests, via cuda-python rather than a framework."""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
from typing import Any

# Cubin sections from ``cuobjdump --list-elf``, e.g. ``libtransformer_engine.4.sm_103a.cubin``.
# The ``[af]`` group must stay: dropping the Blackwell ``a``/``f`` suffix yields the wrong arch.
_ELF_ARCH = re.compile(r"\.sm_(\d+)([af]?)\.cubin")


def _runtime():
    """Return the cuda-python runtime module, or None when it is unavailable."""
    try:
        from cuda.bindings import runtime
    except ImportError:
        try:  # cuda-python < 12.8
            from cuda import cudart as runtime
        except ImportError:
            return None
    return runtime


def cuda_available() -> bool:
    """Return True when at least one CUDA device is visible."""
    runtime = _runtime()
    if runtime is None:
        return False
    error, count = runtime.cudaGetDeviceCount()
    return int(error) == 0 and count > 0


def synchronize(output: Any = None) -> None:
    """Block until all device work, including ``output``, has completed.

    ``output`` is duck-typed for host-async frameworks; the device synchronization
    covers eager ones. Without cuda-python that synchronization is skipped, so timings
    taken around this call are not trustworthy.
    """
    _block_until_ready(output)
    runtime = _runtime()
    if runtime is None:
        return
    (error,) = runtime.cudaDeviceSynchronize()
    if int(error) != 0:
        # Must raise: a sticky error (illegal access in the kernel under test)
        # surfaces here, and swallowing it records a fast, meaningless timing.
        raise RuntimeError(f"cudaDeviceSynchronize failed: {error}")


def _block_until_ready(value: Any) -> None:
    """Block on JAX-style arrays without importing jax."""
    if value is None:
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready(item)


def profiler_start() -> bool:
    """Start CUDA profiler capture, reporting whether it actually started."""
    runtime = _runtime()
    if runtime is None:
        return False
    error = runtime.cudaProfilerStart()
    error = error[0] if isinstance(error, tuple) else error
    return int(error) == 0


def profiler_stop() -> bool:
    """Stop CUDA profiler capture, reporting whether it stopped cleanly."""
    runtime = _runtime()
    if runtime is None:
        return False
    error = runtime.cudaProfilerStop()
    error = error[0] if isinstance(error, tuple) else error
    return int(error) == 0


def device_metadata() -> dict[str, Any]:
    """Describe the visible CUDA devices without consulting any framework."""
    runtime = _runtime()
    if runtime is None:
        return {"available": False, "reason": "cuda-python is not installed"}

    error, count = runtime.cudaGetDeviceCount()
    if int(error) != 0 or count == 0:
        return {"available": False, "device_count": 0}

    error, current = runtime.cudaGetDevice()
    if int(error) != 0:
        return {"available": False, "reason": f"cudaGetDevice failed: {error}"}

    devices = []
    for index in range(count):
        error, props = runtime.cudaGetDeviceProperties(index)
        if int(error) != 0:
            continue
        devices.append(
            {
                "index": index,
                "name": bytes(props.name).decode(errors="replace").rstrip("\x00"),
                "total_memory": int(props.totalGlobalMem),
                "compute_capability": [int(props.major), int(props.minor)],
                "multi_processor_count": int(props.multiProcessorCount),
                "uuid": bytes(props.uuid.bytes).hex(),
            }
        )
    return {
        "available": True,
        "device_count": count,
        "current_device": int(current),
        "devices": devices,
    }


def device_architecture() -> str | None:
    """Return the current device's arch as ``sm_<major*10+minor>``.

    Never raises: any failure yields None, which callers must treat as "unknown" rather
    than as a proven architecture mismatch.
    """
    runtime = _runtime()
    if runtime is None:
        return None
    error, count = runtime.cudaGetDeviceCount()
    if int(error) != 0 or count == 0:
        return None
    error, current = runtime.cudaGetDevice()
    if int(error) != 0:
        return None
    error, props = runtime.cudaGetDeviceProperties(current)
    if int(error) != 0:
        return None
    return f"sm_{int(props.major) * 10 + int(props.minor)}"


@functools.lru_cache(maxsize=None)
def build_architectures() -> dict[str, Any]:
    """Return the SASS architectures embedded in ``libtransformer_engine.so``.

    Reads the loaded shared object with ``cuobjdump`` rather than trusting
    ``NVTE_CUDA_ARCHS``, so it also holds for libraries built elsewhere. Cached because
    ``cuobjdump`` takes seconds. Never raises: any failure yields
    ``{"available": False, "reason": ...}``, which callers must treat as "unknown"
    rather than as a proven architecture mismatch.
    """
    from transformer_engine.common import _get_shared_object_file

    try:
        so_path = _get_shared_object_file("core")
    except (OSError, RuntimeError) as exc:
        return {"available": False, "reason": f"could not locate libtransformer_engine.so: {exc}"}

    exe = shutil.which("cuobjdump")
    if exe is None:
        return {"available": False, "reason": "cuobjdump not found (CUDA toolkit absent)"}

    try:
        result = subprocess.run(
            [exe, "--list-elf", str(so_path)],
            capture_output=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"cuobjdump failed: {exc}"}

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return {"available": False, "reason": f"cuobjdump exited {result.returncode}: {stderr}"}

    archs = sorted(
        {f"sm_{m.group(1)}{m.group(2)}" for m in _ELF_ARCH.finditer(result.stdout)},
        key=lambda s: (int(re.sub(r"\D", "", s)), s),
    )
    return {"available": True, "library": str(so_path), "cuda_architectures": archs}
