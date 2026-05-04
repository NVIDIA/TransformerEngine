# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


"""Unified GEMM benchmark for BF16, FP8 (Current/Delayed/Block), MXFP8, and NVFP4 precisions.

Compares matrix-multiplication throughput across precisions using
Transformer Engine on NVIDIA GPUs.  Supports two timing back-ends,
pre-quantized and autocast quantization modes, arbitrary MxKxN matrix
shapes, Nsight Systems profiling integration, and bar-chart output.

Timing back-ends
----------------
* **cuda-events** -- CUDA event pairs with a leading-kernel trick to
  hide CPU dispatch latency.  Measures the full GPU-side duration of
  the timed loop (includes quantisation when using autocast mode).
* **profiler** -- ``torch.profiler`` (CUPTI) kernel timestamps.
  Only the matched GEMM compute kernels (gemm, nvjet, xmma, cutlass)
  are summed, giving a kernel-only measurement.

Usage examples::

    # Kernel-only timing via torch.profiler:
    python benchmarks/gemm/benchmark_gemm.py --timing profiler --pre-quantize -o kernel.png

    # End-to-end timing via CUDA events:
    python benchmarks/gemm/benchmark_gemm.py --timing cuda-events -o e2e.png

    # Custom non-square shapes:
    python benchmarks/gemm/benchmark_gemm.py --shapes 88064x2560x10240,88064x10240x2560

    # Nsight profiling of a single shape:
    nsys profile --capture-range=cudaProfilerApi \\
        python benchmarks/gemm/benchmark_gemm.py --profile --profile-shape 4096

    # Model config mode (derives all 12 GEMM shapes from hyperparameters):
    python benchmarks/gemm/benchmark_gemm.py \\
        --hidden_size 4096 --intermediate_size 16384 \\
        --num_attention_heads 32 --num_hidden_layers 24 \\
        --micro_batch_size 31 --sequence_length 512
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

try:
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import (
        DelayedScaling,
        Float8BlockScaling,
        Float8CurrentScaling,
        Format,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
    )

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


GEMM_KERNEL_PATTERNS = ("gemm", "nvjet", "xmma", "cutlass")

PRECISION_COLORS = {
    "BF16": "#808080",
    "FP8Current": "#2E8B57",
    "FP8Delayed": "#20B2AA",
    "FP8Block": "#006400",
    "MXFP8": "#4B0082",
    "NVFP4": "#B22222",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GEMMResult:
    """Single GEMM benchmark measurement."""

    tflops: float
    avg_time_ms: float
    shape: tuple[int, int, int]
    precision: str


@dataclass
class ModelConfig:
    """Transformer model hyperparameters for GEMM shape derivation."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    micro_batch_size: int
    sequence_length: int


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------
def is_blackwell_available() -> bool:
    """Return True when the current device is Blackwell (SM100+) for NVFP4 support."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def compute_gemm_flops(M: int, K: int, N: int) -> int:
    """Theoretical FLOP count for C = A @ B:  2 * M * N * K."""
    return 2 * M * N * K


# ---------------------------------------------------------------------------
# torch.profiler helpers  (kernel-only timing)
# ---------------------------------------------------------------------------
def _is_gemm_kernel(name: str) -> bool:
    """Return True when *name* looks like a GEMM compute kernel."""
    low = name.lower()
    return any(p in low for p in GEMM_KERNEL_PATTERNS)


def _extract_gemm_kernel_time_us(
    prof_result: profile,
    num_iters: int,
    verbose: bool = False,
) -> float:
    """Average GEMM-kernel time in microseconds from profiler events."""
    total_us = 0.0
    count = 0
    seen: dict[str, float] = {}

    for evt in prof_result.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and _is_gemm_kernel(evt.name):
            total_us += evt.device_time
            count += 1
            seen[evt.name] = seen.get(evt.name, 0.0) + evt.device_time

    if verbose and seen:
        print(f"    Matched GEMM kernels ({count} invocations):")
        for kname, kus in seen.items():
            print(f"      {kname}: {kus:.0f} us total")

    if count == 0:
        if verbose:
            print("    WARNING: No GEMM kernels found.  All CUDA events:")
            for evt in prof_result.events():
                if evt.device_type == torch.autograd.DeviceType.CUDA:
                    print(f"      {evt.name}: {evt.device_time:.0f} us")
        return 0.0

    return total_us / num_iters


# ---------------------------------------------------------------------------
# Timing wrappers
# ---------------------------------------------------------------------------
def _time_with_profiler(
    run_fn,
    num_iters: int,
    flops: int,
    verbose: bool = False,
) -> tuple[float, float]:
    """Return (tflops, avg_ms) using torch.profiler kernel extraction."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(num_iters):
            run_fn()
        torch.cuda.synchronize()

    avg_us = _extract_gemm_kernel_time_us(prof, num_iters, verbose=verbose)
    avg_s = avg_us / 1e6
    tflops = (flops / avg_s) / 1e12 if avg_s > 0 else 0.0
    return tflops, avg_us / 1000.0


def _time_with_cuda_events(
    run_fn,
    num_iters: int,
    flops: int,
    leading_fn=None,
) -> tuple[float, float]:
    """Return (tflops, avg_ms) using CUDA events with optional leading kernel."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if leading_fn is not None:
        leading_fn()

    start.record()
    for _ in range(num_iters):
        run_fn()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / num_iters
    avg_s = avg_ms / 1000.0
    tflops = (flops / avg_s) / 1e12 if avg_s > 0 else 0.0
    return tflops, avg_ms


# ---------------------------------------------------------------------------
# BF16 benchmark
# ---------------------------------------------------------------------------
def benchmark_bf16(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> GEMMResult:
    """Benchmark BF16 torch.matmul."""
    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    for _ in range(num_warmup):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    def _run():
        torch.matmul(A, B)

    if timing == "profiler":
        tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
    else:
        A_lg = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        B_lg = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
        tflops, avg_ms = _time_with_cuda_events(
            _run, num_iters, flops, leading_fn=lambda: torch.matmul(A_lg, B_lg)
        )
        del A_lg, B_lg

    return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="BF16")


# ---------------------------------------------------------------------------
# FP8 tensor-wise scaling benchmarks (CurrentScaling / DelayedScaling)
# ---------------------------------------------------------------------------
def benchmark_fp8_current(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """FP8 GEMM with Float8CurrentScaling recipe via te.Linear autocast."""
    if not TE_AVAILABLE:
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    recipe = Float8CurrentScaling()

    with te.autocast(enabled=True, recipe=recipe):
        for _ in range(num_warmup):
            linear(x)
        torch.cuda.synchronize()

        def _run():
            linear(x)

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            lin_lg = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
            x_lg = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            tflops, avg_ms = _time_with_cuda_events(
                _run, num_iters, flops, leading_fn=lambda: lin_lg(x_lg)
            )
            del lin_lg, x_lg

    return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="FP8Current")


def benchmark_fp8_current_prequantized(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """Pre-quantized FP8 GEMM with Float8CurrentScaling via tex.generic_gemm."""
    if not TE_AVAILABLE:
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    try:
        quantizer = te.Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=device)

        A_q = quantizer.quantize(torch.randn(K, M, dtype=torch.bfloat16, device=device))
        B_q = quantizer.quantize(torch.randn(K, N, dtype=torch.bfloat16, device=device))
        D = torch.empty(N, M, dtype=torch.bfloat16, device=device)
        ws_size = 32 * 1024 * 1024
        ws = torch.empty(ws_size, dtype=torch.uint8, device=device)

        def _run():
            tex.generic_gemm(
                A_q,
                False,
                B_q,
                True,
                D,
                None,
                tex.DType.kBFloat16,
                None,
                tex.DType.kBFloat16,
                False,
                None,
                False,
                ws,
                ws_size,
                False,
                False,
            )

        for _ in range(num_warmup):
            _run()
        torch.cuda.synchronize()

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            A_lg_q = quantizer.quantize(
                torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            )
            B_lg_q = quantizer.quantize(
                torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            )
            D_lg = torch.empty(4096, 4096, dtype=torch.bfloat16, device=device)

            def _lead():
                tex.generic_gemm(
                    A_lg_q,
                    False,
                    B_lg_q,
                    True,
                    D_lg,
                    None,
                    tex.DType.kBFloat16,
                    None,
                    tex.DType.kBFloat16,
                    False,
                    None,
                    False,
                    ws,
                    ws_size,
                    False,
                    False,
                )

            tflops, avg_ms = _time_with_cuda_events(_run, num_iters, flops, leading_fn=_lead)
            del A_lg_q, B_lg_q, D_lg

        return GEMMResult(
            tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="FP8Current"
        )
    except Exception as e:
        print(f"Warning: FP8 CurrentScaling prequantized benchmark failed: {e}")
        return None


def benchmark_fp8_delayed(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """FP8 GEMM with DelayedScaling recipe via te.Linear autocast."""
    if not TE_AVAILABLE:
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    recipe = DelayedScaling()

    with te.autocast(enabled=True, recipe=recipe):
        for _ in range(num_warmup):
            linear(x)
        torch.cuda.synchronize()

        def _run():
            linear(x)

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            lin_lg = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
            x_lg = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            tflops, avg_ms = _time_with_cuda_events(
                _run, num_iters, flops, leading_fn=lambda: lin_lg(x_lg)
            )
            del lin_lg, x_lg

    return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="FP8Delayed")


# ---------------------------------------------------------------------------
# MXFP8 benchmarks
# ---------------------------------------------------------------------------
def benchmark_fp8(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """MXFP8 GEMM via te.Linear autocast."""
    if not TE_AVAILABLE:
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    with te.autocast(enabled=True, recipe=recipe):
        for _ in range(num_warmup):
            linear(x)
        torch.cuda.synchronize()

        def _run():
            linear(x)

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            lin_lg = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
            x_lg = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            tflops, avg_ms = _time_with_cuda_events(
                _run, num_iters, flops, leading_fn=lambda: lin_lg(x_lg)
            )
            del lin_lg, x_lg

    return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="MXFP8")


def benchmark_fp8_prequantized(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """Pre-quantized MXFP8 GEMM via tex.generic_gemm (raw kernel throughput)."""
    if not TE_AVAILABLE:
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    try:
        quantizer = te.MXFP8Quantizer(tex.DType.kFloat8E4M3)

        # tex.generic_gemm uses column-major convention: A=(K,M), B=(K,N),
        # D=(N,M) with transa=False, transb=True for a logical C(M,N) GEMM.
        A_q = quantizer.quantize(torch.randn(K, M, dtype=torch.bfloat16, device=device))
        B_q = quantizer.quantize(torch.randn(K, N, dtype=torch.bfloat16, device=device))
        D = torch.empty(N, M, dtype=torch.bfloat16, device=device)
        ws_size = 32 * 1024 * 1024
        ws = torch.empty(ws_size, dtype=torch.uint8, device=device)

        def _run():
            tex.generic_gemm(
                A_q,
                False,
                B_q,
                True,
                D,
                None,
                tex.DType.kBFloat16,
                None,
                tex.DType.kBFloat16,
                False,
                None,
                False,
                ws,
                ws_size,
                False,
                False,
            )

        for _ in range(num_warmup):
            _run()
        torch.cuda.synchronize()

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            A_lg_q = quantizer.quantize(
                torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            )
            B_lg_q = quantizer.quantize(
                torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            )
            D_lg = torch.empty(4096, 4096, dtype=torch.bfloat16, device=device)

            def _lead():
                tex.generic_gemm(
                    A_lg_q,
                    False,
                    B_lg_q,
                    True,
                    D_lg,
                    None,
                    tex.DType.kBFloat16,
                    None,
                    tex.DType.kBFloat16,
                    False,
                    None,
                    False,
                    ws,
                    ws_size,
                    False,
                    False,
                )

            tflops, avg_ms = _time_with_cuda_events(_run, num_iters, flops, leading_fn=_lead)
            del A_lg_q, B_lg_q, D_lg

        return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="MXFP8")
    except Exception as e:
        print(f"Warning: FP8 prequantized benchmark failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Float8 Block-Scaling benchmarks
# ---------------------------------------------------------------------------
def benchmark_fp8_block(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """FP8 GEMM with Float8BlockScaling recipe via te.Linear autocast."""
    if not TE_AVAILABLE:
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    recipe = Float8BlockScaling(fp8_format=Format.E4M3)

    with te.autocast(enabled=True, recipe=recipe):
        for _ in range(num_warmup):
            linear(x)
        torch.cuda.synchronize()

        def _run():
            linear(x)

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            lin_lg = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
            x_lg = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            tflops, avg_ms = _time_with_cuda_events(
                _run, num_iters, flops, leading_fn=lambda: lin_lg(x_lg)
            )
            del lin_lg, x_lg

    return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="FP8Block")


def benchmark_fp8_block_prequantized(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """Pre-quantized FP8 GEMM with Float8BlockScaling via tex.generic_gemm."""
    if not TE_AVAILABLE:
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    try:
        quantizer = te.Float8BlockQuantizer(
            tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
        )

        A_q = quantizer.quantize(torch.randn(K, M, dtype=torch.bfloat16, device=device))
        B_q = quantizer.quantize(torch.randn(K, N, dtype=torch.bfloat16, device=device))
        D = torch.empty(N, M, dtype=torch.bfloat16, device=device)
        ws_size = 32 * 1024 * 1024
        ws = torch.empty(ws_size, dtype=torch.uint8, device=device)

        def _run():
            tex.generic_gemm(
                A_q,
                False,
                B_q,
                True,
                D,
                None,
                tex.DType.kBFloat16,
                None,
                tex.DType.kBFloat16,
                False,
                None,
                False,
                ws,
                ws_size,
                False,
                False,
            )

        for _ in range(num_warmup):
            _run()
        torch.cuda.synchronize()

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            A_lg_q = quantizer.quantize(
                torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            )
            B_lg_q = quantizer.quantize(
                torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            )
            D_lg = torch.empty(4096, 4096, dtype=torch.bfloat16, device=device)

            def _lead():
                tex.generic_gemm(
                    A_lg_q,
                    False,
                    B_lg_q,
                    True,
                    D_lg,
                    None,
                    tex.DType.kBFloat16,
                    None,
                    tex.DType.kBFloat16,
                    False,
                    None,
                    False,
                    ws,
                    ws_size,
                    False,
                    False,
                )

            tflops, avg_ms = _time_with_cuda_events(_run, num_iters, flops, leading_fn=_lead)
            del A_lg_q, B_lg_q, D_lg

        return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="FP8Block")
    except Exception as e:
        print(f"Warning: FP8 Block-Scaling prequantized benchmark failed: {e}")
        return None


# ---------------------------------------------------------------------------
# NVFP4 benchmarks  (Blackwell SM100+ only)
# ---------------------------------------------------------------------------
def benchmark_fp4(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """NVFP4 GEMM via te.Linear autocast (Blackwell only)."""
    if not TE_AVAILABLE or not is_blackwell_available():
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    recipe = NVFP4BlockScaling(fp4_format=Format.E2M1)

    with te.autocast(enabled=True, recipe=recipe):
        for _ in range(num_warmup):
            linear(x)
        torch.cuda.synchronize()

        def _run():
            linear(x)

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            lin_lg = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).to(device)
            x_lg = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            tflops, avg_ms = _time_with_cuda_events(
                _run, num_iters, flops, leading_fn=lambda: lin_lg(x_lg)
            )
            del lin_lg, x_lg

    return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="NVFP4")


def benchmark_fp4_prequantized(
    M: int,
    K: int,
    N: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    timing: str = "cuda-events",
    verbose: bool = False,
) -> Optional[GEMMResult]:
    """Pre-quantized NVFP4 GEMM via tex.generic_gemm (Blackwell only)."""
    if not TE_AVAILABLE or not is_blackwell_available():
        return None

    device = torch.device("cuda")
    flops = compute_gemm_flops(M, K, N)

    try:
        quantizer = te.NVFP4Quantizer(tex.DType.kFloat4E2M1)

        # tex.generic_gemm uses column-major convention: A=(K,M), B=(K,N),
        # D=(N,M) with transa=False, transb=True for a logical C(M,N) GEMM.
        A_q = quantizer.quantize(torch.randn(K, M, dtype=torch.bfloat16, device=device))
        B_q = quantizer.quantize(torch.randn(K, N, dtype=torch.bfloat16, device=device))
        D = torch.empty(N, M, dtype=torch.bfloat16, device=device)
        ws_size = 32 * 1024 * 1024
        ws = torch.empty(ws_size, dtype=torch.uint8, device=device)

        def _run():
            tex.generic_gemm(
                A_q,
                False,
                B_q,
                True,
                D,
                None,
                tex.DType.kBFloat16,
                None,
                tex.DType.kBFloat16,
                False,
                None,
                False,
                ws,
                ws_size,
                False,
                False,
            )

        for _ in range(num_warmup):
            _run()
        torch.cuda.synchronize()

        if timing == "profiler":
            tflops, avg_ms = _time_with_profiler(_run, num_iters, flops, verbose=verbose)
        else:
            A_lg_q = quantizer.quantize(
                torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            )
            B_lg_q = quantizer.quantize(
                torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            )
            D_lg = torch.empty(4096, 4096, dtype=torch.bfloat16, device=device)

            def _lead():
                tex.generic_gemm(
                    A_lg_q,
                    False,
                    B_lg_q,
                    True,
                    D_lg,
                    None,
                    tex.DType.kBFloat16,
                    None,
                    tex.DType.kBFloat16,
                    False,
                    None,
                    False,
                    ws,
                    ws_size,
                    False,
                    False,
                )

            tflops, avg_ms = _time_with_cuda_events(_run, num_iters, flops, leading_fn=_lead)
            del A_lg_q, B_lg_q, D_lg

        return GEMMResult(tflops=tflops, avg_time_ms=avg_ms, shape=(M, K, N), precision="NVFP4")
    except Exception as e:
        print(f"Warning: FP4 prequantized benchmark failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------
def get_default_shapes() -> list[tuple[int, int, int]]:
    """Default set of square matrix shapes for benchmarking."""
    return [
        (256, 256, 256),
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (3072, 3072, 3072),
        (4096, 4096, 4096),
        (6144, 6144, 6144),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ]


def parse_shapes_arg(shapes_arg: str) -> list[tuple[int, int, int]]:
    """Parse ``--shapes`` into a list of (M, K, N) tuples.

    Accepts either square sizes (``1024,2048,4096``) or explicit
    triplets (``8192x5120x10240,8192x10240x5120``), or a mix.

    Raises:
        ValueError: On malformed input.
    """
    items = [s.strip() for s in shapes_arg.split(",") if s.strip()]
    if not items:
        raise ValueError("Empty --shapes argument.")

    shapes: list[tuple[int, int, int]] = []
    for item in items:
        if "x" in item:
            parts = [p.strip() for p in item.lower().split("x")]
            if len(parts) != 3:
                raise ValueError(f"Invalid shape '{item}'.  Expected 'MxKxN'.")
            shapes.append((int(parts[0]), int(parts[1]), int(parts[2])))
        else:
            size = int(item)
            shapes.append((size, size, size))
    return shapes


def compute_gemm_shapes(
    config: ModelConfig,
) -> tuple[
    list[tuple[str, int, int, int]],
    list[tuple[str, int, int, int]],
    list[tuple[str, int, int, int]],
]:
    """Derive Fprop, Dgrad, and Wgrad GEMM shapes from a transformer model config.

    For forward Y = X @ W with shape (M, K, N):
      - Dgrad: dX = dY @ Wᵀ  →  (M, N, K)  (K and N swap)
      - Wgrad: dW = Xᵀ @ dY  →  (K, M, N)  (M moves to contraction axis)

    Returns:
        (fprop_shapes, dgrad_shapes, wgrad_shapes) where each is a list of
        (label, M, K, N) tuples.
    """
    H = config.hidden_size
    I = config.intermediate_size
    M = config.micro_batch_size * config.sequence_length

    if H % config.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({H}) must be divisible by "
            f"num_attention_heads ({config.num_attention_heads})"
        )

    N_qkv = 3 * H

    fprop_shapes = [
        ("QKV Proj", M, H, N_qkv),
        ("Attn Out", M, H, H),
        ("MLP Up", M, H, I),
        ("MLP Down", M, I, H),
    ]

    dgrad_shapes = [
        ("QKV Proj (Dgrad)", M, N_qkv, H),
        ("Attn Out (Dgrad)", M, H, H),
        ("MLP Up (Dgrad)", M, I, H),
        ("MLP Down (Dgrad)", M, H, I),
    ]

    wgrad_shapes = [
        ("QKV Proj (Wgrad)", H, M, N_qkv),
        ("Attn Out (Wgrad)", H, M, H),
        ("MLP Up (Wgrad)", H, M, I),
        ("MLP Down (Wgrad)", I, M, H),
    ]

    return fprop_shapes, dgrad_shapes, wgrad_shapes


# ---------------------------------------------------------------------------
# GPU warmup
# ---------------------------------------------------------------------------
def warmup_gpu(duration_seconds: float = 5.0) -> None:
    """Run sustained matmuls to stabilize GPU clocks before benchmarking."""
    print(f"Warming up GPU for {duration_seconds:.1f} seconds...")
    device = torch.device("cuda")
    A = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    B = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    while time.time() - t0 < duration_seconds:
        for _ in range(10):
            torch.matmul(A, B)
        torch.cuda.synchronize()

    del A, B
    torch.cuda.empty_cache()
    print("GPU warmup complete.\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_benchmarks(
    shapes: list[tuple[int, int, int]],
    num_warmup: int = 10,
    num_iters: int = 100,
    include_fp8_current: bool = True,
    include_fp8_delayed: bool = True,
    include_fp8: bool = True,
    include_fp8_block: bool = True,
    include_fp4: bool = True,
    gpu_warmup_seconds: float = 5.0,
    pre_quantize: bool = False,
    timing: str = "cuda-events",
    profile_shape: Optional[int] = None,
) -> dict[str, list[float]]:
    """Run GEMM benchmarks for every shape and enabled precision.

    Returns:
        Dict mapping precision name to a list of TFLOPS values, one per shape.
    """
    results: dict[str, list[float]] = {
        "BF16": [],
        "FP8Current": [],
        "FP8Delayed": [],
        "FP8Block": [],
        "MXFP8": [],
        "NVFP4": [],
    }
    time_results: dict[str, list[float]] = {
        "BF16": [],
        "FP8Current": [],
        "FP8Delayed": [],
        "FP8Block": [],
        "MXFP8": [],
        "NVFP4": [],
    }

    has_blackwell = is_blackwell_available()
    run_fp8_current = include_fp8_current and TE_AVAILABLE
    run_fp8_delayed = include_fp8_delayed and TE_AVAILABLE
    run_fp8 = include_fp8 and TE_AVAILABLE
    run_fp8_block = include_fp8_block and TE_AVAILABLE
    run_fp4 = include_fp4 and TE_AVAILABLE and has_blackwell

    gpu_name = torch.cuda.get_device_name(0)
    timing_label = (
        "torch.profiler (CUPTI kernel timestamps)" if timing == "profiler" else "CUDA events"
    )

    print(f"\nGEMM Benchmark on {gpu_name}")
    print(f"Timing method: {timing_label}")
    print(f"Warmup iterations: {num_warmup}, Timed iterations: {num_iters}")
    if pre_quantize:
        print("Mode: Pre-quantized inputs (raw kernel throughput)")
    else:
        print("Mode: Autocast (includes quantization overhead)")
    if not has_blackwell and include_fp4:
        print("Note: NVFP4 requires Blackwell (SM100+), skipping FP4 benchmarks")

    if profile_shape is not None:
        shapes = [(profile_shape, profile_shape, profile_shape)]
        print(f"\n*** PROFILING MODE: shape {profile_shape}x{profile_shape}x{profile_shape} ***")
        print(
            "*** Run with: nsys profile --capture-range=cudaProfilerApi python <script> --profile"
            " ***\n"
        )

    if gpu_warmup_seconds > 0:
        warmup_gpu(gpu_warmup_seconds)

    # Select benchmark functions
    fp8_current_fn = benchmark_fp8_current_prequantized if pre_quantize else benchmark_fp8_current
    fp8_delayed_fn = benchmark_fp8_delayed  # No prequantized variant (uses amax history)
    fp8_block_fn = benchmark_fp8_block_prequantized if pre_quantize else benchmark_fp8_block
    fp8_fn = benchmark_fp8_prequantized if pre_quantize else benchmark_fp8
    fp4_fn = benchmark_fp4_prequantized if pre_quantize else benchmark_fp4

    # Print table header
    sep_width = 90
    print("=" * sep_width)
    hdr = f"{'Shape':<24} {'BF16 TFLOPS':>12} {'BF16 ms':>9}"
    if run_fp8_current:
        hdr += f" {'FP8Cur TFLOPS':>14} {'FP8Cur ms':>10}"
    if run_fp8_delayed:
        hdr += f" {'FP8Del TFLOPS':>14} {'FP8Del ms':>10}"
    if run_fp8_block:
        hdr += f" {'FP8Block TFLOPS':>16} {'FP8Block ms':>11}"
    if run_fp8:
        hdr += f" {'MXFP8 TFLOPS':>13} {'MXFP8 ms':>9}"
    if run_fp4:
        hdr += f" {'NVFP4 TFLOPS':>13} {'NVFP4 ms':>9}"
    hdr += f" {'Speedup':>8}"
    print(hdr)
    print("-" * sep_width)

    first_shape = True
    for M, K, N in shapes:
        shape_str = f"{M}x{K}x{N}"
        verbose = first_shape and timing == "profiler"
        is_profiling = profile_shape is not None

        if is_profiling:
            torch.cuda.cudart().cudaProfilerStart()

        # BF16
        if verbose:
            print(f"\n  [{shape_str}] BF16 kernel details:")
        if is_profiling:
            torch.cuda.nvtx.range_push(f"BF16_{shape_str}")
        bf16 = benchmark_bf16(M, K, N, num_warmup, num_iters, timing=timing, verbose=verbose)
        if is_profiling:
            torch.cuda.nvtx.range_pop()
        results["BF16"].append(bf16.tflops)
        time_results["BF16"].append(bf16.avg_time_ms)
        row = f"{shape_str:<24} {bf16.tflops:>12.1f} {bf16.avg_time_ms:>9.3f}"
        best_tflops = bf16.tflops

        # FP8 CurrentScaling
        if run_fp8_current:
            if verbose:
                print(f"  [{shape_str}] FP8Current kernel details:")
            if is_profiling:
                torch.cuda.nvtx.range_push(f"FP8Current_{shape_str}")
            fp8c = fp8_current_fn(M, K, N, num_warmup, num_iters, timing=timing, verbose=verbose)
            if is_profiling:
                torch.cuda.nvtx.range_pop()
            if fp8c:
                results["FP8Current"].append(fp8c.tflops)
                time_results["FP8Current"].append(fp8c.avg_time_ms)
                row += f" {fp8c.tflops:>14.1f} {fp8c.avg_time_ms:>10.3f}"
                best_tflops = max(best_tflops, fp8c.tflops)
            else:
                results["FP8Current"].append(0)
                time_results["FP8Current"].append(0)
                row += f" {'N/A':>14} {'N/A':>10}"

        # FP8 DelayedScaling
        if run_fp8_delayed:
            if verbose:
                print(f"  [{shape_str}] FP8Delayed kernel details:")
            if is_profiling:
                torch.cuda.nvtx.range_push(f"FP8Delayed_{shape_str}")
            fp8d = fp8_delayed_fn(M, K, N, num_warmup, num_iters, timing=timing, verbose=verbose)
            if is_profiling:
                torch.cuda.nvtx.range_pop()
            if fp8d:
                results["FP8Delayed"].append(fp8d.tflops)
                time_results["FP8Delayed"].append(fp8d.avg_time_ms)
                row += f" {fp8d.tflops:>14.1f} {fp8d.avg_time_ms:>10.3f}"
                best_tflops = max(best_tflops, fp8d.tflops)
            else:
                results["FP8Delayed"].append(0)
                time_results["FP8Delayed"].append(0)
                row += f" {'N/A':>14} {'N/A':>10}"

        # FP8Block
        if run_fp8_block:
            if verbose:
                print(f"  [{shape_str}] FP8Block kernel details:")
            if is_profiling:
                torch.cuda.nvtx.range_push(f"FP8Block_{shape_str}")
            fp8b = fp8_block_fn(M, K, N, num_warmup, num_iters, timing=timing, verbose=verbose)
            if is_profiling:
                torch.cuda.nvtx.range_pop()
            if fp8b:
                results["FP8Block"].append(fp8b.tflops)
                time_results["FP8Block"].append(fp8b.avg_time_ms)
                row += f" {fp8b.tflops:>16.1f} {fp8b.avg_time_ms:>11.3f}"
                best_tflops = max(best_tflops, fp8b.tflops)
            else:
                results["FP8Block"].append(0)
                time_results["FP8Block"].append(0)
                row += f" {'N/A':>16} {'N/A':>11}"

        # MXFP8
        if run_fp8:
            if verbose:
                print(f"  [{shape_str}] MXFP8 kernel details:")
            if is_profiling:
                torch.cuda.nvtx.range_push(f"MXFP8_{shape_str}")
            fp8 = fp8_fn(M, K, N, num_warmup, num_iters, timing=timing, verbose=verbose)
            if is_profiling:
                torch.cuda.nvtx.range_pop()
            if fp8:
                results["MXFP8"].append(fp8.tflops)
                time_results["MXFP8"].append(fp8.avg_time_ms)
                row += f" {fp8.tflops:>13.1f} {fp8.avg_time_ms:>9.3f}"
                best_tflops = max(best_tflops, fp8.tflops)
            else:
                results["MXFP8"].append(0)
                time_results["MXFP8"].append(0)
                row += f" {'N/A':>13} {'N/A':>9}"

        # NVFP4
        if run_fp4:
            if verbose:
                print(f"  [{shape_str}] NVFP4 kernel details:")
            if is_profiling:
                torch.cuda.nvtx.range_push(f"NVFP4_{shape_str}")
            fp4 = fp4_fn(M, K, N, num_warmup, num_iters, timing=timing, verbose=verbose)
            if is_profiling:
                torch.cuda.nvtx.range_pop()
            if fp4:
                results["NVFP4"].append(fp4.tflops)
                time_results["NVFP4"].append(fp4.avg_time_ms)
                row += f" {fp4.tflops:>13.1f} {fp4.avg_time_ms:>9.3f}"
                best_tflops = max(best_tflops, fp4.tflops)
            else:
                results["NVFP4"].append(0)
                time_results["NVFP4"].append(0)
                row += f" {'N/A':>13} {'N/A':>9}"

        if is_profiling:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

        speedup = best_tflops / bf16.tflops if bf16.tflops > 0 else 0.0
        row += f" {speedup:>7.2f}x"

        if verbose:
            print()
        print(row)
        first_shape = False

    print("=" * sep_width)

    results = {k: v for k, v in results.items() if v and any(x > 0 for x in v)}
    return results


# ---------------------------------------------------------------------------
# Model-config benchmark orchestrator
# ---------------------------------------------------------------------------
def _benchmark_single_shape(
    M: int,
    K: int,
    N: int,
    num_warmup: int,
    num_iters: int,
    run_fp8_current: bool,
    run_fp8_delayed: bool,
    run_fp8: bool,
    run_fp8_block: bool,
    run_fp4: bool,
    pre_quantize: bool,
    timing: str,
) -> dict[str, GEMMResult]:
    """Benchmark one (M, K, N) shape across all enabled precisions.

    Returns:
        Dict mapping precision name to GEMMResult.
    """
    fp8_current_fn = benchmark_fp8_current_prequantized if pre_quantize else benchmark_fp8_current
    fp8_delayed_fn = benchmark_fp8_delayed  # No prequantized variant (uses amax history)
    fp8_fn = benchmark_fp8_prequantized if pre_quantize else benchmark_fp8
    fp8_block_fn = benchmark_fp8_block_prequantized if pre_quantize else benchmark_fp8_block
    fp4_fn = benchmark_fp4_prequantized if pre_quantize else benchmark_fp4

    out: dict[str, GEMMResult] = {}

    bf16 = benchmark_bf16(M, K, N, num_warmup, num_iters, timing=timing)
    out["BF16"] = bf16

    if run_fp8_current:
        fp8c = fp8_current_fn(M, K, N, num_warmup, num_iters, timing=timing)
        if fp8c:
            out["FP8Current"] = fp8c

    if run_fp8_delayed:
        fp8d = fp8_delayed_fn(M, K, N, num_warmup, num_iters, timing=timing)
        if fp8d:
            out["FP8Delayed"] = fp8d

    if run_fp8_block:
        fp8b = fp8_block_fn(M, K, N, num_warmup, num_iters, timing=timing)
        if fp8b:
            out["FP8Block"] = fp8b

    if run_fp8:
        fp8 = fp8_fn(M, K, N, num_warmup, num_iters, timing=timing)
        if fp8:
            out["MXFP8"] = fp8

    if run_fp4:
        fp4 = fp4_fn(M, K, N, num_warmup, num_iters, timing=timing)
        if fp4:
            out["NVFP4"] = fp4

    return out


def run_model_config_benchmarks(
    config: ModelConfig,
    num_warmup: int = 10,
    num_iters: int = 100,
    include_fp8_current: bool = True,
    include_fp8_delayed: bool = True,
    include_fp8: bool = True,
    include_fp8_block: bool = True,
    include_fp4: bool = True,
    gpu_warmup_seconds: float = 5.0,
    pre_quantize: bool = False,
    timing: str = "cuda-events",
    output_path: str = "gemm_benchmark.png",
) -> None:
    """Benchmark GEMM shapes derived from model hyperparameters.

    Computes Fprop, Dgrad, and Wgrad shapes, benchmarks each across
    enabled precisions, and prints per-layer / full-model speedup estimates.
    """
    has_blackwell = is_blackwell_available()
    run_fp8_current = include_fp8_current and TE_AVAILABLE
    run_fp8_delayed = include_fp8_delayed and TE_AVAILABLE
    run_fp8 = include_fp8 and TE_AVAILABLE
    run_fp8_block = include_fp8_block and TE_AVAILABLE
    run_fp4 = include_fp4 and TE_AVAILABLE and has_blackwell

    gpu_name = torch.cuda.get_device_name(0)
    timing_label = (
        "torch.profiler (CUPTI kernel timestamps)" if timing == "profiler" else "CUDA events"
    )

    M = config.micro_batch_size * config.sequence_length
    fprop_shapes, dgrad_shapes, wgrad_shapes = compute_gemm_shapes(config)

    # --- Header ---
    sep = "=" * 90
    dash = "-" * 90
    print(f"\nGEMM Benchmark (Model Config Mode) on {gpu_name}")
    print(f"Timing method: {timing_label}")
    print(f"Warmup iterations: {num_warmup}, Timed iterations: {num_iters}")
    if pre_quantize:
        print("Mode: Pre-quantized inputs (raw kernel throughput)")
    else:
        print("Mode: Autocast (includes quantization overhead)")
    if not has_blackwell and include_fp4:
        print("Note: NVFP4 requires Blackwell (SM100+), skipping FP4 benchmarks")
    print()
    print(sep)
    print(
        f"Model Config: hidden={config.hidden_size}, "
        f"intermediate={config.intermediate_size}, "
        f"heads={config.num_attention_heads}, "
        f"layers={config.num_hidden_layers}"
    )
    print(f"Tokens per step: M = {config.micro_batch_size} x {config.sequence_length} = {M:,}")
    print(sep)

    if gpu_warmup_seconds > 0:
        warmup_gpu(gpu_warmup_seconds)

    # --- Determine active precisions for column headers ---
    precisions = ["BF16"]
    if run_fp8_current:
        precisions.append("FP8Current")
    if run_fp8_delayed:
        precisions.append("FP8Delayed")
    if run_fp8_block:
        precisions.append("FP8Block")
    if run_fp8:
        precisions.append("MXFP8")
    if run_fp4:
        precisions.append("NVFP4")

    def _ms_cols(precs: list[str]) -> str:
        return "".join(f" {p + ' ms':>10}" for p in precs)

    def _ms_vals(results: dict[str, GEMMResult], precs: list[str]) -> str:
        parts = []
        for p in precs:
            if p in results:
                parts.append(f" {results[p].avg_time_ms:>10.3f}")
            else:
                parts.append(f" {'N/A':>10}")
        return "".join(parts)

    def _ms_sums(sums: dict[str, float], precs: list[str]) -> str:
        return "".join(f" {sums.get(p, 0):>10.3f}" for p in precs)

    # --- Benchmark Fprop shapes ---
    print("\nFprop Shapes:")
    print(dash)
    print(f"{'Op':<22} {'Shape':<24}{_ms_cols(precisions)}")
    print(dash)

    fprop_results: list[dict[str, GEMMResult]] = []
    fprop_sums: dict[str, float] = {p: 0.0 for p in precisions}

    for label, m, k, n in fprop_shapes:
        shape_str = f"{m}x{k}x{n}"
        res = _benchmark_single_shape(
            m,
            k,
            n,
            num_warmup,
            num_iters,
            run_fp8_current,
            run_fp8_delayed,
            run_fp8,
            run_fp8_block,
            run_fp4,
            pre_quantize,
            timing,
        )
        fprop_results.append(res)
        for p in precisions:
            if p in res:
                fprop_sums[p] += res[p].avg_time_ms
        print(f"{label:<22} {shape_str:<24}{_ms_vals(res, precisions)}")

    print(dash)
    print(f"{'Fprop sum (ms):':<46}{_ms_sums(fprop_sums, precisions)}")

    # --- Benchmark Dgrad shapes ---
    dgrad_results: list[dict[str, GEMMResult]] = []
    dgrad_sums: dict[str, float] = {p: 0.0 for p in precisions}

    print("\nDgrad Shapes:")
    print(dash)
    print(f"{'Op':<22} {'Shape':<24}{_ms_cols(precisions)}")
    print(dash)

    for label, m, k, n in dgrad_shapes:
        shape_str = f"{m}x{k}x{n}"
        res = _benchmark_single_shape(
            m,
            k,
            n,
            num_warmup,
            num_iters,
            run_fp8_current,
            run_fp8_delayed,
            run_fp8,
            run_fp8_block,
            run_fp4,
            pre_quantize,
            timing,
        )
        dgrad_results.append(res)
        for p in precisions:
            if p in res:
                dgrad_sums[p] += res[p].avg_time_ms
        print(f"{label:<22} {shape_str:<24}{_ms_vals(res, precisions)}")

    print(dash)
    print(f"{'Dgrad sum (ms):':<46}{_ms_sums(dgrad_sums, precisions)}")

    fprop_dgrad_sums: dict[str, float] = {
        p: fprop_sums.get(p, 0) + dgrad_sums.get(p, 0) for p in precisions
    }
    print(f"{'Fprop + Dgrad (measured):':<46}{_ms_sums(fprop_dgrad_sums, precisions)}")

    print("\nFprop vs Dgrad per-shape comparison:")
    print(dash)
    for p in precisions:
        print(f"  {p}:")
        for i, ((fp_label, *_), (_, *_)) in enumerate(zip(fprop_shapes, dgrad_shapes)):
            fp_res = fprop_results[i].get(p)
            dg_res = dgrad_results[i].get(p)
            if fp_res and dg_res:
                fp_ms = fp_res.avg_time_ms
                dg_ms = dg_res.avg_time_ms
                diff_pct = (dg_ms - fp_ms) / fp_ms * 100
                print(
                    f"    {fp_label:<16} Fprop={fp_ms:7.3f}ms  Dgrad={dg_ms:7.3f}ms "
                    f" diff={diff_pct:+.1f}%"
                )
        fp_total = fprop_sums.get(p, 0)
        dg_total = dgrad_sums.get(p, 0)
        if fp_total > 0:
            total_diff = (dg_total - fp_total) / fp_total * 100
            print(
                f"    {'Sum':<16} Fprop={fp_total:7.3f}ms  Dgrad={dg_total:7.3f}ms "
                f" diff={total_diff:+.1f}%"
            )

    # --- Benchmark Wgrad shapes ---
    print("\nWgrad Shapes:")
    print(dash)
    print(f"{'Op':<22} {'Shape':<24}{_ms_cols(precisions)}")
    print(dash)

    wgrad_results: list[dict[str, GEMMResult]] = []
    wgrad_sums: dict[str, float] = {p: 0.0 for p in precisions}

    for label, m, k, n in wgrad_shapes:
        shape_str = f"{m}x{k}x{n}"
        res = _benchmark_single_shape(
            m,
            k,
            n,
            num_warmup,
            num_iters,
            run_fp8_current,
            run_fp8_delayed,
            run_fp8,
            run_fp8_block,
            run_fp4,
            pre_quantize,
            timing,
        )
        wgrad_results.append(res)
        for p in precisions:
            if p in res:
                wgrad_sums[p] += res[p].avg_time_ms
        print(f"{label:<22} {shape_str:<24}{_ms_vals(res, precisions)}")

    print(dash)
    print(f"{'Wgrad sum (ms):':<46}{_ms_sums(wgrad_sums, precisions)}")

    # --- Per-layer and full-model summary ---
    per_layer = {p: fprop_dgrad_sums.get(p, 0) + wgrad_sums.get(p, 0) for p in precisions}
    full_model = {p: v * config.num_hidden_layers for p, v in per_layer.items()}

    print(f"\n{sep}")
    print("Per-Layer GEMM Time:")
    print(f"{'':>30}{_ms_cols(precisions)}")
    print(f"{'Fprop:':<30}{_ms_sums(fprop_sums, precisions)}")
    print(f"{'Dgrad:':<30}{_ms_sums(dgrad_sums, precisions)}")
    print(f"{'Fprop + Dgrad:':<30}{_ms_sums(fprop_dgrad_sums, precisions)}")
    print(f"{'Wgrad:':<30}{_ms_sums(wgrad_sums, precisions)}")
    print(f"{'Per-layer total:':<30}{_ms_sums(per_layer, precisions)}")

    print(f"\nFull Model ({config.num_hidden_layers} layers):")
    print(f"{'Total GEMM time (ms):':<30}{_ms_sums(full_model, precisions)}")

    print("\nEstimated GEMM Speedups:")
    bf16_total = full_model.get("BF16", 0)
    if run_fp8 and bf16_total > 0:
        fp8_total = full_model.get("MXFP8", 0)
        if fp8_total > 0:
            print(f"  MXFP8 vs BF16:  {bf16_total / fp8_total:.2f}x")
    if run_fp4 and run_fp8:
        fp8_total = full_model.get("MXFP8", 0)
        fp4_total = full_model.get("NVFP4", 0)
        if fp8_total > 0 and fp4_total > 0:
            print(f"  NVFP4 vs MXFP8: {fp8_total / fp4_total:.2f}x")
    if run_fp4 and bf16_total > 0:
        fp4_total = full_model.get("NVFP4", 0)
        if fp4_total > 0:
            print(f"  NVFP4 vs BF16:  {bf16_total / fp4_total:.2f}x")
    print(sep)

    # --- Plot ---
    create_model_config_plot(
        config,
        fprop_results,
        dgrad_results,
        wgrad_results,
        fprop_shapes,
        wgrad_shapes,
        precisions,
        output_path,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def create_plot(
    shapes: list[tuple[int, int, int]],
    results: dict[str, list[float]],
    output_path: str = "gemm_benchmark.png",
    title: Optional[str] = None,
) -> Optional[tuple]:
    """Create a grouped bar chart of TFLOPS by precision and save to *output_path*."""
    if not results:
        print("No results to plot.")
        return None

    gpu_name = torch.cuda.get_device_name(0)
    if title is None:
        title = f"Absolute Performance Comparison\nMeasured on {gpu_name}"

    labels = [f"{m}x{k}x{n}" for m, k, n in shapes]
    x = np.arange(len(labels))

    num_bars = len(results)
    bar_width = 0.8 / num_bars

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (prec, tflops_list) in enumerate(results.items()):
        offset = (i - num_bars / 2 + 0.5) * bar_width
        color = PRECISION_COLORS.get(prec, f"C{i}")
        ax.bar(x + offset, tflops_list, bar_width, label=prec, color=color)

    ax.set_xlabel("Matrix Shape (MxKxN)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend(title="Kernel", loc="upper left", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    out = Path(output_path)
    supported = set(fig.canvas.get_supported_filetypes().keys())
    suffix = out.suffix.lower().lstrip(".")
    if suffix not in supported:
        out = out.with_suffix(".png")
        print(f"Warning: '.{suffix}' not supported by matplotlib; saving to '{out}' instead.")

    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out}")
    return fig, ax


def create_model_config_plot(
    config: ModelConfig,
    fprop_results: list[dict[str, GEMMResult]],
    dgrad_results: list[dict[str, GEMMResult]],
    wgrad_results: list[dict[str, GEMMResult]],
    fprop_shapes: list[tuple[str, int, int, int]],
    wgrad_shapes: list[tuple[str, int, int, int]],
    precisions: list[str],
    output_path: str = "gemm_benchmark.png",
) -> Optional[tuple]:
    """Create a stacked bar chart of per-layer GEMM time by precision and op."""
    gpu_name = torch.cuda.get_device_name(0)
    op_labels = ["QKV Proj", "Attn Out", "MLP Up", "MLP Down"]
    op_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.5
    x = np.arange(len(precisions))

    for i, (op_label, op_color) in enumerate(zip(op_labels, op_colors)):
        fprop_ms = []
        wgrad_ms = []
        for p in precisions:
            fp = fprop_results[i].get(p)
            dg = dgrad_results[i].get(p)
            fprop_ms.append(
                (fp.avg_time_ms if fp else 0) + (dg.avg_time_ms if dg else 0)
            )  # Fprop + Dgrad (measured)
            wg = wgrad_results[i].get(p)
            wgrad_ms.append(wg.avg_time_ms if wg else 0)

        # Compute bottoms from prior ops
        fprop_bottom = np.zeros(len(precisions))
        wgrad_bottom = np.zeros(len(precisions))
        for j in range(i):
            for k, p in enumerate(precisions):
                fp_prev = fprop_results[j].get(p)
                dg_prev = dgrad_results[j].get(p)
                fprop_bottom[k] += (fp_prev.avg_time_ms if fp_prev else 0) + (
                    dg_prev.avg_time_ms if dg_prev else 0
                )
                wg_prev = wgrad_results[j].get(p)
                wgrad_bottom[k] += wg_prev.avg_time_ms if wg_prev else 0

        total_fprop_bottom = fprop_bottom
        total_wgrad_bottom = wgrad_bottom
        # Wgrad stacks on top of all Fprop+Dgrad
        all_fprop_total = np.zeros(len(precisions))
        for j in range(len(op_labels)):
            for k, p in enumerate(precisions):
                fp = fprop_results[j].get(p)
                dg = dgrad_results[j].get(p)
                all_fprop_total[k] += (fp.avg_time_ms if fp else 0) + (dg.avg_time_ms if dg else 0)

        ax.bar(
            x,
            fprop_ms,
            bar_width,
            bottom=total_fprop_bottom,
            color=op_color,
            alpha=0.9,
            label=f"{op_label} (Fprop+Dgrad)",
        )
        ax.bar(
            x,
            wgrad_ms,
            bar_width,
            bottom=all_fprop_total + total_wgrad_bottom,
            color=op_color,
            alpha=0.5,
            label=f"{op_label} (Wgrad)",
        )

    # Speedup annotations
    totals = []
    for k, p in enumerate(precisions):
        total = 0
        for i in range(len(op_labels)):
            fp = fprop_results[i].get(p)
            dg = dgrad_results[i].get(p)
            total += (fp.avg_time_ms if fp else 0) + (dg.avg_time_ms if dg else 0)
            wg = wgrad_results[i].get(p)
            total += wg.avg_time_ms if wg else 0
        totals.append(total)

    bf16_total = totals[0] if totals else 0
    for k, (p, total) in enumerate(zip(precisions, totals)):
        if bf16_total > 0 and total > 0:
            speedup = bf16_total / total
            ax.text(
                x[k],
                total + 0.02 * max(totals),
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

    ax.set_xlabel("Precision", fontsize=12)
    ax.set_ylabel("Per-Layer GEMM Time (ms)", fontsize=12)
    ax.set_title(
        f"Per-Layer GEMM Time Breakdown\n{gpu_name} | "
        f"hidden={config.hidden_size}, layers={config.num_hidden_layers}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(precisions, fontsize=11)
    ax.legend(
        loc="upper right",
        fontsize=8,
        ncol=2,
        title="Operation (pass)",
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    out = Path(output_path)
    supported = set(fig.canvas.get_supported_filetypes().keys())
    suffix = out.suffix.lower().lstrip(".")
    if suffix not in supported:
        out = out.with_suffix(".png")

    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out}")
    return fig, ax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    """Entry-point for the unified GEMM benchmark."""
    parser = argparse.ArgumentParser(
        description="Unified GEMM benchmark for BF16 / MXFP8 / NVFP4 precisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="gemm_benchmark.png",
        help="Output plot path (default: %(default)s)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Per-kernel warmup iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--num-iters", type=int, default=100, help="Timed iterations (default: %(default)s)"
    )
    parser.add_argument(
        "--gpu-warmup",
        type=float,
        default=5.0,
        help=(
            "Seconds of sustained matmuls to stabilize GPU clocks (default: %(default)s, 0 to"
            " disable)"
        ),
    )
    parser.add_argument(
        "--no-fp8-current", action="store_true", help="Skip Float8CurrentScaling benchmarks"
    )
    parser.add_argument(
        "--no-fp8-delayed", action="store_true", help="Skip DelayedScaling benchmarks"
    )
    parser.add_argument("--no-fp8", action="store_true", help="Skip MXFP8 benchmarks")
    parser.add_argument(
        "--no-fp8-block", action="store_true", help="Skip Float8BlockScaling benchmarks"
    )
    parser.add_argument("--no-fp4", action="store_true", help="Skip NVFP4 benchmarks")
    parser.add_argument(
        "--pre-quantize",
        action="store_true",
        help="Use pre-quantized inputs (tex.generic_gemm) instead of te.Linear autocast",
    )
    parser.add_argument(
        "--timing",
        choices=["cuda-events", "profiler"],
        default="cuda-events",
        help=(
            "Timing back-end: 'cuda-events' uses CUDA event pairs with a leading-kernel trick;"
            " 'profiler' uses torch.profiler to extract GEMM-kernel-only times (default:"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--shapes",
        default=None,
        help=(
            "Comma-separated GEMM shapes.  Square sizes like '1024,2048,4096' or "
            "explicit MxKxN triplets like '8192x5120x10240,8192x10240x5120'"
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Nsight profiling mode: benchmark only --profile-shape, emit CUDA profiler "
            "start/stop and NVTX ranges.  Run with: nsys profile "
            "--capture-range=cudaProfilerApi python benchmarks/gemm/benchmark_gemm.py --profile"
        ),
    )
    parser.add_argument(
        "--profile-shape",
        type=int,
        default=4096,
        help="Square matrix size used in --profile mode (default: %(default)s)",
    )

    # Model configuration arguments
    model_group = parser.add_argument_group(
        "Model Configuration",
        "Specify model hyperparameters to automatically derive and benchmark "
        "all GEMM shapes for a transformer model.  Mutually exclusive with --shapes.",
    )
    model_group.add_argument(
        "--hidden_size", type=int, default=None, help="Hidden dimension of the model"
    )
    model_group.add_argument(
        "--intermediate_size", type=int, default=None, help="MLP intermediate dimension"
    )
    model_group.add_argument(
        "--num_attention_heads", type=int, default=None, help="Number of attention heads"
    )
    model_group.add_argument(
        "--num_hidden_layers", type=int, default=None, help="Number of transformer layers"
    )
    model_group.add_argument("--micro_batch_size", type=int, default=None, help="Micro batch size")
    model_group.add_argument("--sequence_length", type=int, default=None, help="Sequence length")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available.  This script requires a GPU.")
        return 1

    if not TE_AVAILABLE:
        print("Warning: Transformer Engine not available. FP8/FP4 benchmarks will be skipped.")

    # Detect model-config mode
    model_config_names = [
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "micro_batch_size",
        "sequence_length",
    ]
    model_config_vals = {name: getattr(args, name) for name in model_config_names}
    has_model_config = any(v is not None for v in model_config_vals.values())
    has_shapes = args.shapes is not None

    if has_model_config and has_shapes:
        parser.error("--shapes and model config arguments are mutually exclusive.")

    if has_model_config:
        missing = [n for n, v in model_config_vals.items() if v is None]
        if missing:
            parser.error(
                f"Model config mode requires all of: {', '.join(model_config_names)}. "
                f"Missing: {', '.join(missing)}"
            )

        config = ModelConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_attention_heads,
            num_hidden_layers=args.num_hidden_layers,
            micro_batch_size=args.micro_batch_size,
            sequence_length=args.sequence_length,
        )
        run_model_config_benchmarks(
            config=config,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            include_fp8_current=not args.no_fp8_current,
            include_fp8_delayed=not args.no_fp8_delayed,
            include_fp8=not args.no_fp8,
            include_fp8_block=not args.no_fp8_block,
            include_fp4=not args.no_fp4,
            gpu_warmup_seconds=args.gpu_warmup,
            pre_quantize=args.pre_quantize,
            timing=args.timing,
            output_path=args.output,
        )
    else:
        shapes = parse_shapes_arg(args.shapes) if args.shapes else get_default_shapes()
        prof_shape = args.profile_shape if args.profile else None

        results = run_benchmarks(
            shapes=shapes,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            include_fp8_current=not args.no_fp8_current,
            include_fp8_delayed=not args.no_fp8_delayed,
            include_fp8=not args.no_fp8,
            include_fp8_block=not args.no_fp8_block,
            include_fp4=not args.no_fp4,
            gpu_warmup_seconds=args.gpu_warmup,
            pre_quantize=args.pre_quantize,
            timing=args.timing,
            profile_shape=prof_shape,
        )

        if not args.profile:
            create_plot(shapes, results, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
