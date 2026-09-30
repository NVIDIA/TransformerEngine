# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare warmed CPU dispatch time for native CUDA and CuTeDSL requantization.

Build mxfp8_requantize_cpu.cpp into a shared library first (see --helper).
The C++ timer excludes Python overhead and synchronizes before each eager call.
Graph capture is measured separately: it enqueues nodes without executing GPU
work. The optional PyTorch measurement includes its binding and allocations.
"""

import argparse
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

from benchmark_mxfp8_requantize import CommonAPI, quantize_reference, swizzle_reference


def summary(batches):
    """Retain independent batch means and the overall median/tail latency."""
    means = [statistics.mean(batch) / 1000 for batch in batches]
    samples = sorted(value / 1000 for batch in batches for value in batch)
    return {
        "median_batch_mean_us": statistics.median(means),
        "batch_means_us": means,
        "median_call_us": statistics.median(samples),
        "p95_call_us": samples[int(0.95 * (len(samples) - 1))],
    }


def pytorch_callable(rows, hidden, groups, counts, offsets, data, compact):
    """Build a compact wire tensor; reset its mutated fields outside timing."""
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    quantizer.optimize_for_gemm = False
    wire = GroupedTensor(
        shape=(rows, hidden),
        dtype=torch.bfloat16,
        num_tensors=groups,
        quantizer=quantizer,
        data=data.view(torch.uint8).flatten(),
        scale_inv=compact.flatten(),
        first_dims=counts,
        tensor_offsets=offsets,
        with_gemm_swizzled_scales=False,
    )
    compact = wire.scale_inv
    op = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    op.optimize_for_gemm = True

    def reset():
        wire.scale_inv = compact
        wire.columnwise_data = None
        wire.columnwise_scale_inv = None
        wire._with_gemm_swizzled_scales = False

    def call():
        tex.group_requantize_inplace(
            wire,
            op,
            groups,
            counts,
            te.DType.kBFloat16,
            tensor_offsets=offsets,
            return_dequantized=False,
        )

    return reset, call, wire


def run_case(args, api, helper, rows, hidden, groups):
    """Validate equal outputs, then compare both backends in alternating order."""
    if rows % groups or (rows // groups) % 128:
        raise ValueError("Use uniform, nonempty 128-aligned groups")
    counts = torch.full((groups,), rows // groups, device="cuda", dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))) * hidden
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    data, compact = quantize_reference(source)
    dst = torch.empty(rows * hidden, device="cuda", dtype=torch.uint8)
    row_sf = torch.empty_like(compact).flatten()
    col_sf = torch.empty_like(row_sf)
    metadata = [(7, counts, 3, (groups,)), (9, offsets, 3, (groups + 1,))]
    inp = api.tensor(
        [(0, data, 7, (data.numel(),)), (4, compact, 9, (compact.numel(),))] + metadata,
        rows,
        hidden,
        groups,
    )
    out = api.tensor(
        [
            (0, data, 7, (data.numel(),)),
            (1, dst, 7, (dst.numel(),)),
            (4, row_sf, 9, (row_sf.numel(),)),
            (5, col_sf, 9, (col_sf.numel(),)),
        ]
        + metadata,
        rows,
        hidden,
        groups,
        True,
    )
    stream = torch.cuda.current_stream().cuda_stream
    kernel_call = lambda: api.lib.nvte_grouped_requantize(inp, out, api.config, stream)
    warm_times, reference = {}, None
    for enabled in (False, True):
        api.lib.nvte_set_cutedsl_backend(enabled)
        begin = time.perf_counter()
        kernel_call()
        torch.cuda.synchronize()
        warm_times["cutedsl" if enabled else "native"] = time.perf_counter() - begin
        if reference is None:
            reference = (dst.clone(), row_sf.clone(), col_sf.clone())
        else:
            for actual, expected in zip((dst, row_sf, col_sf), reference):
                assert torch.equal(actual, expected)
        for _ in range(args.warmup):
            kernel_call()
        torch.cuda.synchronize()
    assert torch.equal(row_sf, swizzle_reference(compact))
    timer_args = (inp, out, api.config, stream, args.iterations)
    wall = (ct.c_uint64 * args.iterations)()
    cpu = (ct.c_uint64 * args.iterations)()
    modes = {"eager": 0, "graph_capture": 1, "timer_only": 2, "eager_batched": 3}
    samples = {
        mode: {backend: {"wall": [], "cpu": []} for backend in ("native", "cutedsl")}
        for mode in modes
    }
    # All compilation, backend toggles, warmup, allocation and final GPU waits
    # are outside the C++ timing interval. Alternate order to reduce drift.
    for repeat in range(args.repeats):
        for enabled in (False, True) if repeat % 2 == 0 else (True, False):
            api.lib.nvte_set_cutedsl_backend(enabled)
            backend = "cutedsl" if enabled else "native"
            for mode, number in modes.items():
                error = helper.measure_requantize_cpu(*timer_args, number, wall, cpu)
                if error:
                    raise RuntimeError(f"CUDA error {error}")
                samples[mode][backend]["wall"].append(list(wall))
                samples[mode][backend]["cpu"].append(list(cpu))
    result = {
        "rows": rows,
        "hidden": hidden,
        "groups": groups,
        "first_call_plus_sync_seconds": warm_times,
        "c_api": {
            mode: {
                backend: {clock: summary(batches) for clock, batches in clocks.items()}
                for backend, clocks in backends.items()
            }
            for mode, backends in samples.items()
        },
    }
    if args.pytorch:
        api.lib.nvte_set_cutedsl_backend(False)
        reset, call, wire = pytorch_callable(rows, hidden, groups, counts, offsets, data, compact)
        for enabled in (False, True):
            api.lib.nvte_set_cutedsl_backend(enabled)
            for _ in range(args.warmup):
                reset()
                call()
                torch.cuda.synchronize()
            for actual, expected in zip(
                (wire.columnwise_data, wire.scale_inv, wire.columnwise_scale_inv), reference
            ):
                assert torch.equal(actual.flatten(), expected.flatten())
        samples = {backend: {"wall": [], "cpu": []} for backend in ("native", "cutedsl")}
        for repeat in range(args.repeats):
            for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                api.lib.nvte_set_cutedsl_backend(enabled)
                backend = "cutedsl" if enabled else "native"
                wall_batch, cpu_batch = [], []
                for _ in range(args.iterations):
                    torch.cuda.synchronize()
                    reset()
                    cpu_begin = time.thread_time_ns()
                    begin = time.perf_counter_ns()
                    call()
                    end = time.perf_counter_ns()
                    cpu_end = time.thread_time_ns()
                    wall_batch.append(end - begin)
                    cpu_batch.append(cpu_end - cpu_begin)
                torch.cuda.synchronize()
                samples[backend]["wall"].append(wall_batch)
                samples[backend]["cpu"].append(cpu_batch)
        result["pytorch_binding"] = {
            backend: {clock: summary(batches) for clock, batches in clocks.items()}
            for backend, clocks in samples.items()
        }
        assert wire.columnwise_data is not None
    api.lib.nvte_set_cutedsl_backend(False)
    return result


def main():
    """Time the actual common C API inside a native C++ loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, default=Path("/tmp/mxfp8_requantize_cpu.so"))
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["1024,128,8", "4096,4096,8", "32768,4096,64", "131072,7168,256"],
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--pytorch", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/mxfp8_requantize_cpu.json"))
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1 or args.warmup < 1:
        parser.error("iterations, repeats and warmup must be positive")
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    torch.manual_seed(1234)
    torch.cuda.set_stream(torch.cuda.Stream())
    api = CommonAPI()
    api.lib.nvte_set_cutedsl_backend.argtypes = [ct.c_bool]
    api.lib.nvte_set_cutedsl_backend.restype = None
    helper = ct.CDLL(str(args.helper.resolve()))
    helper.measure_requantize_cpu.argtypes = (
        [ct.c_void_p] * 4 + [ct.c_int] * 2 + [ct.POINTER(ct.c_uint64)] * 2
    )
    helper.measure_requantize_cpu.restype = ct.c_int
    gc.disable()
    report = {
        "environment": {
            "device": torch.cuda.get_device_name(),
            "cpu_affinity": [cpu],
            "lscpu": subprocess.check_output(["lscpu"], text=True),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "te_library": str(api.path),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "envvars": {
                key: value
                for key, value in os.environ.items()
                if key.startswith("NVTE_") or key == "CUDA_VISIBLE_DEVICES"
            },
        },
        "method": vars(args) | {"helper": str(args.helper), "output": str(args.output)},
        "cases": [],
    }
    try:
        for shape in args.shapes:
            result = run_case(args, api, helper, *map(int, shape.split(",")))
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        api.lib.nvte_set_cutedsl_backend(False)
        api.close()
        gc.enable()


if __name__ == "__main__":
    main()
