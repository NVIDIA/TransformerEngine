#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""End-to-end benchmark for grouped FP8 *current scaling* quantization.

Times the full Float8CurrentScalingQuantizer.group_quantize path
(amax kernel + scale-from-amax kernel + cast/transpose kernels).

Default shape mirrors the production use case:
    sum(first_dims) = 98304, hidden = 2880

By default the benchmark sweeps both num_groups=16 and num_groups=64 so that
we can see how kernel performance scales with group count for the same total
work. Override with --num-groups 16 (or 64, or 16 64 ...) to restrict.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional

# IMPORTANT: import transformer_engine before torch to avoid cublasLt symbol-resolution
# issues caused by torch's bundled CUDA libs.
import transformer_engine.pytorch  # noqa: F401  - registers extension
from transformer_engine.pytorch import Float8CurrentScalingQuantizer
import transformer_engine_torch as tex
import torch


MODES = ("rowwise", "columnwise", "both")
SHAPE_CASES = ("same-shape", "varying-first", "varying-first-overalloc")


@dataclass
class CaseResult:
    shape_case: str
    mode: str
    actual_rows: int
    allocated_rows: int
    hidden: int
    num_groups: int
    iters: int
    elapsed_ms_total: float
    per_iter_us: float
    relevant_bytes: int
    bw_actual_TBps: float


def _make_quantizer(mode: str) -> Float8CurrentScalingQuantizer:
    q = Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device="cuda",
        force_pow_2_scales=False,
        amax_epsilon=0.0,
    )
    q.set_usage(rowwise=mode in ("rowwise", "both"),
                columnwise=mode in ("columnwise", "both"))
    return q


def _equal_first_dims(actual_rows: int, num_groups: int) -> List[int]:
    assert actual_rows % num_groups == 0, "actual_rows must divide num_groups"
    return [actual_rows // num_groups] * num_groups


def _make_inputs(shape_case: str, actual_rows: int, allocated_rows: int,
                 hidden: int, num_groups: int, num_buffers: int):
    """Returns (inputs, first_dims_or_None, info)."""
    dtype = torch.bfloat16
    info = {
        "actual_rows": actual_rows,
        "allocated_rows": allocated_rows,
        "hidden": hidden,
        "num_groups": num_groups,
    }
    if shape_case == "same-shape":
        inputs = [torch.randn(actual_rows, hidden, dtype=dtype, device="cuda")
                  for _ in range(num_buffers)]
        return inputs, None, info
    if shape_case == "varying-first":
        first_dims_list = _equal_first_dims(actual_rows, num_groups)
        first_dims = torch.tensor(first_dims_list, dtype=torch.int64, device="cuda")
        inputs = [torch.randn(actual_rows, hidden, dtype=dtype, device="cuda")
                  for _ in range(num_buffers)]
        info["first_dims"] = first_dims_list
        return inputs, first_dims, info
    if shape_case == "varying-first-overalloc":
        first_dims_list = _equal_first_dims(actual_rows, num_groups)
        first_dims = torch.tensor(first_dims_list, dtype=torch.int64, device="cuda")
        inputs = []
        for _ in range(num_buffers):
            t = torch.randn(allocated_rows, hidden, dtype=dtype, device="cuda")
            # Tail rows must NOT influence amax (they shouldn't be read).
            # Fill with poison values: a kernel that reads them will produce
            # amax that disagrees with current_scaling reference.
            t[actual_rows:].fill_(1e30)
            inputs.append(t)
        info["first_dims"] = first_dims_list
        return inputs, first_dims, info
    raise ValueError(shape_case)


def _verify_no_tail_read(quantizer, inp: torch.Tensor, first_dims, num_groups: int,
                         actual_rows: int):
    """Sanity check: amax must not be polluted by the poisoned tail rows."""
    out = tex.group_quantize(inp, quantizer, num_groups, first_dims)
    amax_max = float(out.amax.max().item())
    # If tail rows (1e30) were read, amax >= 1e30. Real bf16 N(0,1) data: amax ~ 5.
    if amax_max > 1e10:
        raise RuntimeError(
            f"group_quantize is reading the over-allocated tail rows: amax_max={amax_max}. "
            f"This means kernels are scanning unused memory (a perf and correctness bug)."
        )
    return out


def _timed_eager(quantizer, inputs, first_dims, num_groups: int, outputs, iters: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for it in range(iters):
        i = it % len(inputs)
        tex.group_quantize(inputs[i], quantizer, num_groups, first_dims,
                           output=outputs[i])
    end.record()
    end.synchronize()
    return start.elapsed_time(end)  # ms


def _timed_cuda_graph(quantizer, inputs, first_dims, num_groups: int, outputs,
                      iters: int, calls_per_replay: int = 32) -> float:
    """Capture `calls_per_replay` group_quantize calls into one CUDA graph and
    replay it `iters / calls_per_replay` times. Effectively eliminates Python /
    cudaLaunchKernel overhead."""
    static_input = inputs[0]
    static_output = outputs[0]

    # Warmup must happen on a side stream before capture (per torch docs)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            tex.group_quantize(static_input, quantizer, num_groups, first_dims,
                               output=static_output)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls_per_replay):
            tex.group_quantize(static_input, quantizer, num_groups, first_dims,
                               output=static_output)

    replays = max(1, iters // calls_per_replay)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    actual_iters = replays * calls_per_replay
    return elapsed, actual_iters


def _bytes_per_call(actual_rows: int, hidden: int, mode: str) -> int:
    in_bytes = actual_rows * hidden * 2  # bf16 read
    out_copies = (1 if mode in ("rowwise", "both") else 0) + \
                 (1 if mode in ("columnwise", "both") else 0)
    out_bytes = actual_rows * hidden * out_copies  # fp8 write(s)
    return in_bytes + out_bytes


def _kernel_bytes_per_call(bucket: str, actual_rows: int, hidden: int, mode: str,
                           num_tensors: int) -> int:
    """Bytes that one launch of the kernel(s) in ``bucket`` actually moves.

    Counts both reads and writes. Tiny kernels (amax_zero, compute_scale) move
    only ``num_tensors``-sized metadata buffers so their reported BW will look
    low; that's expected -- they're launch-overhead dominated, not bandwidth
    bound.
    """
    input_bytes = actual_rows * hidden * 2  # bf16 input
    if bucket == "amax_zero":
        # Writes ``num_tensors`` floats to zero the amax buffer.
        return num_tensors * 4
    if bucket == "amax":
        # Reads the full input over the active region (the kernel uses tensor_offsets
        # to skip rows past sum(first_dims)) and atomicMaxes into the per-group amax
        # slots. Output traffic is negligible vs the input scan.
        return input_bytes + num_tensors * 4
    if bucket == "compute_scale":
        # Reads ``num_tensors`` amax floats, writes ``num_tensors`` of (scale,
        # scale_inv) -- ~3 floats per group.
        return num_tensors * 4 * 3
    if bucket == "cast":
        # Reads bf16 input, writes fp8 rowwise and/or columnwise (1 byte per
        # element per direction). Mirrors the eager BW_actual byte count.
        return _bytes_per_call(actual_rows, hidden, mode)
    return 0


# Substring patterns used to bucket CUDA kernels in --profile mode. Order matters:
# the first matching bucket wins. Anything that doesn't match falls into "cast".
# Patterns are matched against the kernel name AFTER lowercasing -- both the
# kernel name and the pattern strings are lowercased before substring match,
# so patterns here must already be lowercase. Camelcased identifiers like
# ``ComputeScaleAndScaleInvFunctor`` flatten to ``computescaleandscaleinvfunctor``.
_KERNEL_BUCKETS = (
    # Split amax into the trivial zero-init kernel and the real reduction kernel,
    # otherwise the bucket average misleadingly suggests the zero kernel is slow.
    ("amax_zero",     ("grouped_amax_zero",)),
    ("amax",          ("grouped_amax",)),
    # The compute-scale kernel is launched via the multi_tensor_apply framework
    # as ``multi_tensor_apply_kernel<...ComputeScaleAndScaleInvFunctor>``; match
    # on the functor name after lowercasing.
    ("compute_scale", ("computescale",)),
)


def _bucket_for_kernel(name: str) -> str:
    lname = name.lower()
    for bucket, patterns in _KERNEL_BUCKETS:
        if any(p in lname for p in patterns):
            return bucket
    return "cast"


def _profile_breakdown(quantizer, inputs, first_dims, num_groups: int, outputs,
                       iters: int) -> dict:
    """Run ``iters`` group_quantize calls under torch.profiler and aggregate CUDA
    kernel time into {amax, compute_scale, cast} buckets.

    Returns dict: bucket -> {"us_total": float, "calls": int}, plus a "_total_us"
    key holding the summed GPU time across all kernels (not wall-clock).
    """
    from torch.profiler import profile, ProfilerActivity

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for it in range(iters):
            i = it % len(inputs)
            tex.group_quantize(inputs[i], quantizer, num_groups, first_dims,
                               output=outputs[i])
        torch.cuda.synchronize()

    agg = {b: {"us_total": 0.0, "calls": 0} for b, _ in _KERNEL_BUCKETS}
    agg["cast"] = {"us_total": 0.0, "calls": 0}
    total_us = 0.0
    for evt in prof.key_averages():
        # CPU events have cuda_time_total == 0; kernel events have device time.
        if evt.device_type.name != "CUDA":
            continue
        us = float(evt.self_device_time_total)
        if us <= 0:
            continue
        bucket = _bucket_for_kernel(evt.key)
        agg[bucket]["us_total"] += us
        agg[bucket]["calls"] += evt.count
        total_us += us
    agg["_total_us"] = total_us
    return agg


def run_case(shape_case: str, mode: str, *, actual_rows: int, allocated_rows: int,
             hidden: int, num_groups: int, num_buffers: int, warmup: int,
             iters: int, mode_loop: str = "eager", verbose: bool = True,
             profile_breakdown: bool = False) -> CaseResult:
    inputs, first_dims, _ = _make_inputs(shape_case, actual_rows, allocated_rows,
                                          hidden, num_groups, num_buffers)
    quantizer = _make_quantizer(mode)

    # Run once to verify correctness and to allocate output tensors.
    outputs = []
    for inp in inputs:
        out = tex.group_quantize(inp, quantizer, num_groups, first_dims)
        outputs.append(out)
    if shape_case == "varying-first-overalloc":
        _verify_no_tail_read(quantizer, inputs[0], first_dims, num_groups, actual_rows)

    # Warmup so GPU clocks are at boost.
    for it in range(warmup):
        i = it % len(inputs)
        tex.group_quantize(inputs[i], quantizer, num_groups, first_dims,
                           output=outputs[i])
    torch.cuda.synchronize()

    if mode_loop == "graph":
        elapsed_ms, actual_iters = _timed_cuda_graph(
            quantizer, inputs, first_dims, num_groups, outputs, iters)
    else:
        elapsed_ms = _timed_eager(
            quantizer, inputs, first_dims, num_groups, outputs, iters)
        actual_iters = iters

    relevant = _bytes_per_call(actual_rows, hidden, mode)
    total_bytes_actual = relevant * actual_iters
    elapsed_s = elapsed_ms / 1000.0
    bw_actual = total_bytes_actual / elapsed_s / 1.0e12
    res = CaseResult(
        shape_case=shape_case, mode=mode,
        actual_rows=actual_rows, allocated_rows=allocated_rows,
        hidden=hidden, num_groups=num_groups, iters=actual_iters,
        elapsed_ms_total=elapsed_ms,
        per_iter_us=elapsed_ms * 1000.0 / actual_iters,
        relevant_bytes=relevant, bw_actual_TBps=bw_actual,
    )
    if verbose:
        print(f"  {shape_case:30s} groups={num_groups:3d} mode={mode:10s} "
              f"loop={mode_loop:5s} "
              f"per_iter={res.per_iter_us:7.2f}us "
              f"BW_actual={res.bw_actual_TBps:5.2f} TB/s")

    if profile_breakdown:
        profile_iters = min(iters, 50)
        agg = _profile_breakdown(quantizer, inputs, first_dims, num_groups, outputs,
                                 iters=profile_iters)
        total = agg["_total_us"] if agg["_total_us"] > 0 else 1.0
        per_iter_total = total / profile_iters
        print(f"      kernel breakdown over {profile_iters} iters "
              f"(per-iter sum={per_iter_total:6.2f}us):")
        print(f"        {'bucket':14s} {'per_iter_us':>12s} {'(%)':>7s} "
              f"{'launches/iter':>14s} {'per_launch_us':>15s} "
              f"{'bytes/launch':>14s} {'BW_TBps':>9s}")
        for bucket in ("amax_zero", "amax", "compute_scale", "cast"):
            entry = agg[bucket]
            pct = 100.0 * entry["us_total"] / total
            per_iter_us = entry["us_total"] / profile_iters
            launches_per_iter = entry["calls"] / profile_iters
            per_launch_us = (
                entry["us_total"] / entry["calls"] if entry["calls"] else 0.0
            )
            bytes_per_launch = _kernel_bytes_per_call(
                bucket, actual_rows, hidden, mode, num_groups
            )
            # bytes_per_launch / (per_launch_us * 1e-6) / 1e12 = TB/s
            bw_tbps = (
                bytes_per_launch / (per_launch_us * 1.0e-6) / 1.0e12
                if per_launch_us > 0
                else 0.0
            )
            print(f"        {bucket:14s} {per_iter_us:12.2f} {pct:7.1f} "
                  f"{launches_per_iter:14.2f} {per_launch_us:15.2f} "
                  f"{bytes_per_launch:14d} {bw_tbps:9.2f}")
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-rows", type=int, default=98304)
    parser.add_argument("--allocated-rows", type=int, default=None,
                        help="For varying-first-overalloc; default 2 * actual-rows")
    parser.add_argument("--hidden", type=int, default=2880)
    parser.add_argument("--num-groups", type=int, nargs="+", default=[16, 64],
                        help="Sweep one or more group counts (default: 16 64).")
    parser.add_argument("--num-buffers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--shape-cases", nargs="+", default=list(SHAPE_CASES))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--loop", choices=("eager", "graph", "both"), default="both",
                        help="eager Python loop, captured CUDA graph, or both")
    parser.add_argument("--profile", action="store_true",
                        help="After each timed case, also run a torch.profiler pass "
                             "and break down CUDA kernel time into "
                             "{amax, compute_scale, cast} buckets. Adds overhead.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.allocated_rows is None:
        args.allocated_rows = 2 * args.actual_rows

    # Validate that actual_rows is divisible by every requested group count so
    # the equal-first-dims helper produces well-defined splits.
    for ng in args.num_groups:
        if args.actual_rows % ng != 0:
            raise SystemExit(
                f"--actual-rows={args.actual_rows} not divisible by num_groups={ng}"
            )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: actual_rows={args.actual_rows}, allocated_rows={args.allocated_rows}, "
          f"hidden={args.hidden}, num_groups_sweep={args.num_groups}, "
          f"iters={args.iters}, warmup={args.warmup}")
    print()

    loop_modes = ("eager", "graph") if args.loop == "both" else (args.loop,)

    results: List[CaseResult] = []
    for num_groups in args.num_groups:
        print(f"#### num_groups={num_groups} ####")
        for shape_case in args.shape_cases:
            if shape_case not in SHAPE_CASES:
                raise SystemExit(f"unknown shape_case={shape_case}")
            print(f"== {shape_case} ==")
            for mode in args.modes:
                if mode not in MODES:
                    raise SystemExit(f"unknown mode={mode}")
                for loop in loop_modes:
                    # Only profile-breakdown the eager loop -- profiling a captured
                    # CUDA graph aggregates everything into the graph launch event,
                    # which hides per-kernel time. Eager mode gives true per-kernel
                    # device time.
                    do_profile = args.profile and loop == "eager"
                    results.append(run_case(
                        shape_case, mode,
                        actual_rows=args.actual_rows,
                        allocated_rows=args.allocated_rows,
                        hidden=args.hidden,
                        num_groups=num_groups,
                        num_buffers=args.num_buffers,
                        warmup=args.warmup, iters=args.iters,
                        mode_loop=loop,
                        profile_breakdown=do_profile,
                    ))
            print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
