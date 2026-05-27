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
    bw_alloc_TBps: float


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


def run_case(shape_case: str, mode: str, *, actual_rows: int, allocated_rows: int,
             hidden: int, num_groups: int, num_buffers: int, warmup: int,
             iters: int, mode_loop: str = "eager", verbose: bool = True) -> CaseResult:
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
    total_bytes_alloc = _bytes_per_call(allocated_rows, hidden, mode) * actual_iters
    elapsed_s = elapsed_ms / 1000.0
    bw_actual = total_bytes_actual / elapsed_s / 1.0e12
    bw_alloc = total_bytes_alloc / elapsed_s / 1.0e12
    res = CaseResult(
        shape_case=shape_case, mode=mode,
        actual_rows=actual_rows, allocated_rows=allocated_rows,
        hidden=hidden, num_groups=num_groups, iters=actual_iters,
        elapsed_ms_total=elapsed_ms,
        per_iter_us=elapsed_ms * 1000.0 / actual_iters,
        relevant_bytes=relevant, bw_actual_TBps=bw_actual, bw_alloc_TBps=bw_alloc,
    )
    if verbose:
        print(f"  {shape_case:30s} groups={num_groups:3d} mode={mode:10s} "
              f"loop={mode_loop:5s} "
              f"per_iter={res.per_iter_us:7.2f}us "
              f"BW_actual={res.bw_actual_TBps:5.2f} TB/s "
              f"BW_alloc={res.bw_alloc_TBps:5.2f} TB/s")
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
                    results.append(run_case(
                        shape_case, mode,
                        actual_rows=args.actual_rows,
                        allocated_rows=args.allocated_rows,
                        hidden=args.hidden,
                        num_groups=num_groups,
                        num_buffers=args.num_buffers,
                        warmup=args.warmup, iters=args.iters,
                        mode_loop=loop,
                    ))
            print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
