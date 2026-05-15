# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 tensor-scaling quantization."""

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from transformer_engine.pytorch import Float8CurrentScalingQuantizer, Float8Quantizer
import transformer_engine_torch as tex


MODES = ("rowwise", "columnwise", "both")


def _usage_for_mode(mode: str) -> Dict[str, bool]:
    return {
        "rowwise": mode in ("rowwise", "both"),
        "columnwise": mode in ("columnwise", "both"),
    }


def _make_current_quantizer(mode: str) -> Float8CurrentScalingQuantizer:
    usage = _usage_for_mode(mode)
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device="cuda",
        force_pow_2_scales=False,
        amax_epsilon=0.0,
    )
    quantizer.set_usage(**usage)
    return quantizer


def _make_delayed_quantizer(mode: str, scale: torch.Tensor, amax: torch.Tensor) -> Float8Quantizer:
    usage = _usage_for_mode(mode)
    quantizer = Float8Quantizer(
        scale=scale,
        amax=amax,
        fp8_dtype=tex.DType.kFloat8E4M3,
        **usage,
    )
    return quantizer


def _bytes_per_call(actual_elements: int, input_element_size: int, mode: str) -> int:
    output_bytes = 2 if mode == "both" else 1
    return actual_elements * (input_element_size + output_bytes)


def _make_inputs(
    *,
    num_buffers: int,
    num_groups: int,
    rows_per_group: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> tuple[List[torch.Tensor], torch.Tensor, Dict[str, int]]:
    actual_rows = num_groups * rows_per_group
    allocated_rows = actual_rows * 2
    first_dims = torch.full((num_groups,), rows_per_group, dtype=torch.int64, device="cuda")
    inputs = []
    for _ in range(num_buffers):
        tensor = torch.randn(allocated_rows, hidden_size, dtype=dtype, device="cuda")
        tensor[actual_rows:].fill_(10000.0)
        inputs.append(tensor)

    element_counts = {
        "actual_elements": actual_rows * hidden_size,
        "allocated_elements": allocated_rows * hidden_size,
        "unused_tail_elements": (allocated_rows - actual_rows) * hidden_size,
        "actual_rows": actual_rows,
        "allocated_rows": allocated_rows,
    }
    return inputs, first_dims, element_counts


def _prepare_mode(
    mode: str,
    inputs: List[torch.Tensor],
    first_dims: torch.Tensor,
    num_groups: int,
) -> tuple[List[Float8Quantizer], list]:
    delayed_quantizers = []
    outputs = []
    current_quantizer = _make_current_quantizer(mode)
    for tensor in inputs:
        prepared = tex.group_quantize(tensor, current_quantizer, num_groups, first_dims)
        delayed_quantizer = _make_delayed_quantizer(mode, prepared.scale, prepared.amax)
        output = tex.group_quantize(tensor, delayed_quantizer, num_groups, first_dims)
        delayed_quantizers.append(delayed_quantizer)
        outputs.append(output)
    return delayed_quantizers, outputs


def _prepare_benchmark_state(args: argparse.Namespace, mode: str) -> Dict[str, Any]:
    inputs, first_dims, element_counts = _make_inputs(
        num_buffers=args.num_buffers,
        num_groups=args.num_groups,
        rows_per_group=args.rows_per_group,
        hidden_size=args.hidden_size,
        dtype=torch.bfloat16,
    )
    quantizers, outputs = _prepare_mode(mode, inputs, first_dims, args.num_groups)
    return {
        "mode": mode,
        "inputs": inputs,
        "first_dims": first_dims,
        "element_counts": element_counts,
        "quantizers": quantizers,
        "outputs": outputs,
    }


def _warmup_benchmark_state(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    for iteration in range(args.warmup_iters):
        idx = iteration % len(state["inputs"])
        tex.group_quantize(
            state["inputs"][idx],
            state["quantizers"][idx],
            args.num_groups,
            state["first_dims"],
            output=state["outputs"][idx],
        )


def _run_timed_loop(
    *,
    mode: str,
    inputs: List[torch.Tensor],
    quantizers: List[Float8Quantizer],
    outputs: list,
    first_dims: torch.Tensor,
    num_groups: int,
    iterations: int,
    profile: bool,
) -> float:
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(f"grouped_fp8_quantize:{mode}")
    start.record()
    for iteration in range(iterations):
        idx = iteration % len(inputs)
        tex.group_quantize(
            inputs[idx],
            quantizers[idx],
            num_groups,
            first_dims,
            output=outputs[idx],
        )
    end.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    return start.elapsed_time(end) / 1000.0


def _make_result(
    args: argparse.Namespace,
    mode: str,
    element_counts: Dict[str, int],
    elapsed_sec: float,
) -> Dict[str, object]:
    input_element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    relevant_bytes = _bytes_per_call(element_counts["actual_elements"], input_element_size, mode)
    excluded_tail_bytes = _bytes_per_call(
        element_counts["unused_tail_elements"],
        input_element_size,
        mode,
    )
    bandwidth_tbps = relevant_bytes * args.iters / elapsed_sec / 1.0e12
    return {
        "mode": mode,
        "num_groups": args.num_groups,
        "first_dims": [args.rows_per_group] * args.num_groups,
        "hidden_size": args.hidden_size,
        "input_dtype": "bfloat16",
        "output_dtype": "float8_e4m3",
        "num_distinct_input_buffers": args.num_buffers,
        "warmup_iterations": args.warmup_iters,
        "timed_iterations": args.iters,
        "elapsed_sec": elapsed_sec,
        "relevant_bytes_per_call": relevant_bytes,
        "allocated_elements": element_counts["allocated_elements"],
        "actual_elements": element_counts["actual_elements"],
        "unused_tail_elements": element_counts["unused_tail_elements"],
        "tail_fraction_of_allocated": (
            element_counts["unused_tail_elements"] / element_counts["allocated_elements"]
        ),
        "allocated_rows": element_counts["allocated_rows"],
        "actual_rows": element_counts["actual_rows"],
        "allocated_bytes_excluded_from_bandwidth": excluded_tail_bytes,
        "bandwidth_TBps_actual_bytes": bandwidth_tbps,
        "gb200_peak_bandwidth_TBps_expectation": 8.0,
        "fraction_of_8TBps_peak": bandwidth_tbps / 8.0,
        "passes_6TBps_target": bandwidth_tbps >= 6.0,
        "timed_call": "tex.group_quantize(input, Float8Quantizer(precomputed_scale), ..., output=preallocated)",
    }


def benchmark_mode(args: argparse.Namespace, mode: str) -> Dict[str, object]:
    state = _prepare_benchmark_state(args, mode)
    _warmup_benchmark_state(args, state)
    elapsed_sec = _run_timed_loop(
        mode=mode,
        inputs=state["inputs"],
        quantizers=state["quantizers"],
        outputs=state["outputs"],
        first_dims=state["first_dims"],
        num_groups=args.num_groups,
        iterations=args.iters,
        profile=args.profile,
    )
    return _make_result(args, mode, state["element_counts"], elapsed_sec)


def benchmark_modes_with_single_profile(
    args: argparse.Namespace,
    modes: tuple[str, ...],
) -> List[Dict[str, object]]:
    states = [_prepare_benchmark_state(args, mode) for mode in modes]
    for state in states:
        _warmup_benchmark_state(args, state)

    results = []
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for state in states:
            mode = state["mode"]
            elapsed_sec = _run_timed_loop(
                mode=mode,
                inputs=state["inputs"],
                quantizers=state["quantizers"],
                outputs=state["outputs"],
                first_dims=state["first_dims"],
                num_groups=args.num_groups,
                iterations=args.iters,
                profile=False,
            )
            results.append(_make_result(args, mode, state["element_counts"], elapsed_sec))
    finally:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all",) + MODES, default="all")
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--rows-per-group", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--num-buffers", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--output",
        default=os.environ.get(
            "ORCHESTRA_BENCHMARK_RAW_REPORT",
            "grouped_fp8_quantize_report.json",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows_per_group % 128 != 0:
        raise ValueError("--rows-per-group must be divisible by 128 for the grouped FP8 kernel")
    modes = MODES if args.mode == "all" else (args.mode,)
    if args.profile and len(modes) > 1:
        results = benchmark_modes_with_single_profile(args, modes)
    else:
        results = [benchmark_mode(args, mode) for mode in modes]
    report = {
        "benchmark": "grouped_fp8_tensor_scaling_quantize",
        "profile_enabled": args.profile,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
