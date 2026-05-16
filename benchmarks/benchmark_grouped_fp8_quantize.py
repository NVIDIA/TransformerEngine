# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 tensor-scaling quantization."""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformer_engine.pytorch import Float8CurrentScalingQuantizer, Float8Quantizer
from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported
import transformer_engine_torch as tex


MODES = ("rowwise", "columnwise", "both")
SHAPE_CASES = ("same-shape", "varying-first", "varying-last")


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
    return Float8Quantizer(
        scale=scale,
        amax=amax,
        fp8_dtype=tex.DType.kFloat8E4M3,
        **usage,
    )


def _varying_dims(base: int, num_groups: int) -> List[int]:
    step = 128
    center = (num_groups - 1) / 2.0
    dims = []
    for idx in range(num_groups):
        value = base + int(idx - center) * step
        dims.append(max(step, (value // step) * step))
    return dims


def _make_inputs(
    *,
    shape_case: str,
    num_buffers: int,
    num_groups: int,
    rows_per_group: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    if shape_case == "same-shape":
        rows = num_groups * rows_per_group
        inputs = [
            torch.randn(rows, hidden_size, dtype=dtype, device="cuda")
            for _ in range(num_buffers)
        ]
        element_counts = {
            "actual_elements": rows * hidden_size,
            "allocated_elements": rows * hidden_size,
            "unused_tail_elements": 0,
            "actual_rows": rows,
            "allocated_rows": rows,
            "first_dims": [rows_per_group] * num_groups,
            "last_dims": [hidden_size] * num_groups,
        }
        return inputs, None, None, element_counts

    if shape_case == "varying-first":
        first_dims_list = _varying_dims(rows_per_group, num_groups)
        actual_rows = sum(first_dims_list)
        allocated_rows = actual_rows * 2
        first_dims = torch.tensor(first_dims_list, dtype=torch.int64, device="cuda")
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
            "first_dims": first_dims_list,
            "last_dims": [hidden_size] * num_groups,
        }
        return inputs, first_dims, None, element_counts

    if shape_case == "varying-last":
        last_dims_list = _varying_dims(hidden_size, num_groups)
        total_cols = sum(last_dims_list)
        actual_elements = rows_per_group * total_cols
        last_dims = torch.tensor(last_dims_list, dtype=torch.int64, device="cuda")
        inputs = []
        for _ in range(num_buffers):
            flat = torch.empty(actual_elements, dtype=dtype, device="cuda")
            offset = 0
            for cols in last_dims_list:
                group = torch.randn(rows_per_group, cols, dtype=dtype, device="cuda")
                flat[offset : offset + group.numel()].copy_(group.reshape(-1))
                offset += group.numel()
            inputs.append(flat.view(rows_per_group, total_cols))
        element_counts = {
            "actual_elements": actual_elements,
            "allocated_elements": actual_elements,
            "unused_tail_elements": 0,
            "actual_rows": rows_per_group,
            "allocated_rows": rows_per_group,
            "first_dims": [rows_per_group] * num_groups,
            "last_dims": last_dims_list,
        }
        return inputs, None, last_dims, element_counts

    raise ValueError(f"Unknown shape case: {shape_case}")


def _prepare_mode(
    mode: str,
    inputs: List[torch.Tensor],
    first_dims: Optional[torch.Tensor],
    last_dims: Optional[torch.Tensor],
    num_groups: int,
) -> Tuple[List[Float8Quantizer], list]:
    delayed_quantizers = []
    outputs = []
    current_quantizer = _make_current_quantizer(mode)
    for tensor in inputs:
        prepared = tex.group_quantize(
            tensor, current_quantizer, num_groups, first_dims, last_dims=last_dims
        )
        delayed_quantizer = _make_delayed_quantizer(mode, prepared.scale, prepared.amax)
        output = tex.group_quantize(
            tensor, delayed_quantizer, num_groups, first_dims, last_dims=last_dims
        )
        delayed_quantizers.append(delayed_quantizer)
        outputs.append(output)
    return delayed_quantizers, outputs


def _prepare_benchmark_state(
    args: argparse.Namespace, shape_case: str, mode: str
) -> Dict[str, Any]:
    inputs, first_dims, last_dims, element_counts = _make_inputs(
        shape_case=shape_case,
        num_buffers=args.num_buffers,
        num_groups=args.num_groups,
        rows_per_group=args.rows_per_group,
        hidden_size=args.hidden_size,
        dtype=torch.bfloat16,
    )
    quantizers, outputs = _prepare_mode(mode, inputs, first_dims, last_dims, args.num_groups)
    first_output = outputs[0]
    return {
        "shape_case": shape_case,
        "mode": mode,
        "inputs": inputs,
        "first_dims": first_dims,
        "last_dims": last_dims,
        "element_counts": element_counts,
        "quantizers": quantizers,
        "outputs": outputs,
        "materialized_rowwise": first_output.rowwise_data is not None,
        "materialized_columnwise": first_output.columnwise_data is not None,
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
            last_dims=state["last_dims"],
        )


def _bytes_per_call(
    actual_elements: int,
    input_element_size: int,
    materialized_rowwise: bool,
    materialized_columnwise: bool,
) -> int:
    output_copies = int(materialized_rowwise) + int(materialized_columnwise)
    return actual_elements * (input_element_size + output_copies)


def _run_timed_loop(
    *,
    shape_case: str,
    mode: str,
    inputs: List[torch.Tensor],
    quantizers: List[Float8Quantizer],
    outputs: list,
    first_dims: Optional[torch.Tensor],
    last_dims: Optional[torch.Tensor],
    num_groups: int,
    iterations: int,
    profile: bool,
) -> float:
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(f"grouped_fp8_quantize:{shape_case}:{mode}")
    start.record()
    for iteration in range(iterations):
        idx = iteration % len(inputs)
        tex.group_quantize(
            inputs[idx],
            quantizers[idx],
            num_groups,
            first_dims,
            output=outputs[idx],
            last_dims=last_dims,
        )
    end.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    return start.elapsed_time(end) / 1000.0


def _target_tbps(shape_case: str) -> float:
    return 4.0 if shape_case == "varying-last" else 6.0


def _make_result(
    args: argparse.Namespace,
    state: Dict[str, Any],
    elapsed_sec: float,
    iterations: int,
) -> Dict[str, object]:
    element_counts = state["element_counts"]
    input_element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    relevant_bytes = _bytes_per_call(
        element_counts["actual_elements"],
        input_element_size,
        state["materialized_rowwise"],
        state["materialized_columnwise"],
    )
    excluded_tail_bytes = _bytes_per_call(
        element_counts["unused_tail_elements"],
        input_element_size,
        state["materialized_rowwise"],
        state["materialized_columnwise"],
    )
    bandwidth_tbps = relevant_bytes * iterations / elapsed_sec / 1.0e12
    threshold = _target_tbps(state["shape_case"])
    return {
        "shape_case": state["shape_case"],
        "mode": state["mode"],
        "num_groups": args.num_groups,
        "first_dims": element_counts["first_dims"],
        "last_dims": element_counts["last_dims"],
        "hidden_size_argument": args.hidden_size,
        "input_dtype": "bfloat16",
        "output_dtype": "float8_e4m3",
        "non_tn_fp8_gemm_supported": is_non_tn_fp8_gemm_supported(),
        "materialized_rowwise": state["materialized_rowwise"],
        "materialized_columnwise": state["materialized_columnwise"],
        "num_distinct_input_buffers": args.num_buffers,
        "warmup_iterations": args.warmup_iters,
        "timed_iterations": iterations,
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
        "target_TBps_actual_bytes": threshold,
        "passes_target": bandwidth_tbps >= threshold,
        "gb200_peak_bandwidth_TBps_expectation": 8.0,
        "fraction_of_8TBps_peak": bandwidth_tbps / 8.0,
        "timed_call": (
            "tex.group_quantize(input, Float8Quantizer(precomputed_scale), ..., "
            "output=preallocated)"
        ),
    }


def benchmark_case_mode(args: argparse.Namespace, shape_case: str, mode: str) -> Dict[str, object]:
    state = _prepare_benchmark_state(args, shape_case, mode)
    _warmup_benchmark_state(args, state)
    iterations = args.profile_iters if args.profile else args.iters
    elapsed_sec = _run_timed_loop(
        shape_case=shape_case,
        mode=mode,
        inputs=state["inputs"],
        quantizers=state["quantizers"],
        outputs=state["outputs"],
        first_dims=state["first_dims"],
        last_dims=state["last_dims"],
        num_groups=args.num_groups,
        iterations=iterations,
        profile=args.profile,
    )
    return _make_result(args, state, elapsed_sec, iterations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all",) + MODES, default="all")
    parser.add_argument("--shape-case", choices=("all",) + SHAPE_CASES, default="all")
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--rows-per-group", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--num-buffers", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile-iters", type=int, default=5)
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
    if args.hidden_size % 128 != 0:
        raise ValueError("--hidden-size must be divisible by 128 for the grouped FP8 kernel")
    modes = MODES if args.mode == "all" else (args.mode,)
    shape_cases = SHAPE_CASES if args.shape_case == "all" else (args.shape_case,)
    results = [
        benchmark_case_mode(args, shape_case, mode)
        for shape_case in shape_cases
        for mode in modes
    ]
    report = {
        "benchmark": "grouped_fp8_tensor_scaling_quantize",
        "profile_enabled": args.profile,
        "profile_after_warmup": True,
        "regular_iterations": args.iters,
        "profile_iterations": args.profile_iters,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
