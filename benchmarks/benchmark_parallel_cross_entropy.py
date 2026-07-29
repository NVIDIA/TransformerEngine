# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark fused cross entropy saved-state and peak-memory tradeoffs."""

import argparse
import gc
from statistics import mean

import torch

from transformer_engine.pytorch import (
    parallel_cross_entropy,
    parallel_cross_entropy_recompute,
)

MIB = 1024**2


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=128000)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    return parser.parse_args()


def _operator(implementation, logits, target, label_smoothing):
    if implementation == "existing":
        return parallel_cross_entropy(
            logits,
            target,
            label_smoothing=label_smoothing,
            reduce_loss=True,
        )
    return parallel_cross_entropy_recompute(
        logits,
        target,
        label_smoothing=label_smoothing,
        reduce_loss=True,
        overwrite_input=implementation == "destructive",
    )


def _make_inputs(shape, dtype):
    gc.collect()
    torch.cuda.empty_cache()
    caller_input = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    target = torch.randint(0, shape[-1], shape[:-1], device="cuda")
    torch.cuda.synchronize()
    return caller_input, target


def _measure_memory(implementation, shape, dtype, label_smoothing):
    """Measure memory while keeping the caller-owned input alive throughout."""

    caller_input, target = _make_inputs(shape, dtype)
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    loss = _operator(implementation, caller_input, target, label_smoothing)
    torch.cuda.synchronize()
    forward_peak = torch.cuda.max_memory_allocated()
    forward_end_live = torch.cuda.memory_allocated()

    backward_entry = forward_end_live
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    backward_peak = torch.cuda.max_memory_allocated()
    backward_end_live = torch.cuda.memory_allocated()

    # Referencing the input here makes its consistent lifetime explicit. In
    # destructive mode its storage now contains the derivative.
    assert caller_input.data_ptr() != 0
    return {
        "baseline": baseline,
        "forward_peak": forward_peak,
        "forward_end": forward_end_live,
        "backward_entry": backward_entry,
        "backward_peak": backward_peak,
        "backward_end": backward_end_live,
        "e2e_peak": max(forward_peak, backward_peak),
    }


def _measure_latency(implementation, shape, dtype, label_smoothing):
    """Measure uninterrupted forward and backward device latency."""

    caller_input, target = _make_inputs(shape, dtype)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)

    forward_start.record()
    loss = _operator(implementation, caller_input, target, label_smoothing)
    forward_end.record()
    loss.backward()
    backward_end.record()
    torch.cuda.synchronize()

    assert caller_input.data_ptr() != 0
    return {
        "forward_ms": forward_start.elapsed_time(forward_end),
        "backward_ms": forward_end.elapsed_time(backward_end),
        "total_ms": forward_start.elapsed_time(backward_end),
    }


def _format_memory(value, baseline):
    return f"{value / MIB:9.1f} / +{(value - baseline) / MIB:7.1f}"


def _print_results(results):
    memory_fields = (
        ("fwd peak", "forward_peak"),
        ("fwd end", "forward_end"),
        ("bwd entry", "backward_entry"),
        ("bwd peak", "backward_peak"),
        ("e2e peak", "e2e_peak"),
        ("bwd end", "backward_end"),
    )
    header = ["implementation", "baseline MiB"]
    header.extend(f"{label} abs/+inc MiB" for label, _ in memory_fields)
    header.extend(("fwd ms", "bwd ms", "total ms"))
    rows = []
    for implementation, measurements in results.items():
        memory_sample = measurements["memory"]
        timings = measurements["timings"]
        row = [implementation, f"{memory_sample['baseline'] / MIB:.1f}"]
        row.extend(
            _format_memory(memory_sample[field], memory_sample["baseline"])
            for _, field in memory_fields
        )
        row.extend(
            f"{mean(sample[field] for sample in timings):.3f}"
            for field in ("forward_ms", "backward_ms", "total_ms")
        )
        rows.append(row)

    widths = [max(len(header[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(header))]
    print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(header)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def main():
    """Run the benchmark."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    shape = (args.batch_size, args.sequence_length, args.vocab_size)
    n_logits = args.batch_size * args.sequence_length * args.vocab_size
    n_rows = args.batch_size * args.sequence_length
    input_bytes = n_logits * dtype.itemsize

    expected_multipliers = {
        "existing": dtype.itemsize + 4,
        "safe": 2 * dtype.itemsize,
        "destructive": dtype.itemsize,
    }
    print(f"shape={shape}, dtype={dtype}, live input={input_bytes / MIB:.1f} MiB")
    print("Expected logits-sized absolute forward peak (includes live caller input):")
    for name, bytes_per_logit in expected_multipliers.items():
        print(f"  {name:11s}: {bytes_per_logit}N = {bytes_per_logit * n_logits / MIB:.1f} MiB")
    print(
        "Recompute saved row metadata: "
        f"{(n_rows * (2 * 4 + 8) + 8) / MIB:.3f} MiB "
        "(FP32 max/denominator, int64 targets/count)"
    )
    print(f"Existing forward row temporaries: {(n_rows * (3 * 4 + 4) + 8) / MIB:.3f} MiB")
    print(f"Recompute forward loss temporary: {n_rows * 4 / MIB:.3f} MiB")
    print("Tensor-parallel communication buffers: 0 MiB (single-GPU benchmark)")
    print("Memory cells below are 'absolute / +incremental-from-pre-forward-baseline'.")

    implementations = ("existing", "safe", "destructive")
    for implementation in implementations:
        for _ in range(args.warmup):
            _measure_latency(implementation, shape, dtype, args.label_smoothing)

    results = {}
    for implementation in implementations:
        results[implementation] = {
            "memory": _measure_memory(implementation, shape, dtype, args.label_smoothing),
            "timings": [
                _measure_latency(implementation, shape, dtype, args.label_smoothing)
                for _ in range(args.trials)
            ],
        }
    _print_results(results)


if __name__ == "__main__":
    main()
