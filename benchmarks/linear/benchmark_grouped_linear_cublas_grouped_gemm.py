# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare GroupedLinear's multi-stream and cuBLASLt grouped GEMM paths.

The default problem is the routed-expert shape from Qwen3.5-397B-A17B with
sequence length 4096, top-10 routing, and expert parallelism 32:

    512 global experts / EP32 = 16 local experts
    4096 tokens * top-10 / 16 local experts = 2560 rows per expert

The 2560-row split is already 256-aligned, so both paths execute identical
GEMM work. FC1 and FC2 are benchmarked independently:

    FC1: 16 x (M=2560, K=4096, N=2048)
    FC2: 16 x (M=2560, K=1024, N=4096)

Both paths use discrete parameters initialized with identical values:

* ``use_grouped_tensor=False`` receives CPU splits and launches the
  multi-stream grouped GEMM implementation.
* ``use_grouped_tensor=True`` receives CUDA int64 splits and launches the
  cuBLASLt grouped GEMM implementation.

The MXFP8 cases use BF16 primary parameters. Each timing bundle refreshes the
MXFP8 weight cache on its first microbatch and reuses it on later microbatches.

Examples
--------
Run the complete BF16 and MXFP8 comparison:

    python benchmarks/linear/benchmark_grouped_linear_cublas_grouped_gemm.py

Run only MXFP8 FC1 forward and backward:

    python benchmarks/linear/benchmark_grouped_linear_cublas_grouped_gemm.py \
        --precision mxfp8 --projection fc1 --mode fwd_bwd

Profile one microbatch per timing invocation:

    nsys profile \
        --output=grouped_linear_cublas_grouped_gemm \
        --force-overwrite=true \
        --trace=cuda,nvtx,cublas \
        python benchmarks/linear/benchmark_grouped_linear_cublas_grouped_gemm.py \
            --precision mxfp8 --projection fc1 --mode fwd_bwd \
            --num-microbatches 1 --profile
"""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling, Recipe
from transformer_engine.pytorch.module import (
    GroupedLinear,
    is_module_grouped_tensor_path_supported,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


QWEN_NUM_EXPERTS = 512
QWEN_TOP_K = 10
QWEN_SEQUENCE_LENGTH = 4096
QWEN_EXPERT_PARALLEL_SIZE = 32
QWEN_HIDDEN_SIZE = 4096
QWEN_MOE_INTERMEDIATE_SIZE = 1024


@dataclass(frozen=True)
class Projection:
    """GroupedLinear dimensions for one Qwen routed-expert projection."""

    name: str
    in_features: int
    out_features: int


PROJECTIONS = {
    "fc1": Projection(
        name="fc1",
        in_features=QWEN_HIDDEN_SIZE,
        out_features=2 * QWEN_MOE_INTERMEDIATE_SIZE,
    ),
    "fc2": Projection(
        name="fc2",
        in_features=QWEN_MOE_INTERMEDIATE_SIZE,
        out_features=QWEN_HIDDEN_SIZE,
    ),
}


@dataclass(frozen=True)
class ExecutionPath:
    """GroupedLinear path and the corresponding m_splits representation."""

    name: str
    use_grouped_tensor: bool


EXECUTION_PATHS = (
    ExecutionPath(name="multistream", use_grouped_tensor=False),
    ExecutionPath(name="cublas_grouped_gemm", use_grouped_tensor=True),
)


def _quantization_context(recipe: Optional[Recipe]):
    """Construct a fresh quantization context for one invocation."""
    if recipe is None:
        return nullcontext()
    return te.autocast(enabled=True, recipe=recipe)


def _make_m_splits(
    *,
    use_grouped_tensor: bool,
    split_sizes: list[int],
) -> torch.Tensor:
    """Use the split representation consumed natively by each execution path."""
    device = "cuda" if use_grouped_tensor else "cpu"
    return torch.tensor(split_sizes, dtype=torch.int64, device=device)


def _build_layer(
    *,
    projection: Projection,
    num_local_experts: int,
    use_grouped_tensor: bool,
) -> GroupedLinear:
    """Construct a discrete-parameter GroupedLinear module."""
    return GroupedLinear(
        num_local_experts,
        projection.in_features,
        projection.out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        use_grouped_tensor=use_grouped_tensor,
    )


def _run_microbatches(
    layer: GroupedLinear,
    x: torch.Tensor,
    m_splits: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    recipe: Optional[Recipe],
    mode: str,
    num_microbatches: int,
) -> torch.Tensor:
    """Run one timed bundle while preserving realistic MXFP8 weight caching."""
    layer.zero_grad(set_to_none=True)
    x.grad = None

    if mode == "fwd":
        with torch.no_grad(), _quantization_context(recipe):
            for microbatch in range(num_microbatches):
                output = layer(
                    x,
                    m_splits,
                    is_first_microbatch=(microbatch == 0),
                )
        return output

    with _quantization_context(recipe):
        for microbatch in range(num_microbatches):
            output = layer(
                x,
                m_splits,
                is_first_microbatch=(microbatch == 0),
            )
            output.backward(grad_output)
    return output


def _run_correctness_step(
    layer: GroupedLinear,
    x: torch.Tensor,
    m_splits: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    recipe: Optional[Recipe],
    mode: str,
) -> tuple[torch.Tensor, Optional[torch.Tensor], list[Optional[torch.Tensor]]]:
    """Run one microbatch and retain outputs and gradients for path parity."""
    output = _run_microbatches(
        layer,
        x,
        m_splits,
        grad_output,
        recipe=recipe,
        mode=mode,
        num_microbatches=1,
    )
    input_grad = None if x.grad is None else x.grad.detach().clone()
    parameter_grads = [
        None if param.grad is None else param.grad.detach().clone() for param in layer.parameters()
    ]
    return output.detach().clone(), input_grad, parameter_grads


def _validate_path_parity(
    *,
    multistream_layer: GroupedLinear,
    grouped_layer: GroupedLinear,
    multistream_x: torch.Tensor,
    grouped_x: torch.Tensor,
    multistream_splits: torch.Tensor,
    grouped_splits: torch.Tensor,
    grad_output: torch.Tensor,
    recipe: Optional[Recipe],
    mode: str,
) -> None:
    """Require the two execution paths to produce compatible results."""
    reference = _run_correctness_step(
        multistream_layer,
        multistream_x,
        multistream_splits,
        grad_output,
        recipe=recipe,
        mode=mode,
    )
    actual = _run_correctness_step(
        grouped_layer,
        grouped_x,
        grouped_splits,
        grad_output,
        recipe=recipe,
        mode=mode,
    )

    tolerances = {"rtol": 1e-2, "atol": 1e-2}

    torch.testing.assert_close(actual[0], reference[0], **tolerances)
    if mode == "fwd":
        return

    assert actual[1] is not None and reference[1] is not None
    torch.testing.assert_close(actual[1], reference[1], **tolerances)
    assert len(actual[2]) == len(reference[2])
    for actual_grad, reference_grad in zip(actual[2], reference[2]):
        assert actual_grad is not None and reference_grad is not None
        torch.testing.assert_close(actual_grad, reference_grad, **tolerances)


def _benchmark_path(
    *,
    layer: GroupedLinear,
    x: torch.Tensor,
    m_splits: torch.Tensor,
    grad_output: torch.Tensor,
    recipe: Optional[Recipe],
    mode: str,
    num_microbatches: int,
    warmup_steps: int,
    min_run_time: float,
    profile: bool,
    label: str,
) -> float:
    """Benchmark one execution path and return milliseconds per microbatch."""
    _run_microbatches(
        layer,
        x,
        m_splits,
        grad_output,
        recipe=recipe,
        mode=mode,
        num_microbatches=warmup_steps,
    )
    torch.cuda.synchronize()

    if profile:
        torch.cuda.nvtx.range_push(label)

    timing = benchmark.Timer(
        stmt=(
            "_run_microbatches(layer, x, m_splits, grad_output, recipe=recipe, "
            "mode=mode, num_microbatches=num_microbatches)"
        ),
        globals={
            "_run_microbatches": _run_microbatches,
            "layer": layer,
            "x": x,
            "m_splits": m_splits,
            "grad_output": grad_output,
            "recipe": recipe,
            "mode": mode,
            "num_microbatches": num_microbatches,
        },
        num_threads=1,
    ).blocked_autorange(min_run_time=min_run_time)

    if profile:
        torch.cuda.nvtx.range_pop()

    return timing.median * 1000 / num_microbatches


def _gemm_tflops(
    *,
    total_rows: int,
    projection: Projection,
    mode: str,
    time_ms: float,
) -> float:
    """Report GEMM FLOP/s while timing also includes quantization and Python overhead."""
    flops = 2 * total_rows * projection.in_features * projection.out_features
    if mode == "fwd_bwd":
        flops *= 3
    return flops / time_ms / 1e9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        choices=("all", "bf16", "mxfp8"),
        default="all",
    )
    parser.add_argument(
        "--projection",
        choices=("all", "fc1", "fc2"),
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=("fwd", "fwd_bwd"),
        default="fwd_bwd",
    )
    parser.add_argument("--num-microbatches", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-run-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--m-splits",
        type=str,
        default=None,
        help="Optional comma-separated per-expert rows; defaults to Qwen3.5-397B EP32.",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip output, input-gradient, and parameter-gradient parity checks.",
    )
    parser.add_argument("--profile", action="store_true", help="Add per-path NVTX ranges.")
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.num_microbatches < 1 or args.warmup_steps < 1:
        raise ValueError("num_microbatches and warmup_steps must both be positive.")

    if args.m_splits is None:
        num_local_experts = QWEN_NUM_EXPERTS // QWEN_EXPERT_PARALLEL_SIZE
        rows_per_expert = QWEN_SEQUENCE_LENGTH * QWEN_TOP_K // num_local_experts
        split_sizes = [rows_per_expert] * num_local_experts
    else:
        split_sizes = [int(value) for value in args.m_splits.split(",") if value]
        if not split_sizes or any(value < 0 for value in split_sizes):
            raise ValueError("m_splits must contain non-negative integers.")
        num_local_experts = len(split_sizes)

    total_rows = sum(split_sizes)
    if total_rows == 0:
        raise ValueError("At least one routed token row is required for this benchmark.")

    precision_names = ("bf16", "mxfp8") if args.precision == "all" else (args.precision,)
    projection_names = ("fc1", "fc2") if args.projection == "all" else (args.projection,)

    recipes: dict[str, Optional[Recipe]] = {
        "bf16": None,
        "mxfp8": MXFP8BlockScaling(),
    }
    mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

    print("Qwen3.5-397B-A17B GroupedLinear benchmark")
    print(f"  local experts: {num_local_experts}")
    print(f"  total routed rows: {total_rows}")
    print(f"  m_splits: {split_sizes}")
    print(f"  mode: {args.mode}")
    print("  primary parameter dtype: BF16")
    print(f"  microbatches per timing invocation: {args.num_microbatches}")
    print()

    rows = []
    for precision_name in precision_names:
        recipe = recipes[precision_name]
        if precision_name == "mxfp8" and not mxfp8_available:
            print(f"Skipping MXFP8: {reason_for_no_mxfp8}")
            continue
        if not is_module_grouped_tensor_path_supported(recipe, torch.bfloat16):
            print(
                f"Skipping {precision_name}: the cuBLASLt grouped-tensor path is unsupported "
                "on this GPU or cuBLASLt version."
            )
            continue

        for projection_name in projection_names:
            projection = PROJECTIONS[projection_name]
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

            layers = {
                path.name: _build_layer(
                    projection=projection,
                    num_local_experts=num_local_experts,
                    use_grouped_tensor=path.use_grouped_tensor,
                )
                for path in EXECUTION_PATHS
            }
            layers["cublas_grouped_gemm"].load_state_dict(layers["multistream"].state_dict())

            base_x = torch.randn(
                total_rows,
                projection.in_features,
                dtype=torch.bfloat16,
                device="cuda",
            )
            inputs = {
                path.name: base_x.detach().clone().requires_grad_(args.mode == "fwd_bwd")
                for path in EXECUTION_PATHS
            }
            grad_output = torch.randn(
                total_rows,
                projection.out_features,
                dtype=torch.bfloat16,
                device="cuda",
            )
            splits = {
                path.name: _make_m_splits(
                    use_grouped_tensor=path.use_grouped_tensor,
                    split_sizes=split_sizes,
                )
                for path in EXECUTION_PATHS
            }

            if not args.skip_correctness:
                _validate_path_parity(
                    multistream_layer=layers["multistream"],
                    grouped_layer=layers["cublas_grouped_gemm"],
                    multistream_x=inputs["multistream"],
                    grouped_x=inputs["cublas_grouped_gemm"],
                    multistream_splits=splits["multistream"],
                    grouped_splits=splits["cublas_grouped_gemm"],
                    grad_output=grad_output,
                    recipe=recipe,
                    mode=args.mode,
                )

            timings = {}
            for path in EXECUTION_PATHS:
                label = f"{precision_name}_{projection_name}_{args.mode}_{path.name}"
                timing_ms = _benchmark_path(
                    layer=layers[path.name],
                    x=inputs[path.name],
                    m_splits=splits[path.name],
                    grad_output=grad_output,
                    recipe=recipe,
                    mode=args.mode,
                    num_microbatches=args.num_microbatches,
                    warmup_steps=args.warmup_steps,
                    min_run_time=args.min_run_time,
                    profile=args.profile,
                    label=label,
                )
                timings[path.name] = timing_ms

            speedup = timings["multistream"] / timings["cublas_grouped_gemm"]
            for path in EXECUTION_PATHS:
                timing_ms = timings[path.name]
                rows.append(
                    {
                        "precision": precision_name,
                        "projection": projection_name,
                        "mode": args.mode,
                        "execution_path": path.name,
                        "num_local_experts": num_local_experts,
                        "total_rows": total_rows,
                        "in_features": projection.in_features,
                        "out_features": projection.out_features,
                        "time_ms": timing_ms,
                        "gemm_tflops": _gemm_tflops(
                            total_rows=total_rows,
                            projection=projection,
                            mode=args.mode,
                            time_ms=timing_ms,
                        ),
                        "speedup_vs_multistream": (1.0 if path.name == "multistream" else speedup),
                    }
                )

            print(
                f"{precision_name:6s} {projection_name} {args.mode}: "
                f"multistream={timings['multistream']:.3f} ms, "
                f"cuBLAS grouped={timings['cublas_grouped_gemm']:.3f} ms, "
                f"speedup={speedup:.3f}x"
            )

    results = pd.DataFrame(rows)
    print()
    print(results.to_string(index=False))
    if args.output_csv is not None:
        results.to_csv(args.output_csv, index=False)
        print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
