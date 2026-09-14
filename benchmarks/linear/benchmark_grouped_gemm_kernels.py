# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark multi-stream and cuBLASLt grouped GEMM kernels.

Unlike ``benchmark_grouped_linear_cublas_grouped_gemm.py``, this benchmark does
not construct a GroupedLinear module or invoke autograd. Inputs are allocated
and, for MXFP8, quantized and GEMM-swizzled before timing. The timed region only
calls TE's low-level grouped GEMM wrappers.

The default shape is Qwen3.5-397B-A17B with sequence length 4096, top-10
routing, and EP32. Each rank owns 16 experts and processes 2560 rows per expert.
All six expert GEMMs are measured independently:

============== ====== ====== ====== ====== ==================================
Projection     Pass   Layout M      K      N
============== ====== ====== ====== ====== ==================================
FC1            fwd    TN     2560   4096   2048
FC1            dgrad  NN     2560   4096   2048
FC1            wgrad  NT     2560   4096   2048
FC2            fwd    TN     2560   1024   4096
FC2            dgrad  NN     2560   1024   4096
FC2            wgrad  NT     2560   1024   4096
============== ====== ====== ====== ====== ==================================

The layouts describe TE's operands for each expert:

* Forward (TN): A=``weight[N,K]``, B=``input[M,K]``;
  ``input @ weight.T -> output[M,N]``.
* Dgrad (NN): A=``weight[N,K]``, B=``dy[M,N]``;
  ``dy @ weight -> dx[M,K]``.
* Wgrad (NT): A=``input[M,K]``, B=``dy[M,N]``;
  ``dy.T @ input -> dweight[N,K]``.

The multi-stream path receives lists of discrete tensors. The cuBLASLt grouped
path receives packed GroupedTensor activations and discrete weights/wgrads,
matching GroupedLinear with discrete parameters.

Examples
--------
Run the full BF16 and MXFP8 matrix:

    python benchmarks/linear/benchmark_grouped_gemm_kernels.py

Run only MXFP8 FC1 wgrad:

    python benchmarks/linear/benchmark_grouped_gemm_kernels.py \
        --precision mxfp8 --projection fc1 --gemm wgrad
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

from transformer_engine.common.recipe import MXFP8BlockScaling, Recipe
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.cpp_extensions import (
    general_grouped_gemm,
    general_grouped_gemm_for_grouped_tensor,
)
from transformer_engine.pytorch.module import is_module_grouped_tensor_path_supported
from transformer_engine.pytorch.module.base import (
    _2X_ACC_DGRAD,
    _2X_ACC_FPROP,
    _2X_ACC_WGRAD,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor import GroupedTensor, GroupedTensorStorage
import transformer_engine_torch as tex


QWEN_NUM_EXPERTS = 512
QWEN_TOP_K = 10
QWEN_SEQUENCE_LENGTH = 4096
QWEN_EXPERT_PARALLEL_SIZE = 32
QWEN_HIDDEN_SIZE = 4096
QWEN_MOE_INTERMEDIATE_SIZE = 1024


@dataclass(frozen=True)
class GemmSpec:
    """One expert GEMM in the Qwen routed MLP."""

    projection: str
    gemm: str
    layout: str
    k: int
    n: int


GEMM_SPECS = tuple(
    GemmSpec(projection, gemm, layout, k, n)
    for projection, k, n in (
        ("fc1", QWEN_HIDDEN_SIZE, 2 * QWEN_MOE_INTERMEDIATE_SIZE),
        ("fc2", QWEN_MOE_INTERMEDIATE_SIZE, QWEN_HIDDEN_SIZE),
    )
    for gemm, layout in (("fwd", "TN"), ("dgrad", "NN"), ("wgrad", "NT"))
)


@dataclass
class PreparedGemm:
    """Preallocated operands and outputs for both grouped GEMM paths."""

    spec: GemmSpec
    split_sizes: list[int]
    multistream_a: list[Any]
    multistream_b: list[Any]
    multistream_out: list[torch.Tensor]
    grouped_a: Any
    grouped_b: GroupedTensorStorage
    grouped_out: Any


def _make_grouped_bf16(
    packed: Optional[torch.Tensor],
    split_sizes: list[int],
    last_dim: int,
) -> GroupedTensorStorage:
    """Create packed GroupedTensor storage and optionally initialize its data."""
    first_dims = torch.tensor(split_sizes, dtype=torch.int64, device="cuda")
    grouped = GroupedTensor.make_grouped_tensor(
        num_tensors=len(split_sizes),
        first_dims=first_dims,
        last_dims=None,
        logical_first_dim=sum(split_sizes),
        logical_last_dim=last_dim,
        quantizer=None,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    if packed is not None:
        grouped.rowwise_data.view(-1).copy_(packed.reshape(-1))
    return grouped


def _new_mxfp8_quantizer(*, rowwise: bool, columnwise: bool) -> MXFP8Quantizer:
    """Construct a quantizer that emits GEMM-ready scales before timing."""
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    quantizer.optimize_for_gemm = True
    return quantizer


def _quantize_discrete_mxfp8(
    tensors: list[torch.Tensor],
    *,
    rowwise: bool,
    columnwise: bool,
) -> list[Any]:
    """Quantize per-expert tensors outside the timed region."""
    quantizer = _new_mxfp8_quantizer(rowwise=rowwise, columnwise=columnwise)
    return [quantizer(tensor) for tensor in tensors]


def _quantize_grouped_mxfp8(
    packed: torch.Tensor,
    split_sizes: list[int],
    *,
    rowwise: bool,
    columnwise: bool,
) -> GroupedTensorStorage:
    """Quantize one packed activation into GEMM-ready grouped storage."""
    quantizer = _new_mxfp8_quantizer(rowwise=rowwise, columnwise=columnwise)
    first_dims = torch.tensor(split_sizes, dtype=torch.int64, device="cuda")
    return tex.group_quantize(
        packed,
        quantizer,
        len(split_sizes),
        first_dims,
    )


def _make_high_precision_operands(
    spec: GemmSpec,
    split_sizes: list[int],
) -> tuple[list[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Create canonical BF16 operands shared by both precision paths."""
    num_experts = len(split_sizes)
    total_rows = sum(split_sizes)

    if spec.gemm in ("fwd", "dgrad"):
        weights = [
            torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_experts)
        ]
        b_width = spec.k if spec.gemm == "fwd" else spec.n
        packed_b = torch.randn(total_rows, b_width, dtype=torch.bfloat16, device="cuda")
        return weights, None, packed_b

    packed_x = torch.randn(total_rows, spec.k, dtype=torch.bfloat16, device="cuda")
    packed_dy = torch.randn(total_rows, spec.n, dtype=torch.bfloat16, device="cuda")
    return [], packed_x, packed_dy


def _operand_usage(spec: GemmSpec) -> tuple[tuple[bool, bool], tuple[bool, bool]]:
    """Return rowwise/columnwise MXFP8 storage used by A and B."""
    if spec.gemm == "fwd":
        return (True, False), (True, False)
    if spec.gemm == "dgrad":
        return (False, True), (True, False)
    return (False, True), (False, True)


def _allocate_outputs(
    spec: GemmSpec,
    split_sizes: list[int],
) -> tuple[list[torch.Tensor], Any]:
    """Allocate output storage matching each low-level API's native contract."""
    num_experts = len(split_sizes)
    if spec.gemm == "wgrad":
        multistream_out = [
            torch.empty(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_experts)
        ]
        grouped_out = [
            torch.empty(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_experts)
        ]
        return multistream_out, grouped_out

    out_features = spec.n if spec.gemm == "fwd" else spec.k
    multistream_out = [
        torch.empty(sum(split_sizes), out_features, dtype=torch.bfloat16, device="cuda")
    ]
    grouped_out = _make_grouped_bf16(None, split_sizes, out_features)
    return multistream_out, grouped_out


def _prepare_gemm(
    spec: GemmSpec,
    split_sizes: list[int],
    precision: str,
) -> PreparedGemm:
    """Prepare one GEMM without leaving quantization inside the timed region."""
    hp_a, packed_a, packed_b = _make_high_precision_operands(spec, split_sizes)
    a_usage, b_usage = _operand_usage(spec)

    if spec.gemm in ("fwd", "dgrad"):
        if precision == "mxfp8":
            multistream_a = _quantize_discrete_mxfp8(
                hp_a,
                rowwise=a_usage[0],
                columnwise=a_usage[1],
            )
        else:
            multistream_a = hp_a
        grouped_a = multistream_a
    elif precision == "mxfp8":
        assert packed_a is not None
        grouped_a = _quantize_grouped_mxfp8(
            packed_a,
            split_sizes,
            rowwise=a_usage[0],
            columnwise=a_usage[1],
        )
    else:
        assert packed_a is not None
        grouped_a = _make_grouped_bf16(packed_a, split_sizes, spec.k)
    if spec.gemm == "wgrad":
        multistream_a = list(grouped_a.split_into_quantized_tensors())

    if precision == "mxfp8":
        grouped_b = _quantize_grouped_mxfp8(
            packed_b,
            split_sizes,
            rowwise=b_usage[0],
            columnwise=b_usage[1],
        )
    else:
        b_width = spec.k if spec.gemm == "fwd" else spec.n
        grouped_b = _make_grouped_bf16(packed_b, split_sizes, b_width)
    # Give multi-stream zero-copy member views into the exact same packed operand used by
    # cuBLASLt. This prevents quantization or input-data differences from entering the result.
    multistream_b = list(grouped_b.split_into_quantized_tensors())

    multistream_out, grouped_out = _allocate_outputs(spec, split_sizes)
    prepared = PreparedGemm(
        spec=spec,
        split_sizes=split_sizes,
        multistream_a=multistream_a,
        multistream_b=multistream_b,
        multistream_out=multistream_out,
        grouped_a=grouped_a,
        grouped_b=grouped_b,
        grouped_out=grouped_out,
    )
    if precision == "mxfp8":
        _assert_mxfp8_operands_are_gemm_ready(prepared)
    return prepared


def _iter_operands(operand: Any):
    """Yield discrete members or one packed operand."""
    if isinstance(operand, (list, tuple)):
        yield from operand
    else:
        yield operand


def _assert_mxfp8_operands_are_gemm_ready(prepared: PreparedGemm) -> None:
    """Ensure scale swizzling cannot leak into the GEMM timing."""
    operands = (
        prepared.multistream_a,
        prepared.multistream_b,
        prepared.grouped_a,
        prepared.grouped_b,
    )
    for operand_group in operands:
        for operand in _iter_operands(operand_group):
            assert getattr(
                operand, "_with_gemm_swizzled_scales", False
            ), "MXFP8 operand was not prepared with GEMM-swizzled scales"


def _use_split_accumulator(spec: GemmSpec) -> bool:
    """Match GroupedLinear's accumulator policy for each training GEMM."""
    if spec.gemm == "fwd":
        return _2X_ACC_FPROP
    if spec.gemm == "dgrad":
        return _2X_ACC_DGRAD
    return _2X_ACC_WGRAD


def _run_multistream(prepared: PreparedGemm, iterations: int) -> None:
    """Launch the multi-stream grouped GEMM repeatedly."""
    spec = prepared.spec
    for _ in range(iterations):
        general_grouped_gemm(
            prepared.multistream_a,
            prepared.multistream_b,
            prepared.multistream_out,
            [None] * len(prepared.split_sizes),
            torch.bfloat16,
            layout=spec.layout,
            m_splits=prepared.split_sizes,
            grad=spec.gemm != "fwd",
            single_output=spec.gemm != "wgrad",
            use_split_accumulator=_use_split_accumulator(spec),
        )


def _run_cublas_grouped(prepared: PreparedGemm, iterations: int) -> None:
    """Launch the device-described cuBLASLt grouped GEMM repeatedly."""
    for _ in range(iterations):
        general_grouped_gemm_for_grouped_tensor(
            prepared.grouped_a,
            prepared.grouped_b,
            prepared.grouped_out,
            layout=prepared.spec.layout,
            use_split_accumulator=_use_split_accumulator(prepared.spec),
        )


def _canonical_output(prepared: PreparedGemm, *, grouped: bool) -> torch.Tensor:
    """Return either path's output in one comparable packed tensor."""
    output = prepared.grouped_out if grouped else prepared.multistream_out
    if prepared.spec.gemm == "wgrad":
        return torch.stack(output, dim=0)
    if grouped:
        out_features = prepared.spec.n if prepared.spec.gemm == "fwd" else prepared.spec.k
        return output.rowwise_data.view(sum(prepared.split_sizes), out_features)
    return output[0]


def _validate_outputs(prepared: PreparedGemm, precision: str) -> None:
    """Check that execution-path selection does not change GEMM numerics unexpectedly."""
    _run_multistream(prepared, 1)
    _run_cublas_grouped(prepared, 1)
    if precision == "mxfp8":
        tolerances = {"rtol": 0.125, "atol": 0.0675}
    else:
        tolerances = {"rtol": 1e-2, "atol": 1e-2}
    torch.testing.assert_close(
        _canonical_output(prepared, grouped=True),
        _canonical_output(prepared, grouped=False),
        **tolerances,
    )


def _benchmark_path(
    prepared: PreparedGemm,
    *,
    path: str,
    iterations_per_run: int,
    warmup_iterations: int,
    min_run_time: float,
    profile: bool,
    label: str,
) -> float:
    """Return milliseconds for one grouped GEMM launch."""
    run = _run_multistream if path == "multistream" else _run_cublas_grouped
    run(prepared, warmup_iterations)
    torch.cuda.synchronize()

    if profile:
        torch.cuda.nvtx.range_push(label)
    timing = benchmark.Timer(
        stmt="run(prepared, iterations_per_run)",
        globals={
            "run": run,
            "prepared": prepared,
            "iterations_per_run": iterations_per_run,
        },
        num_threads=1,
    ).blocked_autorange(min_run_time=min_run_time)
    if profile:
        torch.cuda.nvtx.range_pop()
    return timing.median * 1000 / iterations_per_run


def _gemm_tflops(spec: GemmSpec, total_rows: int, time_ms: float) -> float:
    """Compute aggregate GEMM throughput across all local experts."""
    flops = 2 * total_rows * spec.k * spec.n
    return flops / time_ms / 1e9


def _shape_description(spec: GemmSpec, rows_per_expert: str) -> str:
    """Format the actual per-expert operands passed to TE."""
    if spec.gemm == "fwd":
        return (
            f"A[{spec.n},{spec.k}] B[{rows_per_expert},{spec.k}] -> D[{rows_per_expert},{spec.n}]"
        )
    if spec.gemm == "dgrad":
        return (
            f"A[{spec.n},{spec.k}] B[{rows_per_expert},{spec.n}] -> D[{rows_per_expert},{spec.k}]"
        )
    return f"A[{rows_per_expert},{spec.k}] B[{rows_per_expert},{spec.n}] -> D[{spec.n},{spec.k}]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("all", "bf16", "mxfp8"), default="all")
    parser.add_argument("--projection", choices=("all", "fc1", "fc2"), default="all")
    parser.add_argument("--gemm", choices=("all", "fwd", "dgrad", "wgrad"), default="all")
    parser.add_argument("--iterations-per-run", type=int, default=1000)
    parser.add_argument("--warmup-iterations", type=int, default=500)
    parser.add_argument("--min-run-time", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--m-splits",
        type=str,
        default=None,
        help="Optional comma-separated per-expert rows; defaults to Qwen3.5-397B EP32.",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Add per-path NVTX ranges.")
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if os.getenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "0") == "1":
        raise RuntimeError(
            "Unset NVTE_USE_CUTLASS_GROUPED_GEMM: this benchmark requires the "
            "multi-stream cuBLAS baseline."
        )
    if args.iterations_per_run < 1 or args.warmup_iterations < 1:
        raise ValueError("iterations-per-run and warmup-iterations must be positive.")

    if args.m_splits is None:
        num_local_experts = QWEN_NUM_EXPERTS // QWEN_EXPERT_PARALLEL_SIZE
        rows_per_expert = QWEN_SEQUENCE_LENGTH * QWEN_TOP_K // num_local_experts
        split_sizes = [rows_per_expert] * num_local_experts
    else:
        split_sizes = [int(value) for value in args.m_splits.split(",") if value]
        if not split_sizes or any(value <= 0 for value in split_sizes):
            raise ValueError("m_splits must contain positive integers.")
        num_local_experts = len(split_sizes)

    total_rows = sum(split_sizes)
    precisions = ("bf16", "mxfp8") if args.precision == "all" else (args.precision,)
    specs = [
        spec
        for spec in GEMM_SPECS
        if (args.projection == "all" or spec.projection == args.projection)
        and (args.gemm == "all" or spec.gemm == args.gemm)
    ]
    recipes: dict[str, Optional[Recipe]] = {
        "bf16": None,
        "mxfp8": MXFP8BlockScaling(),
    }
    mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

    uniform_rows = str(split_sizes[0]) if len(set(split_sizes)) == 1 else "variable"
    print("Qwen3.5-397B-A17B grouped GEMM kernel benchmark")
    print(f"  local experts: {num_local_experts}")
    print(f"  total rows: {total_rows}")
    print(f"  m_splits: {split_sizes}")
    print("  quantization is outside the timed region")
    print()
    for spec in specs:
        print(
            f"  {spec.projection} {spec.gemm:5s} {spec.layout}: "
            f"{_shape_description(spec, uniform_rows)}"
        )
    print()

    rows = []
    for precision in precisions:
        recipe = recipes[precision]
        if precision == "mxfp8" and not mxfp8_available:
            print(f"Skipping MXFP8: {reason_for_no_mxfp8}")
            continue
        if not is_module_grouped_tensor_path_supported(recipe, torch.bfloat16):
            print(
                f"Skipping {precision}: cuBLASLt grouped GEMM is unsupported on this "
                "GPU or cuBLASLt version."
            )
            continue

        for spec in specs:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            prepared = _prepare_gemm(spec, split_sizes, precision)
            if not args.skip_correctness:
                _validate_outputs(prepared, precision)

            timings = {}
            for path in ("multistream", "cublas_grouped_gemm"):
                label = f"{precision}_{spec.projection}_{spec.gemm}_{path}"
                timings[path] = _benchmark_path(
                    prepared,
                    path=path,
                    iterations_per_run=args.iterations_per_run,
                    warmup_iterations=args.warmup_iterations,
                    min_run_time=args.min_run_time,
                    profile=args.profile,
                    label=label,
                )

            speedup = timings["multistream"] / timings["cublas_grouped_gemm"]
            for path, time_ms in timings.items():
                rows.append(
                    {
                        "precision": precision,
                        "projection": spec.projection,
                        "gemm": spec.gemm,
                        "layout": spec.layout,
                        "execution_path": path,
                        "num_local_experts": num_local_experts,
                        "total_rows": total_rows,
                        "k": spec.k,
                        "n": spec.n,
                        "time_ms": time_ms,
                        "tflops": _gemm_tflops(spec, total_rows, time_ms),
                        "speedup_vs_multistream": (
                            speedup if path == "cublas_grouped_gemm" else 1.0
                        ),
                    }
                )

            print(
                f"{precision:6s} {spec.projection} {spec.gemm:5s}: "
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
