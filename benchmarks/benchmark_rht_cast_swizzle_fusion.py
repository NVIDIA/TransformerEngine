# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark NVFP4 RHT cast-fusion with vs without fused GEMM-swizzled SF output.

For each shape we measure two paths and two builds:

  * path = "quant_only":  just NVFP4Quantizer(x)
  * path = "quant_plus_swizzle": NVFP4Quantizer(x) + tex.swizzle_scales_for_gemm_(t)
    (this is what te.Linear -> tex.generic_gemm does right before the
    cuBLAS LT NVFP4 GEMM dispatch)

  * build = "baseline":  optimize_for_gemm=False
    -> quant kernel emits compact SF;
       tex.swizzle_scales_for_gemm_ launches the standalone
       swizzle_{row,col}_scaling_kernel pass before GEMM.
  * build = "swizzle_fusion":  optimize_for_gemm=True
    -> quant kernel emits GEMM-swizzled SF directly (via the
       kEnableSwizzleSFOutput compile-time switch in
       row_cast_col_hadamard_transform_cast_fusion.cu);
       tex.swizzle_scales_for_gemm_ early-returns and the standalone
       swizzle pass disappears from the timeline.

The wall-clock delta on the "quant_plus_swizzle" path is the production
saving of this PR.
"""

import argparse
import torch
import pandas as pd
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te  # noqa: F401 must be first per te-python-import-order
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


def make_quantizer(optimize_for_gemm: bool) -> NVFP4Quantizer:
    q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=True,
    )
    q.optimize_for_gemm = optimize_for_gemm
    return q


def _bench(stmt: str, globals_dict: dict, min_run_time: float) -> float:
    """Returns median wall-clock per call in microseconds."""
    timing = benchmark.Timer(
        stmt=stmt,
        globals=globals_dict,
        num_threads=1,
    ).blocked_autorange(min_run_time=min_run_time)
    return timing.median * 1e6


def run_shape(shape, min_run_time: float):
    M, K = shape
    assert M % 16 == 0 and K % 16 == 0, "Shape must be divisible by 16"

    x = torch.randn([M, K], dtype=torch.bfloat16, device="cuda")
    q_base = make_quantizer(optimize_for_gemm=False)
    q_swf = make_quantizer(optimize_for_gemm=True)

    # quant_only path
    quant_only_base_us = _bench(
        stmt="q(x)",
        globals_dict={"q": q_base, "x": x},
        min_run_time=min_run_time,
    )
    quant_only_swf_us = _bench(
        stmt="q(x)",
        globals_dict={"q": q_swf, "x": x},
        min_run_time=min_run_time,
    )

    # quant_plus_swizzle path (this is what te.Linear actually runs)
    quant_plus_swizzle_base_us = _bench(
        stmt="t = q(x); tex.swizzle_scales_for_gemm_(t)",
        globals_dict={"q": q_base, "x": x, "tex": tex},
        min_run_time=min_run_time,
    )
    quant_plus_swizzle_swf_us = _bench(
        stmt="t = q(x); tex.swizzle_scales_for_gemm_(t)",
        globals_dict={"q": q_swf, "x": x, "tex": tex},
        min_run_time=min_run_time,
    )

    saved_us = quant_plus_swizzle_base_us - quant_plus_swizzle_swf_us
    speedup = (
        quant_plus_swizzle_base_us / quant_plus_swizzle_swf_us
        if quant_plus_swizzle_swf_us > 0
        else float("inf")
    )

    print(
        f"  shape={shape}: quant_only base={quant_only_base_us:.2f}us, "
        f"SUT={quant_only_swf_us:.2f}us | "
        f"quant+swizzle base={quant_plus_swizzle_base_us:.2f}us, "
        f"SUT={quant_plus_swizzle_swf_us:.2f}us "
        f"-> saved {saved_us:.2f}us ({speedup:.2f}x)"
    )

    return {
        "shape": shape,
        "M": M,
        "K": K,
        "quant_only_base_us": quant_only_base_us,
        "quant_only_swf_us": quant_only_swf_us,
        "quant_plus_swizzle_base_us": quant_plus_swizzle_base_us,
        "quant_plus_swizzle_swf_us": quant_plus_swizzle_swf_us,
        "saved_us": saved_us,
        "speedup": speedup,
    }


# Nsight Compute Profiling Command (for verifying the swizzle kernel disappears):
# ncu -f -o swizzle_fusion --set=full \
#     --kernel-name "regex:swizzle_(row|col)_scaling_kernel|cast_col_hadamard_transform_cast_fusion" \
#     -s 5 -c 10 python benchmarks/benchmark_rht_cast_swizzle_fusion.py --profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run only one shape for use with ncu/nsys; longer min_run_time",
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=2.0,
        help="Minimum total measured time per cell in seconds (benchmark.Timer)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="benchmark_rht_cast_swizzle_fusion.csv",
        help="CSV output path",
    )
    args = parser.parse_args()

    if args.profile:
        print("Profiling mode enabled (single shape).")
        shapes = [(8192, 4096)]
        min_run_time = max(5.0, args.min_run_time)
    else:
        shapes = [
            # production-class shapes
            (8192, 5120),
            (8192, 10240),
            (8192, 2560),
            (8192, 11328),
            (8192, 3584),
            (5120, 8192),
            (10240, 8192),
            (2560, 8192),
            (11328, 8192),
            (3584, 8192),
            (4096, 16384),
            (14336, 16384),
        ]
        min_run_time = args.min_run_time

    print(
        f"NVFP4 RHT cast-fusion: swizzle-fusion (optimize_for_gemm=True) vs baseline. "
        f"min_run_time={min_run_time}s per cell, BF16 input, "
        f"rowwise+columnwise SF, RHT=True+post_rht_amax."
    )
    rows = []
    for shape in shapes:
        print(f"Running {shape} ...")
        rows.append(run_shape(shape, min_run_time))

    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print()
    print(df.to_string(index=False))
    df.to_csv(args.csv, index=False)
    print(f"\nWrote {args.csv}")
