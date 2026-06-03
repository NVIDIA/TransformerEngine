# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark direct NVFP4 4over6 quantization kernel paths."""

import argparse
import os

import torch
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.quantization import check_fp8_block_scaling_support
import transformer_engine_torch as tex


BENCHMARK_SHAPES = [
    (8192, 5120),
    (8192, 10240),
    (8192, 2560),
    (8192, 11328),
    (8192, 512),
    (8192, 3584),
    (5120, 8192),
    (10240, 8192),
    (2560, 8192),
    (11328, 8192),
    (512, 8192),
    (3584, 8192),
    (4096, 16384),
    (14336, 16384),
]
PROFILE_SHAPES = [(16384, 6144)]


# Nsight Compute profiling command:
# ncu -f -o nvfp4_4over6 --set=full --profile-from-start off --target-processes all \
#   --kernel-name "quantize_4over6_kernel" \
#   python3 benchmarks/benchmark_4over6.py --profile --profile-repeats 10


def make_quantizer(use_2d_quantization: bool, use_4over6: bool, err_mode: str) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=use_2d_quantization,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=448,
        nvfp4_4over6_err_mode=err_mode,
        with_random_sign_mask=True,
    )


def set_err_fast_math(enabled: bool) -> None:
    os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = "1" if enabled else "0"


def benchmark_quantize(
    shape: tuple[int, int],
    use_2d_quantization: bool,
    use_4over6: bool,
    err_mode: str,
    err_fast_math: bool,
    warmup: int,
    iters: int,
) -> float:
    set_err_fast_math(err_fast_math)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = make_quantizer(use_2d_quantization, use_4over6, err_mode)
    out = quantizer.make_empty(shape, dtype=x.dtype, device=x.device, requires_grad=False)

    for _ in range(warmup):
        quantizer.update_quantized(x, out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        quantizer.update_quantized(x, out)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def iter_cases(shapes):
    for shape in shapes:
        for mode_name, use_2d_quantization in (("1d", False), ("2d", True)):
            yield shape, mode_name, "nvfp4", "MAE", False, use_2d_quantization, False
            for err_mode in ("MAE", "MSE"):
                for err_fast_math in (False, True):
                    yield (
                        shape,
                        mode_name,
                        "4over6",
                        err_mode,
                        err_fast_math,
                        use_2d_quantization,
                        True,
                    )


def prepare_profile_case(case):
    shape, mode_name, kernel, err_mode, err_fast_math, use_2d_quantization, use_4over6 = case
    set_err_fast_math(err_fast_math)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = make_quantizer(use_2d_quantization, use_4over6, err_mode)
    out = quantizer.make_empty(shape, dtype=x.dtype, device=x.device, requires_grad=False)
    quantizer.update_quantized(x, out)
    torch.cuda.synchronize()
    return {
        "shape": shape,
        "mode_name": mode_name,
        "kernel": kernel,
        "err_mode": err_mode,
        "err_fast_math": err_fast_math,
        "quantizer": quantizer,
        "x": x,
        "out": out,
    }


def run_profile(profile_repeats: int) -> None:
    cases = [prepare_profile_case(case) for case in iter_cases(PROFILE_SHAPES)]
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for case in cases:
        set_err_fast_math(case["err_fast_math"])
        label = (
            f"shape={case['shape']} mode={case['mode_name']} kernel={case['kernel']} "
            f"err={case['err_mode']} err_fast={case['err_fast_math']}"
        )
        print(f"PROFILE {label}", flush=True)
        torch.cuda.nvtx.range_push(label)
        for _ in range(profile_repeats):
            case["quantizer"].update_quantized(case["x"], case["out"])
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def run_benchmark(shapes, warmup: int, iters: int) -> None:
    rows = []
    for shape in shapes:
        for mode_name, use_2d_quantization in (("1d", False), ("2d", True)):
            baseline_us = benchmark_quantize(
                shape=shape,
                use_2d_quantization=use_2d_quantization,
                use_4over6=False,
                err_mode="MAE",
                err_fast_math=False,
                warmup=warmup,
                iters=iters,
            )
            rows.append((shape, mode_name, "nvfp4", "-", baseline_us, 1.0, None, None))

            for err_mode in ("MAE", "MSE"):
                strict_us = benchmark_quantize(
                    shape=shape,
                    use_2d_quantization=use_2d_quantization,
                    use_4over6=True,
                    err_mode=err_mode,
                    err_fast_math=False,
                    warmup=warmup,
                    iters=iters,
                )
                fast_us = benchmark_quantize(
                    shape=shape,
                    use_2d_quantization=use_2d_quantization,
                    use_4over6=True,
                    err_mode=err_mode,
                    err_fast_math=True,
                    warmup=warmup,
                    iters=iters,
                )
                rows.append(
                    (
                        shape,
                        mode_name,
                        "4over6",
                        err_mode,
                        strict_us,
                        strict_us / baseline_us,
                        fast_us,
                        fast_us / baseline_us,
                    )
                )

    print(
        f"{'shape':>18}  {'mode':>4}  {'kernel':>7}  {'err':>3}  "
        f"{'strict_us':>10}  {'strict':>8}  {'fast_us':>10}  {'fast':>8}"
    )
    for (
        shape,
        mode_name,
        kernel,
        err_mode,
        strict_us,
        strict_slowdown,
        fast_us,
        fast_slowdown,
    ) in rows:
        fast_us_str = "-" if fast_us is None else f"{fast_us:10.3f}"
        fast_slowdown_str = "-" if fast_slowdown is None else f"{fast_slowdown:8.3f}x"
        print(
            f"{str(shape):>18}  {mode_name:>4}  {kernel:>7}  {err_mode:>3}  "
            f"{strict_us:10.3f}  {strict_slowdown:8.3f}x  "
            f"{fast_us_str:>10}  {fast_slowdown_str:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable Nsight Compute profile mode")
    parser.add_argument("--profile-repeats", default=1, type=int)
    parser.add_argument("--shapes", choices=("profile", "all"), default="profile")
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--iters", default=1000, type=int)
    args = parser.parse_args()

    supported, reason = check_fp8_block_scaling_support()
    assert supported, reason
    shapes = PROFILE_SHAPES if args.shapes == "profile" else BENCHMARK_SHAPES
    if args.profile:
        run_profile(args.profile_repeats)
    else:
        run_benchmark(shapes, args.warmup, args.iters)


if __name__ == "__main__":
    main()
