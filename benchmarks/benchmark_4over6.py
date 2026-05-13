# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import os

import torch
import torch.utils.benchmark as benchmark
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


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
PROFILE_SHAPES = [
    (16384, 6144),
]
MIN_RUN_TIME = 5


# Nsight Compute profiling command:
# ncu -f -o nvfp4_4over6 --set=full --profile-from-start off --target-processes all \
#   python3 benchmarks/benchmark_4over6.py --profile


def make_quantizer(use_2d_quantization, use_4over6, err_mode):
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
        use_4over6=use_4over6,
        nvfp4_e4m3_max=448,
        nvfp4_4over6_err_mode=err_mode,
        with_random_sign_mask=True,
    )


def set_err_fast_math(enabled):
    os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = "1" if enabled else "0"


def benchmark_quantize(shape, use_2d_quantization, use_4over6, err_mode, err_fast_math):
    set_err_fast_math(err_fast_math)

    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = make_quantizer(use_2d_quantization, use_4over6, err_mode)
    out = quantizer.make_empty(shape, dtype=x.dtype, device=x.device, requires_grad=False)
    quantizer.update_quantized(x, out)
    torch.cuda.synchronize()

    timing = benchmark.Timer(
        stmt="quantizer.update_quantized(x, out)",
        globals={"quantizer": quantizer, "x": x, "out": out},
        num_threads=1,
    ).blocked_autorange(min_run_time=MIN_RUN_TIME)
    return timing.median * 1e6


def iter_profile_cases():
    for shape in PROFILE_SHAPES:
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


def run_profile(profile_repeats):
    cases = [prepare_profile_case(case) for case in iter_profile_cases()]
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for case in cases:
        set_err_fast_math(case["err_fast_math"])
        print(
            "PROFILE "
            f"shape={case['shape']} mode={case['mode_name']} kernel={case['kernel']} "
            f"err={case['err_mode']} err_fast={case['err_fast_math']}",
            flush=True,
        )
        for _ in range(profile_repeats):
            case["quantizer"].update_quantized(case["x"], case["out"])
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def run_benchmark():
    rows = []
    for shape in BENCHMARK_SHAPES:
        for mode_name, use_2d_quantization in (("1d", False), ("2d", True)):
            baseline_us = benchmark_quantize(
                shape=shape,
                use_2d_quantization=use_2d_quantization,
                use_4over6=False,
                err_mode="MAE",
                err_fast_math=False,
            )
            rows.append((shape, mode_name, "nvfp4", "-", baseline_us, 1.0, None, None))

            for err_mode in ("MAE", "MSE"):
                strict_timing_us = benchmark_quantize(
                    shape=shape,
                    use_2d_quantization=use_2d_quantization,
                    use_4over6=True,
                    err_mode=err_mode,
                    err_fast_math=False,
                )
                fast_timing_us = benchmark_quantize(
                    shape=shape,
                    use_2d_quantization=use_2d_quantization,
                    use_4over6=True,
                    err_mode=err_mode,
                    err_fast_math=True,
                )
                rows.append(
                    (
                        shape,
                        mode_name,
                        "4over6",
                        err_mode,
                        strict_timing_us,
                        strict_timing_us / baseline_us,
                        fast_timing_us,
                        fast_timing_us / baseline_us,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable Nsight Compute profile mode")
    parser.add_argument(
        "--profile-repeats",
        default=1,
        type=int,
        help="Number of profiled update_quantized calls per case",
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(args.profile_repeats)
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
