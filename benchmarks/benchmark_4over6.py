# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os

import torch
import torch.utils.benchmark as benchmark
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


SHAPES = [
    (16384, 6144),
]
MIN_RUN_TIME = 5


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


def main():
    rows = []
    for shape in SHAPES:
        for mode_name, use_2d_quantization in (("1d", False), ("2d", True)):
            baseline_us = benchmark_quantize(
                shape=shape,
                use_2d_quantization=use_2d_quantization,
                use_4over6=False,
                err_mode="MAE",
                err_fast_math=False,
            )
            rows.append((shape, mode_name, "nvfp4", "-", "-", baseline_us, 1.0))

            for err_mode in ("MAE", "MSE"):
                for err_fast_math in (False, True):
                    timing_us = benchmark_quantize(
                        shape=shape,
                        use_2d_quantization=use_2d_quantization,
                        use_4over6=True,
                        err_mode=err_mode,
                        err_fast_math=err_fast_math,
                    )
                    rows.append(
                        (
                            shape,
                            mode_name,
                            "4over6",
                            err_mode,
                            str(err_fast_math),
                            timing_us,
                            timing_us / baseline_us,
                        )
                    )

    print(
        f"{'shape':>18}  {'mode':>4}  {'kernel':>7}  {'err':>3}  "
        f"{'err_fast':>8}  {'time_us':>10}  {'slowdown':>8}"
    )
    for shape, mode_name, kernel, err_mode, err_fast_math, timing_us, slowdown in rows:
        print(
            f"{str(shape):>18}  {mode_name:>4}  {kernel:>7}  {err_mode:>3}  "
            f"{err_fast_math:>8}  {timing_us:10.3f}  {slowdown:8.3f}x"
        )


if __name__ == "__main__":
    main()
