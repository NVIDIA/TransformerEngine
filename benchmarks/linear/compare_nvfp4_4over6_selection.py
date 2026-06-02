# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare NVFP4 4over6 E4M3 scales with and without error fast math."""

import os
from contextlib import contextmanager

import torch
from transformer_engine.pytorch import NVFP4Quantizer
import transformer_engine_torch as tex


M, K = 98304, 7168


@contextmanager
def _error_fast_math(enabled: bool):
    old_value = os.environ.get("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH")
    os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", None)
        else:
            os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = old_value


def _quantize_scale_bytes(
    x: torch.Tensor,
    err_mode: str,
    err_fast_math: bool,
    row_scaled: bool,
    with_2d_quantization: bool,
    nvfp4_e4m3_max: int,
) -> torch.Tensor:
    quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=with_2d_quantization,
        row_scaled_nvfp4=row_scaled,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=err_mode,
    )
    with _error_fast_math(err_fast_math):
        quantized = quantizer(x)
    assert quantized._rowwise_scale_inv is not None
    return quantized._rowwise_scale_inv.contiguous().view(torch.uint8)


def _compare_e4m3(
    x: torch.Tensor,
    dtype_name: str,
    scale_mode: str,
    row_scaled: bool,
    quant_mode: str,
    with_2d_quantization: bool,
    nvfp4_e4m3_max: int,
) -> None:
    for err_mode in ("MAE", "MSE"):
        regular = _quantize_scale_bytes(
            x, err_mode, False, row_scaled, with_2d_quantization, nvfp4_e4m3_max
        )
        fast = _quantize_scale_bytes(
            x, err_mode, True, row_scaled, with_2d_quantization, nvfp4_e4m3_max
        )
        same = torch.count_nonzero(regular == fast).item()
        total = regular.numel()
        print(
            f"{scale_mode:>6} {quant_mode:>5} {nvfp4_e4m3_max:8d} "
            f"{dtype_name:>5} {err_mode:>3} "
            f"{100.0 * same / total:12.6f} {total - same:15d} {total}"
        )


def main():
    torch.set_grad_enabled(False)
    print(f"shape=({M}, {K}), 1d_e4m3_values={M * K // 16}")
    print("scale quant e4m3_max dtype mode same_e4m3_pct different_e4m3 total_e4m3")
    for scale_mode, row_scaled, quant_mode, with_2d_quantization in (
        ("tensor", False, "1d", False),
        ("tensor", False, "2d", True),
        ("row", True, "1d", False),
    ):
        for nvfp4_e4m3_max in (256, 448):
            for dtype, dtype_name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
                torch.manual_seed(1234)
                x = torch.randn((M, K), dtype=dtype, device="cuda")
                _compare_e4m3(
                    x,
                    dtype_name,
                    scale_mode,
                    row_scaled,
                    quant_mode,
                    with_2d_quantization,
                    nvfp4_e4m3_max,
                )
                del x
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
