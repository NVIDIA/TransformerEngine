# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import torch
import pandas as pd
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
import transformer_engine.pytorch.cpp_extensions as ext

from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

scale_padding_to = 1
permute_scale = False

TORCH_TO_TE_FLOAT_MAP = {
    torch.bfloat16: tex.DType.kBFloat16,
}


def run_kernel(shape, stochastic_rounding: bool, input_dtype=torch.bfloat16):
    # Generate random input data
    M, K = shape
    x = torch.randn([M, K], dtype=input_dtype, device="cuda")

    assert shape[0] % 16 == 0, "Shape must be divisible by 16"
    assert shape[1] % 16 == 0, "Shape must be divisible by 16"

    # Quantize
    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=True,
        stochastic_rounding=stochastic_rounding,
    )
    x_nvfp4_sut = nvfp4_quantizer.make_empty(
        (M, K), dtype=x.dtype, device=x.device, requires_grad=False
    )
    x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    with torch.no_grad():
        stmt = "kernel_func(input, output)"
        globals_dict = {
            "kernel_func": nvfp4_quantizer.update_quantized,
            "input": x,
            "output": x_nvfp4_sut,
        }

        timing = benchmark.Timer(
            stmt=stmt,
            globals=globals_dict,
            num_threads=1,
        ).blocked_autorange(min_run_time=5)
    print(timing)
    timing_us = timing.median * 1e6

    input_nbytes = shape[0] * shape[1] * 2  # bf16
    output_nbytes = shape[0] * shape[1] // 2  # //2 for fp4
    sf_nbytes = shape[0] * shape[1] // 16  # //16 for 1 byte per 16 elems

    total_nbytes = (
        0
        + input_nbytes
        * 3  # Reading input for Amax(x)&Amax(RHT(x.T)), Reading input for Cast(x), Reaindg input for Cast(RHT(x.T))
        + 2 * 4  # Output 2 * float for scale & amax
        + 2 * 4  # Input 2 * float
        + output_nbytes * 2  # Output from Cast(x) and Cast(RHT(x.T))
        + sf_nbytes * 2  # Scale factor
    )

    throughput_GBps = total_nbytes / (1024 * 1024 * 1024) / (timing_us / 1e6)

    print(
        f"Stochastic rounding: {stochastic_rounding}, Total: {total_nbytes} bytes, Throughput:"
        f" {throughput_GBps} GB/s"
    )
    return timing_us, throughput_GBps


# Nsight Compute Profiling Command:
# ncu -f -o block_scaled_1d_cast_transpose_kernel --set=full --kernel-name "block_scaled_1d_cast_transpose_kernel" -s 5 -c 5 python benchmark_cast_transpose_1d_block.py --profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    args = parser.parse_args()

    if args.profile:
        print("Profiling is enabled.")
    else:
        print("Profiling is disabled.")

    shapes = [
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

    if args.profile:
        shapes = [
            (16384, 6144),
        ]

    data = []
    for stochastic_rounding in [True]:  # , False]:
        for shape in shapes:
            print(
                f"Running benchmark_func with shape {shape} and stochastic_rounding"
                f" {stochastic_rounding}"
            )
            timing_us, throughput_GBps = run_kernel(shape, stochastic_rounding)
            data.append(
                [
                    "benchmark_func",
                    shape,
                    stochastic_rounding,
                    timing_us,
                    throughput_GBps,
                ]
            )

    df = pd.DataFrame(
        data=data,
        columns=[
            "kernel",
            "shape",
            "stochastic_rounding",
            "timing_us",
            "throughput(GB/s)",
        ],
    )
    print(df)
    df.to_csv("benchmark_cast_nvfp4.csv", index=False)
