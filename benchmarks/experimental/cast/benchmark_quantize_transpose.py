import argparse
import logging
import os
import pathlib
import sys

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
import transformer_engine_torch as tex


def run_kernel(
    shape,
    is_1d: bool,
    return_transpose: bool,
    input_dtype=torch.bfloat16,
    quant_dtype=tex.DType.kFloat8E4M3,
):
    # Generate random input data
    M, K = shape
    src = torch.randn([M, K], dtype=input_dtype, device="cuda")

    quantizer = Float8BlockQuantizer(
        fp8_dtype=quant_dtype,
        rowwise=True,
        columnwise=return_transpose,
        block_scaling_dim=1 if is_1d else 2,
    )
    dst = quantizer.make_empty(shape, dtype=input_dtype, device="cuda")

    kernel_func = tex.quantize
    stmt = "kernel_func(src, quantizer, dst)"
    globals_dict = {
        "kernel_func": kernel_func,
        "quantizer": quantizer,
        "src": src,
        "dst": dst,
    }
    measurement = benchmark.Timer(
        stmt=stmt,
        globals=globals_dict,
        num_threads=1,
        setup="",
    ).adaptive_autorange(threshold=0.1, min_run_time=1.0, max_run_time=5.0)
    logging.info(f"Measurement: {measurement}")
    timing_us = measurement.median * 1e6
    return timing_us


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_output/",
        help="output path for report",
    )
    args = parser.parse_args()

    shapes = [
        (256, 1024),
        # (256, 1020),
        # 8B model shape
        (4096, 3072),
        (4096, 4096),
        (4096, 5440),
        # 15B model shape
        (16384, 1024),
        (16384, 3072),
        (16384, 6144),
        (16384, 12288),
        (16384, 24576),
    ]

    dim_1d_opts = [True, False]
    return_transpose_opts = [True, False]

    data = []
    for dim_1d_opt in dim_1d_opts:
        for return_transpose in return_transpose_opts:
            for shape in shapes:
                print(f"Running 1D={dim_1d_opt} with shape {shape}")
                timing_us = run_kernel(shape, dim_1d_opt, return_transpose)
                data.append([dim_1d_opt, return_transpose, shape, timing_us])

    df = pd.DataFrame(data=data, columns=["is_1d_kernel", "return_transpose", "shape", "timing_us"])
    logging.info(df)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report_file = pathlib.Path(args.output_dir) / f"{pathlib.Path(__file__).stem}_report.csv"
    df.to_csv(report_file, index=False)
    print(df)
    logging.info(f"Report saved to {report_file}")
