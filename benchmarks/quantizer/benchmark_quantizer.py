import argparse
import pathlib
import sys

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Quantizer,
    Float8Tensor,
    Float8CurrentScalingQuantizer,
)

fp8_dtype_te = tex.DType.kFloat8E4M3

def run_kernel(quantizer, shape, dtype):
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn(shape, dtype=dtype, device=device)

    stmt = "quantizer(x)"
    globals_dict = {
        "quantizer": quantizer,
        "x": x,
    }

    measurement = benchmark.Timer(
        stmt=stmt,
        globals=globals_dict,
        num_threads=1,
        setup="",
    ).timeit(10000)
    timing_us = measurement.median * 1e6
    return timing_us


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_output/",
        help="output path for report",
    )
    args = parser.parse_args()

    if args.profile:
        print("Profiling is enabled.")
    else:
        print("Profiling is disabled.")

    # define all the quantizers to benchmark
    quantizers = [
        Float8CurrentScalingQuantizer(fp8_dtype_te, device="cuda", rowwise=True, columnwise=True)
    ]

    if args.profile:
        shapes = [(4 * 4096, 8192)]
        dtypes = [torch.float32]
    else:
        # llama2 70B typical GEMM tensor shapes https://github.com/pytorch/ao/blob/77ca57d16844d1473825cc574b6cf1b56d4f6fff/benchmarks/float8/utils.py#L159
        shapes = [
            (1024, 8192),
            (3584, 8192),
            (8192, 1280),
            (8192, 7168),
            (4 * 4096, 1024),
            (4 * 4096, 3584),
            (4 * 4096, 8192),
        ]
        dtypes = [torch.float32, torch.bfloat16]

    data = []
    for quantizer in quantizers:
        quantizer_class_name = quantizer.__class__.__name__
        for dtype in dtypes:
            for shape in shapes:
                print(f"Running {quantizer} with dtype {dtype}  shape {shape} and rowwise {quantizer.rowwise_usage} and columnwise {quantizer.columnwise_usage}")
                timing_us = run_kernel(quantizer, shape, dtype)
                data.append([quantizer_class_name, shape, dtype, timing_us])

    df = pd.DataFrame(data=data, columns=["kernel", "shape", "dtype", "timing_us"])
    print(df)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report_file = (
        pathlib.Path(args.output_dir) / f"{pathlib.Path(__file__).stem}_report.csv"
    )
    # df.to_csv(report_file, index=False)
    print(f"Report saved to {report_file}")
