# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kernel-only benchmark of the MXFP8 dequantize kernels (CUPTI kernel timing, cold L2).

Both implementations run through TE's dispatch: --impl CUDA forces its CUDA kernels; --impl
CuTeDSL turns the CuTeDSL backend on, which runs a CuTeDSL kernel where the TE build has one
and CUDA otherwise; the kernel column says which ran. --direction picks which data the MXFP8
tensor holds (rowwise or columnwise); its quantization is set up outside the timing. Writes a
CSV.

examples:
  python benchmark_dequantize_mxfp8.py
  python benchmark_dequantize_mxfp8.py --direction colwise --output "bf16;fp32" --impl CUDA
  python benchmark_dequantize_mxfp8.py --compare --old CUDA --format html
"""

import functools
import itertools

import common

DEFAULT_N = [2880, 4096, 5120, 7168, 8192]
DEFAULT_M = [1024, 2048, 4096, 8192, 16384, 32768]
DEBUG_SHAPES = list(itertools.product([1024, 4096], [2880, 4096]))
DIRECTIONS = ("rowwise", "colwise")  # dequantize reads one of them
CONFIG = ["M", "N", "direction", "swizzle", "output"]
TE_DTYPES = {"bf16": "kBFloat16", "fp16": "kFloat16", "fp32": "kFloat32"}


def run(args, impl):
    """Times the `impl` dequantize kernel for every config."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch
    from transformer_engine.pytorch import MXFP8Quantizer

    tex = common.load_tex_backend(impl)

    def factory(m, n, direction, swizzle, output):
        rowwise, colwise = common.DIRECTIONS[direction]
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=colwise
        )
        quantizer.optimize_for_gemm = swizzle == "on"
        x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
        quantized = tex.quantize(x, quantizer, None, None)
        otype = getattr(tex.DType, TE_DTYPES[output])
        return lambda: tex.dequantize(quantized, otype)

    def nbytes(m, n, direction, output):
        return common.mxfp8_data_bytes(m * n, direction) + m * n * common.INPUTS[output][2]

    configs = [
        (m, n, d, sw, out)
        for (m, n) in args.shape_list
        for out in args.outputs
        for d in args.direction
        for sw in args.swizzles
    ]
    factories = (
        (nbytes(m, n, d, out), functools.partial(factory, m, n, d, sw, out))
        for m, n, d, sw, out in configs
    )
    timed = common.time_kernels(factories, args.runs, args.warmup, common.te_kernels("dequantize"))
    rows = []
    for config, result in zip(configs, timed):
        if result is None:
            continue
        times, kernel, _ = result
        m, n, d, _, out = config
        row = dict(zip(CONFIG, config))
        row.update(common.result_row(times, kernel))
        row["GB/s"] = nbytes(m, n, d, out) / (row["time_us"] * 1e3)
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFIG + common.RESULT_COLUMNS)


def add_arguments(parser):
    """The dequantize options."""
    common.add_suite_arguments(parser)
    parser.add_argument("--shapes", metavar='"MxN;..."', help="M, N multiples of 32")
    parser.add_argument(
        "--direction",
        nargs="+",
        default=list(DIRECTIONS),
        choices=list(DIRECTIONS),
        help="which MXFP8 data is dequantized (default: both, one at a time)",
    )
    common.add_swizzle_argument(parser)
    common.add_dtype_argument(parser, "--output", "output")


def parse_arguments(parser, args):
    """Validates the dequantize options."""
    common.parse_suite_arguments(parser, args)
    args.swizzles = list(common.SWIZZLES) if args.swizzle == "both" else [args.swizzle]
    args.outputs = common.parse_list(parser, args.output, list(common.INPUTS), "--output")
    try:
        if args.mode == "manual":
            args.shape_list = common.parse_shapes(args.shapes)
        elif args.mode == "debug":
            args.shape_list = DEBUG_SHAPES
        else:
            args.shape_list = list(itertools.product(DEFAULT_M, DEFAULT_N))
    except ValueError as e:
        parser.error(str(e))


def worker_arguments(args):
    """The dequantize options for a compare worker."""
    return common.suite_worker_arguments(args) + [
        "--direction",
        *args.direction,
        "--swizzle",
        args.swizzle,
        "--output",
        ";".join(args.outputs),
    ]


def describe(args):
    """config.txt lines for the benchmarked suite."""
    return [
        f"mode:       {args.mode}",
        f"shapes:     {'; '.join(f'{m}x{n}' for m, n in args.shape_list)}",
        f"directions: {', '.join(args.direction)}",
        f"swizzle:    {', '.join(args.swizzles)}",
        f"output:     {', '.join(args.outputs)}",
    ]


def heading(values, _activation):
    """`colwise → bf16 · swizzled scale` for a chart."""
    parts = [f"{values['direction']} → {values['output']}"]
    if values["swizzle"] == "on":
        parts.append("swizzled scale")
    return " · ".join(parts)


BENCHMARK = common.Benchmark(
    title="MXFP8 dequantize kernel benchmark",
    config=CONFIG,
    selectors={"output": "Output", "direction": "Direction", "swizzle": "Swizzled Scale"},
    x="M",
    split="N",
    add_arguments=add_arguments,
    parse_arguments=parse_arguments,
    worker_arguments=worker_arguments,
    run=run,
    describe=describe,
    heading=heading,
)

if __name__ == "__main__":
    common.main(BENCHMARK, __file__, __doc__)
