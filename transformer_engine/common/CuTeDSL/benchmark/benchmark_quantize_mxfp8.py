# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kernel-only benchmark of the MXFP8 quantize kernels (CUPTI kernel timing, cold L2).

Both implementations run through TE's dispatch (common.te_backend): --impl CUDA forces its
CUDA kernels; --impl CuTeDSL turns the CuTeDSL backend on, which runs a CuTeDSL kernel where TE
has one and CUDA otherwise; the kernel column says which ran. Writes a CSV.

examples:
  python benchmark_quantize_mxfp8.py
  python benchmark_quantize_mxfp8.py --mode manual --shapes "4096x7168;16384x4096" --impl CUDA
  python benchmark_quantize_mxfp8.py --compare                          # main vs HEAD
  python benchmark_quantize_mxfp8.py --compare --old CUDA --mode manual --shapes "4096x7168"
  python benchmark_quantize_mxfp8.py --fusion "none;dact;dactbias" --format html
  python benchmark_quantize_mxfp8.py --scaling both --mode debug --format raw
"""

import functools
import itertools
import sys

import common

# Hidden sizes of current frontier models: gpt-oss (2880, also the non-128-aligned case),
# Qwen3-235B (4096), Llama 4 / Qwen3-32B / GLM-4.5 (5120), DeepSeek-V3 / Kimi K2 (7168),
# Llama 3 70B (8192).
DEFAULT_N = [2880, 4096, 5120, 7168, 8192]
DEFAULT_M = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_SHAPES = list(itertools.product(DEFAULT_M, DEFAULT_N))
# Smallest set that still exercises every output: two M (a line per chart), two N (one of
# them not 128-aligned).
DEBUG_SHAPES = list(itertools.product([1024, 4096], [2880, 4096]))
FUSIONS = ("none", "act", "dact", "dbias", "dactbias")
ACTIVATIONS = ("gelu", "relu", "silu", "qgelu", "srelu")
CONFIG = ["M", "N", "direction", "fusion", "scaling", "swizzle", "input"]


def _te_launcher(tex, activation):
    """Returns setup(x, act_input, direction, fusion, scaling, swizzle) -> launch(), running
    TE's quantize, which dispatches to whichever backend `tex` was imported with."""
    from transformer_engine.pytorch import MXFP8Quantizer  # pylint: disable=import-outside-toplevel

    def setup(x, act_input, direction, fusion, scaling, swizzle):
        rowwise, colwise = common.DIRECTIONS[direction]
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=rowwise,
            columnwise=colwise,
            with_2d_quantization=scaling == "2d",
        )
        quantizer.optimize_for_gemm = swizzle == "on"
        if fusion == "none":
            y = quantizer.make_empty(x.shape, dtype=x.dtype, device=x.device)
            return lambda: tex.quantize(x, quantizer, y, None)
        if fusion == "act":
            return lambda: getattr(tex, activation)(x, quantizer)
        if fusion == "dbias":
            return lambda: tex.bgrad_quantize(x, quantizer)
        op = getattr(tex, f"dbias_d{activation}" if fusion == "dactbias" else f"d{activation}")
        return lambda: op(x, act_input, quantizer)

    return setup


def run(args, impl):
    """Times the `impl` quantize kernel for every config."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch

    setup = _te_launcher(common.te_backend(impl), args.activation or "gelu")

    def factory(m, n, input_dtype, direction, fusion, scaling, swizzle):
        dtype = getattr(torch, common.INPUTS[input_dtype][0])
        x = torch.randn((m, n), dtype=dtype, device="cuda")
        # The forward activation input, for dact only.
        act_input = torch.randn((m, n), dtype=dtype, device="cuda") if "dact" in fusion else None
        return setup(x, act_input, direction, fusion, scaling, swizzle)

    configs = [
        (m, n, d, f, sc, sw, inp)
        for (m, n) in args.shape_list
        for inp in args.inputs
        for d in args.direction
        for f in args.fusions
        for sc in args.scalings
        for sw in args.swizzles
        if sc == "1d" or f == "none"  # TE only fuses 2D scaling into a plain cast
    ]
    if not configs:
        sys.exit("No config to run: 2D scaling only applies to --fusion none")
    factories = (
        (
            common.mxfp8_bytes(m * n, inp, d, 2 if "dact" in f else 1),
            functools.partial(factory, m, n, inp, d, f, sc, sw),
        )
        for m, n, d, f, sc, sw, inp in configs
    )
    # Helper kernels such as scale-padding zeroing or the dbias reduction are not timed.
    timed = common.time_kernels(factories, args.runs, args.warmup, common.te_kernels("^quantize"))
    rows = []
    for config, result in zip(configs, timed):
        if result is None:
            continue
        times, kernel, _ = result
        m, n, d, f, _, _, inp = config
        row = dict(zip(["M", "N", "direction", "fusion", "scaling", "swizzle", "input"], config))
        row.update(common.result_row(times, kernel))
        num_inputs = 2 if "dact" in f else 1
        row["GB/s"] = common.mxfp8_bytes(m * n, inp, d, num_inputs) / (row["time_us"] * 1e3)
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFIG + common.RESULT_COLUMNS)


def add_arguments(parser):
    """The single-tensor options."""
    common.add_quantize_arguments(parser, FUSIONS)
    parser.add_argument("--shapes", metavar='"MxN;..."', help="M, N multiples of 32")
    parser.add_argument(
        "--activation", choices=ACTIVATIONS, help="only with act/dact/dactbias (default: gelu)"
    )


def parse_arguments(parser, args):
    """Validates the single-tensor options."""
    common.parse_quantize_arguments(parser, args, FUSIONS)
    try:
        if args.mode == "manual":
            args.shape_list = common.parse_shapes(args.shapes)
        else:
            args.shape_list = DEBUG_SHAPES if args.mode == "debug" else DEFAULT_SHAPES
    except ValueError as e:
        parser.error(str(e))
    if args.activation and not {"act", "dact", "dactbias"} & set(args.fusions):
        parser.error("--activation only applies to the act, dact and dactbias fusions")


def worker_arguments(args):
    """The single-tensor options for a compare worker."""
    worker = common.quantize_worker_arguments(args)
    return worker + (["--activation", args.activation] if args.activation else [])


def describe(args):
    """config.txt lines for the benchmarked suite."""
    if args.mode != "default":
        shapes = f"{args.mode}: " + "; ".join(f"{m}x{n}" for m, n in args.shape_list)
    else:
        shapes = (
            f"default suite, M = {', '.join(map(str, DEFAULT_M))}"
            f" x N = {', '.join(map(str, DEFAULT_N))} ({len(DEFAULT_SHAPES)} shapes)"
        )
    lines = [
        f"mode:       {args.mode}",
        f"shapes:     {shapes}",
        f"directions: {', '.join(args.direction)}",
        f"fusions:    {', '.join(args.fusions)}",
        f"scaling:    {', '.join(args.scalings)}",
        f"swizzle:    {', '.join(args.swizzles)}",
        f"input:      {', '.join(args.inputs)}",
    ]
    if {"act", "dact", "dactbias"} & set(args.fusions):
        lines.append(f"activation: {args.activation or 'gelu'}")
    return lines


BENCHMARK = common.Benchmark(
    title="MXFP8 quantize kernel benchmark",
    config=CONFIG,
    selectors={
        "input": "Input",
        "fusion": "Fusion",
        "scaling": "Scaling",
        "direction": "Direction",
        "swizzle": "Swizzled Scale",
    },
    x="M",
    split="N",
    add_arguments=add_arguments,
    parse_arguments=parse_arguments,
    worker_arguments=worker_arguments,
    run=run,
    describe=describe,
    heading=common.quantize_heading,
)

if __name__ == "__main__":
    common.main(BENCHMARK, __file__, __doc__)
