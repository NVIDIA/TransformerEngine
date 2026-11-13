# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kernel-only benchmark of the gated-activation MXFP8 kernels (CUPTI kernel timing, cold L2):
the forward act(x_act) * x_gate quantized to MXFP8, and its backward.

Both implementations run through TE's dispatch: --impl CUDA forces its CUDA kernels; --impl
CuTeDSL turns the CuTeDSL backend on, which runs a CuTeDSL kernel where the TE build has one
and CUDA otherwise; the kernel column says which ran. The forward takes an M x 2N input and
writes M x N; the backward takes the M x N incoming gradient and the M x 2N forward input and
writes M x 2N. Writes a CSV.

examples:
  python benchmark_gated_mxfp8.py
  python benchmark_gated_mxfp8.py --activation "swiglu;clamped_swiglu" --pass bwd --impl CUDA
  python benchmark_gated_mxfp8.py --compare --old CUDA --format html
"""

import functools
import itertools

import common

# N is the FFN (or expert) intermediate size: DeepSeek-V3 / Kimi K2 experts (2048), gpt-oss
# (2880, also the non-128-aligned case), Qwen3-32B (25600 / 4 TP = 6400), Llama 3 70B
# (28672 / 4 TP = 7168).
DEFAULT_N = [2048, 2880, 6400, 7168]
DEFAULT_M = [1024, 2048, 4096, 8192, 16384, 32768]
DEBUG_SHAPES = list(itertools.product([1024, 4096], [2048, 2880]))
# Gated activations TE fuses with MXFP8 quantization; their parameters keep TE's defaults.
ACTIVATIONS = (
    "swiglu",
    "geglu",
    "reglu",
    "qgeglu",
    "sreglu",
    "glu",
    "situglu",
    "clamped_swiglu",
)
PASSES = ("fwd", "bwd")
CONFIG = ["M", "N", "pass", "activation", "direction", "swizzle", "input"]


def _backward(activation):
    return "clamped_dswiglu" if activation == "clamped_swiglu" else f"d{activation}"


def run(args, impl):
    """Times the `impl` gated kernel for every config."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch
    from transformer_engine.pytorch import MXFP8Quantizer

    tex = common.load_tex_backend(impl)

    def factory(m, n, step, activation, direction, swizzle, input_dtype):
        dtype = getattr(torch, common.INPUTS[input_dtype][0])
        rowwise, colwise = common.DIRECTIONS[direction]
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=colwise
        )
        quantizer.optimize_for_gemm = swizzle == "on"
        x = torch.randn((m, 2 * n), dtype=dtype, device="cuda")
        if step == "fwd":
            op = getattr(tex, activation)
            return lambda: op(x, quantizer)
        grad = torch.randn((m, n), dtype=dtype, device="cuda")
        op = getattr(tex, _backward(activation))
        return lambda: op(grad, x, quantizer)

    def nbytes(m, n, step, direction, input_dtype):
        size = common.INPUTS[input_dtype][2]
        if step == "fwd":
            return 2 * m * n * size + common.mxfp8_data_bytes(m * n, direction)
        return 3 * m * n * size + common.mxfp8_data_bytes(2 * m * n, direction)

    configs = [
        (m, n, step, act, d, sw, inp)
        for (m, n) in args.shape_list
        for inp in args.inputs
        for step in args.passes
        for act in args.activations
        for d in args.direction
        for sw in args.swizzles
    ]
    factories = (
        (nbytes(m, n, step, d, inp), functools.partial(factory, m, n, step, act, d, sw, inp))
        for m, n, step, act, d, sw, inp in configs
    )
    timed = common.time_kernels(factories, args.runs, args.warmup, common.te_kernels("gated"))
    rows = []
    for config, result in zip(configs, timed):
        if result is None:
            continue
        times, kernel, _ = result
        m, n, step, _, d, _, inp = config
        row = dict(zip(CONFIG, config))
        row.update(common.result_row(times, kernel))
        row["GB/s"] = nbytes(m, n, step, d, inp) / (row["time_us"] * 1e3)
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFIG + common.RESULT_COLUMNS)


def add_arguments(parser):
    """The gated options."""
    common.add_suite_arguments(parser)
    parser.add_argument(
        "--shapes", metavar='"MxN;..."', help="M x output N (input M x 2N), multiples of 32"
    )
    parser.add_argument(
        "--pass",
        dest="pass_",
        default="fwd;bwd",
        metavar="fwd;bwd",
        help="forward and/or backward (default: fwd;bwd)",
    )
    parser.add_argument(
        "--activation",
        default="swiglu",
        metavar="all|A;A;...",
        help=f"all, or a ;-separated subset of {';'.join(ACTIVATIONS)} (default: swiglu)",
    )
    parser.add_argument(
        "--direction", nargs="+", default=list(common.DIRECTIONS), choices=list(common.DIRECTIONS)
    )
    common.add_swizzle_argument(parser)
    common.add_dtype_argument(parser)


def parse_arguments(parser, args):
    """Validates the gated options."""
    common.parse_suite_arguments(parser, args)
    args.passes = common.parse_list(parser, args.pass_, PASSES, "--pass")
    args.activations = common.parse_list(parser, args.activation, ACTIVATIONS, "--activation")
    args.swizzles = list(common.SWIZZLES) if args.swizzle == "both" else [args.swizzle]
    args.inputs = common.parse_list(parser, args.input, list(common.INPUTS), "--input")
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
    """The gated options for a compare worker."""
    return common.suite_worker_arguments(args) + [
        "--pass",
        ";".join(args.passes),
        "--activation",
        ";".join(args.activations),
        "--direction",
        *args.direction,
        "--swizzle",
        args.swizzle,
        "--input",
        ";".join(args.inputs),
    ]


def describe(args):
    """config.txt lines for the benchmarked suite."""
    return [
        f"mode:        {args.mode}",
        f"shapes:      {'; '.join(f'{m}x{n}' for m, n in args.shape_list)} (M x output N)",
        f"passes:      {', '.join(args.passes)}",
        f"activations: {', '.join(args.activations)}",
        f"directions:  {', '.join(args.direction)}",
        f"swizzle:     {', '.join(args.swizzles)}",
        f"input:       {', '.join(args.inputs)}",
    ]


def heading(values, _activation):
    """`bf16 · swiglu fwd · both · swizzled scale` for a chart."""
    parts = [values["input"], f"{values['activation']} {values['pass']}", values["direction"]]
    if values["swizzle"] == "on":
        parts.append("swizzled scale")
    return " · ".join(parts)


BENCHMARK = common.Benchmark(
    title="Gated MXFP8 kernel benchmark",
    config=CONFIG,
    selectors={
        "input": "Input",
        "activation": "Activation",
        "pass": "Pass",
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
    heading=heading,
)

if __name__ == "__main__":
    common.main(BENCHMARK, __file__, __doc__)
