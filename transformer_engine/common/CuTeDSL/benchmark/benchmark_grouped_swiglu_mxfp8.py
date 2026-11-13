# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kernel-only benchmark of the grouped scaled SwiGLU MXFP8 kernels (CUPTI kernel timing, cold
L2): the MoE FC2-input recompute (silu(x_act) * x_gate) * prob, quantized columnwise.

A group is G experts in a buffer of M tokens; the input is M x 2N and the output M x N. As
with CUDA graphs, varying_first experts fill only part of the buffer (see
benchmark_grouped_quantize_mxfp8.py). Both implementations run through TE's dispatch: --impl
CUDA forces its CUDA kernels; --impl CuTeDSL turns the CuTeDSL backend on, which runs a
CuTeDSL kernel where the TE build has one and CUDA otherwise; the kernel column says which
ran. Writes a CSV.

examples:
  python benchmark_grouped_swiglu_mxfp8.py
  python benchmark_grouped_swiglu_mxfp8.py --activation "swiglu;clamped_swiglu" --impl CUDA
  python benchmark_grouped_swiglu_mxfp8.py --compare --old CUDA --format html
"""

import functools
import itertools
import sys

import common
import benchmark_grouped_quantize_mxfp8 as grouped

# N is the expert intermediate size: Qwen3-235B (1536), DeepSeek-V3 / Kimi K2 (2048), gpt-oss
# (2880, not 128-aligned).
DEFAULT_N = [1536, 2048, 2880]
DEBUG_N = [2048, 2880]
LAYOUTS = ("same", "varying_first")  # the kernel takes per-expert token counts only
ACTIVATIONS = ("swiglu", "clamped_swiglu")  # clamped: gpt-oss, TE's default limit 7.0
CLAMP_LIMIT = 7.0
CONFIG = ["M", "N", "experts", "layout", "activation", "swizzle", "input"]


def run(args, impl):
    """Times the `impl` grouped scaled SwiGLU kernel for every config."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch

    tex = common.load_tex_backend(impl)

    def factory(m, n, g, layout, activation, swizzle, input_dtype, group):
        dtype = getattr(torch, common.INPUTS[input_dtype][0])
        quantizer = grouped._quantizer(tex, "colwise", swizzle)  # pylint: disable=protected-access
        x = torch.randn((m, 2 * n), dtype=dtype, device="cuda")
        prob = torch.rand(m, dtype=dtype, device="cuda")
        first, _ = grouped._dims_tensors(group, layout)  # pylint: disable=protected-access
        if activation == "clamped_swiglu":
            return lambda: tex.group_scaled_clamped_swiglu(
                x, prob, quantizer, g, CLAMP_LIMIT, first_dims=first
            )
        return lambda: tex.group_scaled_swiglu(x, prob, quantizer, g, first_dims=first)

    def nbytes(input_dtype, group):
        tokens = sum(a for a, _ in group)
        n = group[0][1]
        size = common.INPUTS[input_dtype][2]
        return tokens * (2 * n + 1) * size + common.mxfp8_data_bytes(tokens * n, "colwise")

    configs = [
        (m, n, g, lay, act, sw, inp, grouped.members(m, n, g, lay))
        for (m, n), g, lay, inp, act, sw in itertools.product(
            args.shape_list,
            args.experts_list,
            args.layouts,
            args.inputs,
            args.activations,
            args.swizzles,
        )
        if grouped.valid_group(m, g, lay)
    ]
    configs = grouped.without_empty_groups(configs)
    if not configs:
        sys.exit("No config to run: check --shapes/--experts/--layout")
    factories = ((nbytes(c[6], c[7]), functools.partial(factory, *c)) for c in configs)
    keep = common.te_kernels("^group_scaled")
    timed = common.time_kernels(factories, args.runs, args.warmup, keep)
    rows = []
    for config, result in zip(configs, timed):
        if result is None:
            continue
        times, kernel, _ = result
        *values, group = config
        row = dict(zip(CONFIG, values))
        row.update(common.result_row(times, kernel))
        row["GB/s"] = nbytes(row["input"], group) / (row["time_us"] * 1e3)
        row["members"] = ";".join(f"{a}x{b}" for a, b in group)
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFIG + common.RESULT_COLUMNS + ["members"])


def add_arguments(parser):
    """The grouped SwiGLU options."""
    common.add_suite_arguments(parser)
    grouped.add_group_arguments(parser, LAYOUTS)
    parser.add_argument(
        "--activation",
        default="swiglu",
        metavar="all|A;A;...",
        help=f"all, or a ;-separated subset of {';'.join(ACTIVATIONS)} (default: swiglu)",
    )
    common.add_swizzle_argument(parser)
    common.add_dtype_argument(parser)


def parse_arguments(parser, args):
    """Validates the grouped SwiGLU options."""
    common.parse_suite_arguments(parser, args)
    grouped.parse_group_arguments(parser, args, LAYOUTS, DEFAULT_N, DEBUG_N)
    args.activations = common.parse_list(parser, args.activation, ACTIVATIONS, "--activation")
    args.swizzles = list(common.SWIZZLES) if args.swizzle == "both" else [args.swizzle]
    args.inputs = common.parse_list(parser, args.input, list(common.INPUTS), "--input")


def worker_arguments(args):
    """The grouped SwiGLU options for a compare worker."""
    return (
        common.suite_worker_arguments(args)
        + grouped.group_worker_arguments(args)
        + ["--activation", ";".join(args.activations), "--swizzle", args.swizzle]
        + ["--input", ";".join(args.inputs)]
    )


def describe(args):
    """config.txt lines for the benchmarked suite."""
    return grouped.describe_group(args) + [
        "direction:   colwise (the kernel's only output)",
        f"activations: {', '.join(args.activations)}",
        f"swizzle:     {', '.join(args.swizzles)}",
        f"input:       {', '.join(args.inputs)}",
    ]


def heading(values, _activation):
    """`bf16 · varying_first · G=8 · swiglu` for a chart."""
    parts = [values["input"], values["layout"], f"G={values['experts']}", values["activation"]]
    if values["swizzle"] == "on":
        parts.append("swizzled scale")
    return " · ".join(parts)


BENCHMARK = common.Benchmark(
    title="Grouped scaled SwiGLU MXFP8 kernel benchmark",
    config=CONFIG,
    selectors={
        "input": "Input",
        "layout": "Layout",
        "experts": "Experts",
        "activation": "Activation",
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
    extra=("members",),
)

if __name__ == "__main__":
    common.main(BENCHMARK, __file__, __doc__)
