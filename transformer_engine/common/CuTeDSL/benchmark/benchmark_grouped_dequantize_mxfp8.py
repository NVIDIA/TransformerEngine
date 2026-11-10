# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kernel-only benchmark of the grouped MXFP8 dequantize kernels (CUPTI kernel timing, cold L2).

Groups are those of benchmark_grouped_quantize_mxfp8.py: G members in an M x N buffer, which
varying members fill only part of, as with CUDA graphs. Both implementations run through TE's
dispatch: --impl CUDA forces its CUDA kernels; --impl CuTeDSL turns the CuTeDSL backend on,
which runs a CuTeDSL kernel where the TE build has one and CUDA otherwise; the kernel column
says which ran. --direction picks which data the grouped MXFP8 tensor holds; its quantization
is set up outside the timing. Scales are compact: TE's group_dequantize binding does not pass
the GEMM-swizzled flag on. Writes a CSV.

examples:
  python benchmark_grouped_dequantize_mxfp8.py
  python benchmark_grouped_dequantize_mxfp8.py --layout all --direction colwise --impl CUDA
  python benchmark_grouped_dequantize_mxfp8.py --compare --old CUDA --format html
"""

import functools
import itertools
import sys

import common
import benchmark_grouped_quantize_mxfp8 as grouped
from benchmark_dequantize_mxfp8 import DIRECTIONS, TE_DTYPES

CONFIG = ["M", "N", "experts", "layout", "direction", "output"]


def run(args, impl):
    """Times the `impl` grouped dequantize kernel for every config."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch

    tex = common.te_backend(impl)

    def factory(m, n, g, layout, direction, output, group):
        quantizer = grouped._quantizer(tex, direction, "off")  # pylint: disable=protected-access
        x = torch.randn(grouped.capacity(m, n, g, layout), dtype=torch.bfloat16, device="cuda")
        first, last = grouped._dims_tensors(group, layout)  # pylint: disable=protected-access
        quantized = tex.group_quantize(x, quantizer, len(group), first, last)
        otype = getattr(tex.DType, TE_DTYPES[output])
        return lambda: tex.group_dequantize(quantized, otype)

    def nbytes(direction, output, group):
        numel = sum(a * b for a, b in group)
        return common.mxfp8_data_bytes(numel, direction) + numel * common.INPUTS[output][2]

    configs = [
        (m, n, g, lay, d, out, grouped.members(m, n, g, lay))
        for (m, n), g, lay, out, d in itertools.product(
            args.shape_list, args.experts_list, args.layouts, args.outputs, args.direction
        )
        if grouped.valid_group(m, g, lay)
    ]
    if not configs:
        sys.exit("No config to run: check --shapes/--experts/--layout")
    factories = ((nbytes(c[4], c[5], c[6]), functools.partial(factory, *c)) for c in configs)
    # As in the grouped quantize benchmark, a descriptor update is part of the dequantize work.
    keep = common.te_kernels("^group_dequantize", "^update_tma_descriptors$")
    timed = common.time_kernels(factories, args.runs, args.warmup, keep)
    rows = []
    for config, result in zip(configs, timed):
        if result is None:
            continue
        times, kernel, _ = result
        *values, group = config
        row = dict(zip(CONFIG, values))
        row.update(common.result_row(times, kernel))
        row["GB/s"] = nbytes(row["direction"], row["output"], group) / (row["time_us"] * 1e3)
        row["members"] = ";".join(f"{a}x{b}" for a, b in group)
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFIG + common.RESULT_COLUMNS + ["members"])


def add_arguments(parser):
    """The grouped dequantize options."""
    common.add_suite_arguments(parser)
    grouped.add_group_arguments(parser)
    parser.add_argument(
        "--direction",
        nargs="+",
        default=list(DIRECTIONS),
        choices=list(DIRECTIONS),
        help="which MXFP8 data is dequantized (default: both, one at a time)",
    )
    common.add_dtype_argument(parser, "--output", "output")


def parse_arguments(parser, args):
    """Validates the grouped dequantize options."""
    common.parse_suite_arguments(parser, args)
    grouped.parse_group_arguments(parser, args)
    args.outputs = common.parse_list(parser, args.output, list(common.INPUTS), "--output")


def worker_arguments(args):
    """The grouped dequantize options for a compare worker."""
    return (
        common.suite_worker_arguments(args)
        + grouped.group_worker_arguments(args)
        + ["--direction", *args.direction]
        + ["--output", ";".join(args.outputs)]
    )


def describe(args):
    """config.txt lines for the benchmarked suite."""
    return grouped.describe_group(args) + [
        f"directions: {', '.join(args.direction)}",
        f"output:     {', '.join(args.outputs)}",
    ]


def heading(values, _activation):
    """`varying_first · G=8 · colwise → bf16` for a chart."""
    parts = [values["layout"], f"G={values['experts']}"]
    parts.append(f"{values['direction']} → {values['output']}")
    return " · ".join(parts)


BENCHMARK = common.Benchmark(
    title="Grouped MXFP8 dequantize kernel benchmark",
    config=CONFIG,
    selectors={
        "output": "Output",
        "layout": "Layout",
        "experts": "Experts",
        "direction": "Direction",
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
