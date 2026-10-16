# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kernel-only benchmark of the grouped MXFP8 quantize kernels (CUPTI kernel timing, L2
flushed), e.g. the per-expert activations of an MoE layer.

A group is G members (experts) holding M rows of hidden size N in total; --layout picks how
the member shapes vary. Uneven splits are random but seeded from the shape, so every run and
both sides of --compare see the same group. CuTeDSL kernels come from this source tree (no
rebuild needed); CUDA kernels come from the installed TE (set PYTHONPATH=<repo> to use your own
build). The CUDA time includes its separate TMA-descriptor update kernel, which CuTeDSL does
inside its kernel. Writes a CSV.

examples:
  python benchmark_grouped_quantize_mxfp8.py
  python benchmark_grouped_quantize_mxfp8.py --mode manual --shapes "16384x7168" --experts "8;32"
  python benchmark_grouped_quantize_mxfp8.py --compare --old CUDA --format html
  python benchmark_grouped_quantize_mxfp8.py --layout all --fusion "none;dbias"
"""

import functools
import itertools
import random
import sys

import common

# MoE shapes: N is the hidden size (DeepSeek-V3 / Kimi K2 7168, Qwen3-235B 4096) or an expert's
# intermediate size (DeepSeek-V3 / Kimi K2 2048, gpt-oss 2880); M is the tokens x top-k that
# land on one GPU; G is the experts on one GPU.
DEFAULT_N = [2048, 2880, 4096, 7168]
DEFAULT_M = [4096, 8192, 16384, 32768, 65536]
DEFAULT_EXPERTS = [4, 8, 16]
DEBUG_N = [2048, 4096]
DEBUG_M = [4096, 16384]
DEBUG_EXPERTS = [4]
FUSIONS = ("none", "dbias")  # grouped quantization has no PyTorch binding for (d)activations
# --layout name -> shape representation of the CuTeDSL config (ShapeRepresentation in C++)
LAYOUTS = {
    "same": "sbd",  # every member M/G x N
    "varying_first": "vfd",  # members m_i x N, sum m_i = M: uneven tokens per expert
    "varying_last": "vld",  # members M/G x n_i, mean n_i ~ N
    "varying_both": "vbd",  # members m_i x n_i
}
SINGLE_TENSOR_LAYOUTS = ("same", "varying_first")  # dbias needs a common last dim
ROW_BLOCK = 128  # every member's first dim (and varying last dims) must be a multiple of it
CONFIG = ["M", "N", "experts", "layout", "direction", "fusion", "swizzle", "input"]


def _split(total, parts, block, rng):
    """`parts` random multiples of `block` summing to `total` (some may be 0, as for an expert
    no token routes to)."""
    weights = [rng.lognormvariate(0, 0.6) for _ in range(parts)]
    units = total // block
    shares = [w / sum(weights) * units for w in weights]
    counts = [int(s) for s in shares]
    for i in sorted(range(parts), key=lambda i: shares[i] - counts[i], reverse=True)[
        : units - sum(counts)
    ]:
        counts[i] += 1
    return [c * block for c in counts]


def members(m, n, experts, layout):
    """Member shapes [(m_i, n_i)] of the group; deterministic for a given shape."""
    rng = random.Random(f"{m}x{n}x{experts}x{layout}")
    if layout in ("same", "varying_last"):
        firsts = [m // experts] * experts
    else:
        firsts = _split(m, experts, ROW_BLOCK, rng)
    if layout in ("same", "varying_first"):
        lasts = [n] * experts
    else:
        lasts = _split(_round_up(n * experts, ROW_BLOCK), experts, ROW_BLOCK, rng)
        lasts = [max(v, ROW_BLOCK) for v in lasts]  # a zero-width member has no last dim
    return list(zip(firsts, lasts))


def _round_up(x, y):
    return (x + y - 1) // y * y


def logical_shape(group, layout):
    """The 2D view the grouped kernels take, as in tests/pytorch/mxfp8 group_dims()."""
    if layout in ("same", "varying_first"):
        return sum(m for m, _ in group), group[0][1]
    if layout == "varying_last":
        return group[0][0], sum(n for _, n in group)
    return 1, sum(m * n for m, n in group)


def _dims_tensors(group, layout):
    """(first_dims, last_dims) int64 device tensors, None where the dim is shared."""
    import torch  # pylint: disable=import-outside-toplevel

    def to_tensor(dims):
        return torch.tensor(dims, dtype=torch.int64, device="cuda")

    first = None if layout in ("same", "varying_last") else to_tensor([m for m, _ in group])
    last = None if layout in ("same", "varying_first") else to_tensor([n for _, n in group])
    return first, last


def _te():
    """TE's PyTorch extension, with its quantize calls on the CUDA kernels."""
    # pylint: disable=import-outside-toplevel
    common.cuda_backend(False)
    import transformer_engine.pytorch  # pylint: disable=unused-import
    import transformer_engine_torch as tex

    core = common.disable_cutedsl_dispatch()
    return tex, core


def _quantizer(tex, direction, swizzle):
    from transformer_engine.pytorch import MXFP8Quantizer  # pylint: disable=import-outside-toplevel

    rowwise, colwise = common.DIRECTIONS[direction]
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=colwise)
    quantizer.optimize_for_gemm = swizzle == "on"
    return quantizer


def _te_group_quantize(tex, x, quantizer, group, layout, fusion):
    """Returns launch() running TE's grouped quantize, plus the grouped output it writes."""
    first, last = _dims_tensors(group, layout)
    if fusion == "dbias":
        out, _ = tex.bgrad_group_quantize(x, quantizer, len(group), first, last)
        return lambda: tex.bgrad_group_quantize(x, quantizer, len(group), first, last), out
    # TE only reuses an output (output=) for uniform groups, so every launch allocates one.
    out = tex.group_quantize(x, quantizer, len(group), first, last)
    return lambda: tex.group_quantize(x, quantizer, len(group), first, last), out


def _cuda_launcher():
    """Returns setup(x, group, layout, direction, fusion, swizzle) -> launch(), running the
    installed TE's CUDA grouped kernel."""
    tex, core = _te()
    print(f"CUDA kernels from {core}")

    def setup(x, group, layout, direction, fusion, swizzle):
        quantizer = _quantizer(tex, direction, swizzle)
        return _te_group_quantize(tex, x, quantizer, group, layout, fusion)[0]

    return setup


def _cutedsl_launcher():
    """Returns setup(x, group, layout, direction, fusion, swizzle) -> launch(), calling the
    compiled CuTeDSL grouped kernel directly on the buffers of a TE grouped output."""
    # pylint: disable=import-outside-toplevel
    import torch

    tex, _ = _te()  # only allocates the grouped outputs, with the CUDA kernels
    from transformer_engine.common.CuTeDSL import get_mxfp8_group_quantization_function

    # Resolve the defining module instead of naming it, in case it gets renamed.
    gq = sys.modules[get_mxfp8_group_quantization_function.__module__]
    print(f"CuTeDSL kernels from {gq.__file__}")

    @functools.lru_cache(maxsize=None)
    def compiled(dtype, layout, direction, fusion, swizzle):
        rowwise, colwise = common.DIRECTIONS[direction]
        cfg = gq.MXFP8GroupQuantizeConfig(
            dtype,
            "Float8E4M3",
            rowwise,
            colwise,
            LAYOUTS[layout],
            with_gemm_swizzled_scales=swizzle == "on",
            with_dbias=fusion == "dbias",
        )
        fn = gq.compile_cutedsl_function_from_cfg(cfg)
        return getattr(fn, "__tvm_ffi_object__", lambda: None)() or fn

    def setup(x, group, layout, direction, fusion, swizzle):
        rowwise, colwise = common.DIRECTIONS[direction]
        quantizer = _quantizer(tex, direction, swizzle)
        _, out = _te_group_quantize(tex, x, quantizer, group, layout, "none")
        shape = tuple(x.shape)
        fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
        o_row = out.rowwise_data.view(fp8).view(shape) if rowwise else None
        o_col = out.columnwise_data.view(fp8).view(shape) if colwise else None
        # As the C++ bridge does: a missing direction's scale slot takes the other's buffer.
        s_row = (out.scale_inv if rowwise else out.columnwise_scale_inv).view(e8m0)
        s_col = (out.columnwise_scale_inv if colwise else out.scale_inv).view(e8m0)
        first, last = _dims_tensors(group, layout)
        offsets = None if layout == "same" else out.tensor_offsets
        single = layout in SINGLE_TENSOR_LAYOUTS
        # Scratch the kernel builds its per-member TMA descriptors in.
        tensormaps = (
            None
            if single
            else torch.zeros(gq.TENSOR_MAP_STORAGE_INT64_COUNT, dtype=torch.int64, device="cuda")
        )
        workspace = (
            torch.empty((-(-shape[0] // ROW_BLOCK), shape[1]), dtype=torch.float32, device="cuda")
            if fusion == "dbias"
            else None
        )
        dtype = next(v[1] for v in common.INPUTS.values() if getattr(torch, v[0]) == x.dtype)
        fn = compiled(dtype, layout, direction, fusion, swizzle)
        stream = torch.cuda.current_stream().cuda_stream
        args = (x, o_row, o_col, s_row, s_col, offsets, first, last, tensormaps, len(group))
        return lambda: fn(*args, None, None, workspace, stream)

    return setup


def _cuda_kernel(name):
    # The descriptor update is part of the CUDA path's quantize work (CuTeDSL does it in-kernel);
    # the offsets computation and the dbias reduction run outside both kernels.
    return name.startswith("group_quantize") or name == "update_tma_descriptors"


def run(args, impl):
    """Times the `impl` grouped quantize kernel for every config."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch

    setup = (_cuda_launcher if impl == "CUDA" else _cutedsl_launcher)()
    keep = _cuda_kernel if impl == "CUDA" else (lambda k: k.startswith("kernel_cutlass"))

    @functools.lru_cache(maxsize=1)
    def make_input(shape, input_dtype):
        return torch.randn(
            shape, dtype=getattr(torch, common.INPUTS[input_dtype][0]), device="cuda"
        )

    configs = []
    for (m, n), g, lay, inp, d, f, sw in itertools.product(
        args.shape_list,
        args.experts_list,
        args.layouts,
        args.inputs,
        args.direction,
        args.fusions,
        args.swizzles,
    ):
        if f == "dbias" and lay not in SINGLE_TENSOR_LAYOUTS:
            continue
        if lay in ("same", "varying_last") and m % (g * ROW_BLOCK):
            continue  # needs M/G rows per member, a multiple of 128
        configs.append((m, n, g, lay, d, f, sw, inp, members(m, n, g, lay)))
    if not configs:
        sys.exit("No config to run: check --shapes/--experts/--layout/--fusion")
    launches = (
        setup(make_input(logical_shape(grp, lay), inp), grp, lay, d, f, sw)
        for _, _, _, lay, d, f, sw, inp, grp in configs
    )
    timed = common.time_kernels(launches, args.runs, args.warmup, keep)
    rows = []
    for (m, n, g, lay, d, f, sw, inp, grp), (times, kernel) in zip(configs, timed):
        row = dict(zip(CONFIG, (m, n, g, lay, d, f, sw, inp)))
        row.update(common.result_row(times, kernel))
        numel = sum(a * b for a, b in grp)
        row["GB/s"] = common.mxfp8_bytes(numel, inp, d) / (row["time_us"] * 1e3)
        row["members"] = ";".join(f"{a}x{b}" for a, b in grp)
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFIG + common.RESULT_COLUMNS + ["members"])


def add_arguments(parser):
    """The grouped options."""
    common.add_quantize_arguments(parser, FUSIONS)
    parser.add_argument(
        "--shapes",
        metavar='"MxN;..."',
        help=(
            "total rows M (a multiple of 128) x hidden size N (a multiple of 32), for --mode manual"
        ),
    )
    parser.add_argument(
        "--experts",
        metavar="G;...",
        help=f"members per group (default: {';'.join(map(str, DEFAULT_EXPERTS))})",
    )
    parser.add_argument(
        "--layout",
        default="same;varying_first",
        metavar="all|L;L;...",
        help=(
            f"all, or a ;-separated subset of {';'.join(LAYOUTS)}: whether the members' rows"
            " and/or columns vary (default: same;varying_first)"
        ),
    )


def parse_arguments(parser, args):
    """Validates the grouped options."""
    common.parse_quantize_arguments(parser, args, FUSIONS)
    args.layouts = common.parse_list(parser, args.layout, list(LAYOUTS), "--layout")
    debug = args.mode == "debug"
    if args.experts:
        args.experts_list = common.parse_ints(parser, args.experts, "--experts")
    else:
        args.experts_list = DEBUG_EXPERTS if debug else DEFAULT_EXPERTS
    try:
        if args.mode == "manual":
            args.shape_list = common.parse_shapes(args.shapes)
            if any(m % ROW_BLOCK for m, _ in args.shape_list):
                parser.error(f"--shapes: M must be a multiple of {ROW_BLOCK} for grouped quantize")
        else:
            ms, ns = (DEBUG_M, DEBUG_N) if debug else (DEFAULT_M, DEFAULT_N)
            args.shape_list = list(itertools.product(ms, ns))
    except ValueError as e:
        parser.error(str(e))


def worker_arguments(args):
    """The grouped options for a compare worker."""
    worker = common.quantize_worker_arguments(args)
    return worker + [
        "--experts",
        ";".join(map(str, args.experts_list)),
        "--layout",
        ";".join(args.layouts),
    ]


def describe(args):
    """config.txt lines for the benchmarked suite."""
    if args.mode == "manual":
        shapes = "manual: " + "; ".join(f"{m}x{n}" for m, n in args.shape_list)
    else:
        ms = sorted({m for m, _ in args.shape_list})
        ns = sorted({n for _, n in args.shape_list})
        shapes = f"{args.mode} suite, M = {', '.join(map(str, ms))} x N = {', '.join(map(str, ns))}"
    return [
        f"mode:       {args.mode}",
        f"shapes:     {shapes}",
        f"experts:    {', '.join(map(str, args.experts_list))}",
        f"layouts:    {', '.join(args.layouts)} (member shapes are in results.csv)",
        f"directions: {', '.join(args.direction)}",
        f"fusions:    {', '.join(args.fusions)}",
        f"swizzle:    {', '.join(args.swizzles)}",
        f"input:      {', '.join(args.inputs)}",
    ]


def heading(values, activation):
    """`bf16 · varying_first · G=8 · none · both` for a chart."""
    shared = common.quantize_heading(values, activation).split(" · ")
    return " · ".join([shared[0], values["layout"], f"G={values['experts']}", *shared[1:]])


BENCHMARK = common.Benchmark(
    title="Grouped MXFP8 quantize kernel benchmark",
    config=CONFIG,
    selectors={
        "input": "Input",
        "layout": "Layout",
        "experts": "Experts",
        "fusion": "Fusion",
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
    extra=("members",),
)

if __name__ == "__main__":
    common.main(BENCHMARK, __file__, __doc__)
