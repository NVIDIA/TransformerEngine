# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kernel-only benchmark of the grouped MXFP8 quantize kernels (CUPTI kernel timing, cold
L2), e.g. the per-expert activations of an MoE layer.

A group is G members (experts) in a buffer of M rows of hidden size N; --layout picks how the
member shapes vary. As with CUDA graphs, where the buffer is sized ahead of time and the member
sizes are only known on device, varying members fill a random 50-100% of the buffer and the
kernels skip the rest. Member sizes are random but seeded from the shape, so every run and
both sides of --compare see the same group. CuTeDSL kernels come from this source tree (no
rebuild needed); CUDA kernels come from the installed TE (set PYTHONPATH=<repo> to use your own
build). The CUDA time includes its separate TMA-descriptor update kernel, which CuTeDSL does
inside its kernel. A config the source tree's CuTeDSL kernels do not implement (e.g. 2D scaling)
runs through TE's dispatch instead: CUDA unless the TE build has a CuTeDSL kernel for it; the
kernel column says which ran. Writes a CSV.

examples:
  python benchmark_grouped_quantize_mxfp8.py
  python benchmark_grouped_quantize_mxfp8.py --mode manual --shapes "16384x7168" --experts "8;32"
  python benchmark_grouped_quantize_mxfp8.py --compare --old CUDA --format html
  python benchmark_grouped_quantize_mxfp8.py --layout all --fusion "none;dbias"
"""

import functools
import itertools
import math
import random
import sys
import warnings

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
    "same": "sbd",  # every member M/G x N, filling the buffer
    "varying_first": "vfd",  # members m_i x N, sum m_i <= M: uneven tokens per expert
    "varying_last": "vld",  # members M/G x n_i, sum n_i <= G x N
    "varying_both": "vbd",  # members m_i x n_i, sum m_i x n_i <= M x N
}
SINGLE_TENSOR_LAYOUTS = ("same", "varying_first")  # dbias needs a common last dim
ROW_BLOCK = 128  # every member's first dim (and varying last dims) must be a multiple of it
FILL = (0.5, 1.0)  # range of the buffer fraction varying members fill
CONFIG = ["M", "N", "experts", "layout", "direction", "fusion", "scaling", "swizzle", "input"]


def _split(budget, costs, rng):
    """Random block counts c_i with sum c_i x costs[i] <= budget (some may be 0, as for an
    expert no token routes to)."""
    weights = [rng.lognormvariate(0, 0.6) for _ in costs]
    return [int(w / sum(weights) * budget // c) for w, c in zip(weights, costs)]


def members(m, n, experts, layout):
    """Member shapes [(m_i, n_i)] of the group; deterministic for a given shape."""
    rng = random.Random(f"{m}x{n}x{experts}x{layout}")
    if layout == "same":
        return [(m // experts, n)] * experts
    fill = rng.uniform(*FILL)
    if layout == "varying_last":
        # Every member is at least one block wide: a zero-width member has no last dim.
        spare = max(int(fill * capacity(m, n, experts, layout)[1]) - experts * ROW_BLOCK, 0)
        extra = _split(spare, [ROW_BLOCK] * experts, rng)
        return [(m // experts, (1 + c) * ROW_BLOCK) for c in extra]
    if layout == "varying_first":
        lasts = [n] * experts
    else:
        lasts = [(1 + c) * ROW_BLOCK for c in _split(n * experts, [ROW_BLOCK] * experts, rng)]
    blocks = _split(fill * m * n, [ROW_BLOCK * v for v in lasts], rng)
    return [(c * ROW_BLOCK, v) for c, v in zip(blocks, lasts)]


def without_empty_groups(configs):
    """The configs (each ending in its group's member shapes) whose group has any element; a
    buffer too small for the layout can leave every member empty."""
    kept = []
    for config in configs:
        if any(a * b for a, b in config[-1]):
            kept.append(config)
        else:
            warnings.warn(f"Skipped a config: every member of its group is empty ({config[:-1]})")
    return kept


def valid_group(m, experts, layout):
    """Whether M rows split into `experts` members of this layout: equal-row layouts need M/G
    rows per member, a multiple of 128."""
    return layout not in ("same", "varying_last") or m % (experts * ROW_BLOCK) == 0


def capacity(m, n, experts, layout):
    """The buffer's 2D shape, the logical shape the grouped kernels take (as in
    tests/pytorch/mxfp8 group_dims()); the members may fill only part of it."""
    if layout in ("same", "varying_first"):
        return m, n
    if layout == "varying_last":
        # pylint: disable-next=import-outside-toplevel
        from transformer_engine.pytorch.utils import round_up_to_nearest_multiple

        cols = round_up_to_nearest_multiple(n * experts, ROW_BLOCK)
        return m // experts, max(cols, experts * ROW_BLOCK)
    return 1, m * n


def _dims_tensors(group, layout):
    """(first_dims, last_dims) int64 device tensors, None where the dim is shared."""
    import torch  # pylint: disable=import-outside-toplevel

    def to_tensor(dims):
        return torch.tensor(dims, dtype=torch.int64, device="cuda")

    first = None if layout in ("same", "varying_last") else to_tensor([m for m, _ in group])
    last = None if layout in ("same", "varying_first") else to_tensor([n for _, n in group])
    return first, last


def _quantizer(tex, direction, swizzle, scaling="1d"):
    from transformer_engine.pytorch import MXFP8Quantizer  # pylint: disable=import-outside-toplevel

    rowwise, colwise = common.DIRECTIONS[direction]
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=colwise,
        with_2d_quantization=scaling == "2d",
    )
    quantizer.optimize_for_gemm = swizzle == "on"
    return quantizer


def _te_launcher(tex):
    """Returns setup(x, group, layout, direction, fusion, scaling, swizzle) -> launch(), running
    TE's grouped quantize, which dispatches to whichever backend `tex` was imported with."""

    def setup(x, group, layout, direction, fusion, scaling, swizzle):
        quantizer = _quantizer(tex, direction, swizzle, scaling)
        first, last = _dims_tensors(group, layout)
        if fusion == "dbias":
            return lambda: tex.bgrad_group_quantize(x, quantizer, len(group), first, last)
        # TE only reuses an output (output=) for uniform groups, so every launch allocates one.
        return lambda: tex.group_quantize(x, quantizer, len(group), first, last)

    return setup


def _cutedsl_launcher():
    """Returns setup(x, group, layout, direction, fusion, scaling, swizzle) -> launch(), calling
    the compiled CuTeDSL grouped kernel directly on buffers laid out as TE's grouped output.
    Configs this source tree's CuTeDSL kernels do not support go through TE's dispatch instead,
    which runs CUDA unless the TE build has a CuTeDSL kernel."""
    # pylint: disable=import-outside-toplevel
    import torch

    common.cuda_backend(True)
    try:
        # Not in source trees from before the grouped CuTeDSL kernels.
        # pylint: disable-next=no-name-in-module
        from transformer_engine.common.CuTeDSL import get_mxfp8_group_quantization_function
    except ImportError:
        print("No grouped CuTeDSL kernels in this source tree; every config runs TE's dispatch")
        return _te_launcher(common.te_backend("CuTeDSL"))

    # Resolve the defining module instead of naming it, in case it gets renamed.
    gq = sys.modules[get_mxfp8_group_quantization_function.__module__]
    print(f"CuTeDSL kernels from {gq.__file__}")

    @functools.lru_cache(maxsize=None)
    def compiled(dtype, layout, direction, fusion, scaling, swizzle):
        rowwise, colwise = common.DIRECTIONS[direction]
        flags = {"use_2d_quantization": True} if scaling == "2d" else {}
        try:
            cfg = gq.MXFP8GroupQuantizeConfig(
                dtype,
                "Float8E4M3",
                rowwise,
                colwise,
                LAYOUTS[layout],
                with_gemm_swizzled_scales=swizzle == "on",
                with_dbias=fusion == "dbias",
                **flags,
            )
        except (TypeError, ValueError):  # not implemented in this revision
            return None
        fn = gq.compile_cutedsl_function_from_cfg(cfg)
        return getattr(fn, "__tvm_ffi_object__", lambda: None)() or fn

    @functools.lru_cache(maxsize=None)
    def te_setup():
        return _te_launcher(common.te_backend("CuTeDSL"))

    def setup(x, group, layout, direction, fusion, scaling, swizzle):
        dtype = next(v[1] for v in common.INPUTS.values() if getattr(torch, v[0]) == x.dtype)
        fn = compiled(dtype, layout, direction, fusion, scaling, swizzle)
        if fn is None:
            return te_setup()(x, group, layout, direction, fusion, scaling, swizzle)
        # Allocated here rather than by a TE grouped quantize call, so setting up a buffer set
        # runs no kernel. Sizes and offsets are those of TE's GroupedTensor
        # (MXFP8Quantizer::create_grouped_tensor): scales sized for the whole buffer, element
        # offsets of the members' data.
        rowwise, colwise = common.DIRECTIONS[direction]
        shape = tuple(x.shape)
        empty = functools.partial(torch.empty, device="cuda")
        fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
        o_row = empty(shape, dtype=fp8) if rowwise else None
        o_col = empty(shape, dtype=fp8) if colwise else None
        rows, cols = shape
        if layout == "varying_both":
            row_scales = col_scales = rows * cols // 32
        else:
            row_scales = math.prod(common.mxfp8_scale_shape(shape, False))
            col_scales = math.prod(common.mxfp8_scale_shape(shape, True))
        s_row = empty(row_scales, dtype=e8m0) if rowwise else None
        s_col = empty(col_scales, dtype=e8m0) if colwise else None
        # As the C++ bridge does: a missing direction's scale slot takes the other's buffer.
        s_row, s_col = (s_row if rowwise else s_col), (s_col if colwise else s_row)
        first, last = _dims_tensors(group, layout)
        ends = list(itertools.accumulate((m * n for m, n in group), initial=0))
        offsets = None if layout == "same" else torch.tensor(ends, dtype=torch.int64, device="cuda")
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
        stream = torch.cuda.current_stream().cuda_stream
        args = (x, o_row, o_col, s_row, s_col, offsets, first, last, tensormaps, len(group))
        return lambda: fn(*args, None, None, workspace, stream)

    return setup


# The descriptor update is part of the CUDA path's quantize work (CuTeDSL does it in-kernel);
# the offsets computation and the dbias reduction run outside both kernels.
KEEP = common.te_kernels("^group_quantize", "^update_tma_descriptors$")


def run(args, impl):
    """Times the `impl` grouped quantize kernel for every config."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch

    if impl == "CUDA":
        setup = _te_launcher(common.te_backend("CUDA"))
    else:
        setup = _cutedsl_launcher()

    def factory(m, n, g, lay, d, f, sc, sw, inp, grp):
        dtype = getattr(torch, common.INPUTS[inp][0])
        x = torch.randn(capacity(m, n, g, lay), dtype=dtype, device="cuda")
        return setup(x, grp, lay, d, f, sc, sw)

    configs = [
        (m, n, g, lay, d, f, sc, sw, inp, grp)
        for (m, n), g, lay, inp, d, f, sc, sw in itertools.product(
            args.shape_list,
            args.experts_list,
            args.layouts,
            args.inputs,
            args.direction,
            args.fusions,
            args.scalings,
            args.swizzles,
        )
        if valid_group(m, g, lay)
        and (f != "dbias" or lay in SINGLE_TENSOR_LAYOUTS)
        and (sc == "1d" or f == "none")  # TE only fuses 2D scaling into a plain cast
        for grp in [members(m, n, g, lay)]
    ]
    configs = without_empty_groups(configs)
    if not configs:
        sys.exit("No config to run: check --shapes/--experts/--layout/--fusion/--scaling")
    factories = (
        (
            common.mxfp8_bytes(sum(a * b for a, b in c[-1]), c[-2], c[4]),
            functools.partial(factory, *c),
        )
        for c in configs
    )
    timed = common.time_kernels(factories, args.runs, args.warmup, KEEP)
    rows = []
    for (m, n, g, lay, d, f, sc, sw, inp, grp), result in zip(configs, timed):
        if result is None:
            continue
        times, kernel, _ = result
        row = dict(zip(CONFIG, (m, n, g, lay, d, f, sc, sw, inp)))
        row.update(common.result_row(times, kernel))
        numel = sum(a * b for a, b in grp)
        row["GB/s"] = common.mxfp8_bytes(numel, inp, d) / (row["time_us"] * 1e3)
        row["members"] = ";".join(f"{a}x{b}" for a, b in grp)
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFIG + common.RESULT_COLUMNS + ["members"])


def add_group_arguments(parser, layouts=tuple(LAYOUTS), default_layouts="same;varying_first"):
    """--shapes, --experts and --layout, shared by the grouped benchmarks."""
    parser.add_argument(
        "--shapes",
        metavar='"MxN;..."',
        help=(
            "buffer rows M (a multiple of 128) x hidden size N (a multiple of 32), for --mode"
            " manual"
        ),
    )
    parser.add_argument(
        "--experts",
        metavar="G;...",
        help=f"members per group (default: {';'.join(map(str, DEFAULT_EXPERTS))})",
    )
    parser.add_argument(
        "--layout",
        default=default_layouts,
        metavar="all|L;L;...",
        help=(
            f"all, or a ;-separated subset of {';'.join(layouts)}: whether the members' rows"
            f" and/or columns vary (default: {default_layouts})"
        ),
    )


def parse_group_arguments(parser, args, layouts=tuple(LAYOUTS), default_n=None, debug_n=None):
    """Validates add_group_arguments' options; sets args.layouts/experts_list/shape_list."""
    args.layouts = common.parse_list(parser, args.layout, list(layouts), "--layout")
    debug = args.mode == "debug"
    if args.experts:
        args.experts_list = common.parse_ints(parser, args.experts, "--experts", minimum=1)
    else:
        args.experts_list = DEBUG_EXPERTS if debug else DEFAULT_EXPERTS
    try:
        if args.mode == "manual":
            args.shape_list = common.parse_shapes(args.shapes)
            if any(m % ROW_BLOCK for m, _ in args.shape_list):
                parser.error(f"--shapes: M must be a multiple of {ROW_BLOCK} for grouped kernels")
        else:
            ms = DEBUG_M if debug else DEFAULT_M
            ns = (debug_n or DEBUG_N) if debug else (default_n or DEFAULT_N)
            args.shape_list = list(itertools.product(ms, ns))
    except ValueError as e:
        parser.error(str(e))


def group_worker_arguments(args):
    """--experts and --layout for a compare worker."""
    return [
        "--experts",
        ";".join(map(str, args.experts_list)),
        "--layout",
        ";".join(args.layouts),
    ]


def describe_group(args):
    """config.txt lines for add_group_arguments' options."""
    if args.mode == "manual":
        shapes = "manual: " + "; ".join(f"{m}x{n}" for m, n in args.shape_list)
    else:
        ms = sorted({m for m, _ in args.shape_list})
        ns = sorted({n for _, n in args.shape_list})
        shapes = f"{args.mode} suite, M = {', '.join(map(str, ms))} x N = {', '.join(map(str, ns))}"
    return [
        f"mode:       {args.mode}",
        f"shapes:     {shapes} (M x N is the buffer; members fill part of it)",
        f"experts:    {', '.join(map(str, args.experts_list))}",
        f"layouts:    {', '.join(args.layouts)} (member shapes are in results.csv)",
    ]


def add_arguments(parser):
    """The grouped quantize options."""
    common.add_quantize_arguments(parser, FUSIONS)
    add_group_arguments(parser)


def parse_arguments(parser, args):
    """Validates the grouped quantize options."""
    common.parse_quantize_arguments(parser, args, FUSIONS)
    parse_group_arguments(parser, args)


def worker_arguments(args):
    """The grouped quantize options for a compare worker."""
    return common.quantize_worker_arguments(args) + group_worker_arguments(args)


def describe(args):
    """config.txt lines for the benchmarked suite."""
    return describe_group(args) + [
        f"directions: {', '.join(args.direction)}",
        f"fusions:    {', '.join(args.fusions)}",
        f"scaling:    {', '.join(args.scalings)}",
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
    heading=heading,
    extra=("members",),
)

if __name__ == "__main__":
    common.main(BENCHMARK, __file__, __doc__)
