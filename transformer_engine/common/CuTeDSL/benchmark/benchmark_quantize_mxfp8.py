# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kernel-only benchmark of the MXFP8 quantize kernels (CUPTI kernel timing, L2 flushed).

CuTeDSL kernels come from this source tree (no rebuild needed); CUDA kernels come from the
installed TE (set PYTHONPATH=<repo> to use your own build). Writes a CSV.

examples:
  python benchmark_quantize_mxfp8.py
  python benchmark_quantize_mxfp8.py --mode manual --shapes "4096x7168;16384x4096" --impl CUDA
  python benchmark_quantize_mxfp8.py --compare                          # main vs HEAD
  python benchmark_quantize_mxfp8.py --compare --old CUDA --mode manual --shapes "4096x7168"
  python benchmark_quantize_mxfp8.py --fusion "none;dact;dactbias" --format html
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
CONFIG = ["M", "N", "direction", "fusion", "swizzle", "input"]


def _round_up(x, y):
    return (x + y - 1) // y * y


def _cutedsl_launcher(activation):
    """Returns setup(x, act_input, direction, fusion, swizzle) -> launch(), calling the compiled
    CuTeDSL kernel directly."""
    # pylint: disable=import-outside-toplevel
    import torch

    common.cuda_backend(True)  # not needed for the direct call, keeps TE's dispatch consistent
    from transformer_engine.common.CuTeDSL import get_mxfp8_quantization_function

    # Resolve the defining module instead of naming it: it was renamed between revisions.
    qc = sys.modules[get_mxfp8_quantization_function.__module__]
    print(f"CuTeDSL kernels from {qc.__file__}")

    @functools.lru_cache(maxsize=None)
    def compiled(dtype, direction, fusion, swizzle):
        rowwise, colwise = common.DIRECTIONS[direction]
        with_dact = "dact" in fusion
        cfg = qc.MXFP8QuantizeConfig(
            dtype,
            "Float8E4M3",
            rowwise,
            colwise,
            with_gemm_swizzled_scales=swizzle == "on",
            with_amax=False,
            with_dbias="bias" in fusion,
            with_dact=with_dact,
            with_act=fusion == "act",
            activation={"act": activation}.get(fusion, f"d{activation}" if with_dact else "none"),
        )
        fn = qc.compile_cutedsl_function_from_cfg(cfg)
        return getattr(fn, "__tvm_ffi_object__", lambda: None)() or fn

    def setup(x, act_input, direction, fusion, swizzle):
        m, n = x.shape
        rowwise, colwise = common.DIRECTIONS[direction]
        empty = functools.partial(torch.empty, device="cuda")
        fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
        o_row = empty((m, n), dtype=fp8) if rowwise else None
        s_row = (
            empty((_round_up(m, 128), _round_up(-(-n // 32), 4)), dtype=e8m0) if rowwise else None
        )
        o_col = empty((m, n), dtype=fp8) if colwise else None
        s_col = (
            empty((_round_up(-(-m // 32), 4), _round_up(n, 128)), dtype=e8m0) if colwise else None
        )
        act_input = act_input if "dact" in fusion else None
        # One partial-sum row per CTA; a CTA covers at least one 32-row MXFP8 block.
        workspace = empty((-(-m // 32), n), dtype=torch.float32) if "bias" in fusion else None
        dtype = next(v[1] for v in common.INPUTS.values() if getattr(torch, v[0]) == x.dtype)
        fn = compiled(dtype, direction, fusion, swizzle)
        stream = torch.cuda.current_stream().cuda_stream
        outs = (o_row, s_row, o_col, s_col)
        return lambda: fn(x, *outs, None, None, act_input, workspace, stream)

    return setup


def _cuda_launcher(activation):
    """Returns setup(x, act_input, direction, fusion, swizzle) -> launch(), running the
    installed TE's CUDA kernel."""
    # pylint: disable=import-outside-toplevel
    common.cuda_backend(False)
    import transformer_engine.pytorch  # pylint: disable=unused-import
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import MXFP8Quantizer

    print(f"CUDA kernels from {common.disable_cutedsl_dispatch()}")

    def setup(x, act_input, direction, fusion, swizzle):
        rowwise, colwise = common.DIRECTIONS[direction]
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=colwise
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
    """Times the `impl` quantize kernel for every shape, direction, fusion, swizzle and input."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch

    activation = args.activation or "gelu"
    setup = (_cuda_launcher if impl == "CUDA" else _cutedsl_launcher)(activation)
    # Helper kernels such as scale-padding zeroing or the dbias reduction are not timed.
    prefix = "quantize" if impl == "CUDA" else "kernel_cutlass"

    @functools.lru_cache(maxsize=1)
    def make_inputs(m, n, input_dtype):
        """The input (or incoming gradient) and, for dact, the forward activation input."""
        dtype = getattr(torch, common.INPUTS[input_dtype][0])
        return [torch.randn((m, n), dtype=dtype, device="cuda") for _ in range(2)]

    configs = [
        (m, n, d, f, sw, inp)
        for (m, n) in args.shape_list
        for inp in args.inputs
        for d in args.direction
        for f in args.fusions
        for sw in args.swizzles
    ]
    launches = (setup(*make_inputs(m, n, inp), d, f, sw) for m, n, d, f, sw, inp in configs)
    timed = common.time_kernels(launches, args.runs, args.warmup, lambda k: k.startswith(prefix))
    rows = []
    for (m, n, d, f, sw, inp), (times, kernel) in zip(configs, timed):
        row = {"M": m, "N": n, "direction": d, "fusion": f, "swizzle": sw, "input": inp}
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
