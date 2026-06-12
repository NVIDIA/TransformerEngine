# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark MXFP8 quantization through the standard TE dispatch path.

This bench measures ONE backend per process. The kernel that actually runs is
decided entirely by the env var the process is launched with:

    NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=0   -> C++ CUDA kernel   (label "cpp")
    NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1   -> CuTeDSL kernel     (label "dsl")

The Python code path is identical either way (`tex.quantize` / `tex.<act>`), so
the two runs are directly comparable — only the backend under the hood differs.
Use run_nsys_profile.sh to run this script twice (disabled vs enabled) and align
the results.

Timing paths:
  --gpu-nsys            GPU time from nsys. Warms up (incl. the CuTeDSL JIT
                        compile), then runs `iters` cold-L2 iterations with each
                        fn() call wrapped in a same-named QBENCH|tag|MxN|dir NVTX
                        range, and prints the per-workload byte count. The kernel
                        time is read from nsys's NVTX Range Kernel Summary
                        (nvtx_kern_sum) by run_mxfp8_benchmark.py — NOT here. The
                        whole matrix can run in ONE nsys process (ranges, not
                        kernel names, separate the workloads).
  --no-evict-l2         CPU time, warm cache. Tight launch loop, NO sync and NO
                        L2 flush — host dispatch cost / warm-cache throughput.
  --evict-l2 (default)  GPU time, cold cache, via CUDA events (sync per iter).
                        Kept for standalone use; the driver uses --gpu-nsys for
                        GPU timing since CUDA events add a small measurement skew.
  --single              One-shot cold-cache wall-clock. Overrides --iters.

Produces NVTX-tagged iterations for Nsight Systems timeline profiling.
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.cuda.nvtx as nvtx

import transformer_engine.pytorch as te  # must precede transformer_engine_torch
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

# Shape presets — names map to lists of (M, N) pairs.
# All shapes are multiples of 64 (the CuTeDSL kernel's CHUNK_DIM).
SHAPE_PRESETS = {
    "tiny":   [(128, 128), (256, 256), (512, 512)],
    "small":  [(1024, 1024), (2048, 2048), (4096, 4096)],
    "medium": [(8192, 8192), (8192, 4096), (4096, 8192)],
    "large":  [(16384, 8192), (16384, 16384), (32768, 8192)],
    "square": [(1024, 1024), (2048, 2048), (4096, 4096),
               (8192, 8192), (16384, 16384)],
    # LLM-typical shapes: (batch*seq, hidden) for common hidden sizes
    "llm":    [(2048, 5120), (2048, 8192), (4096, 12288),
               (8192, 14336), (16384, 16384)],
    # Aspect-ratio sweep: tall-narrow and short-wide
    "aspect": [(1024, 16384), (4096, 4096), (16384, 1024),
               (512, 32768), (32768, 512)],
    "default": [(1024, 1024), (4096, 4096), (8192, 8192),
                (16384, 8192), (16384, 16384)],
    "test": [(2048, 5120)],
}


def backend_label():
    """Mirror the C++ gate in tvm_ffi_bridge.h: enabled iff the env var is set
    and its first char is not '0'."""
    v = os.environ.get("NVTE_ENABLE_CUTEDSL_QUANT_BACKEND", "")
    return "dsl" if (len(v) > 0 and v[0] != "0") else "cpp"


def parse_shapes(shapes_str: str):
    """Parse a ';'-separated list of 'M,N' pairs."""
    shapes = []
    for pair in shapes_str.split(";"):
        m, n = pair.strip().split(",")
        shapes.append((int(m), int(n)))
    return shapes


# All five elementwise activations CuTeDSL supports (gated geglu/qgeglu excluded
# — they don't route through this MXFP8 quantize path).
_ACTS = ["gelu", "silu", "relu", "qgelu", "srelu"]
_ACT_FNS = {a: getattr(tex, a) for a in _ACTS}
_DACT_FNS = {"d" + a: getattr(tex, "d" + a) for a in _ACTS}
_DBIAS_DACT_FNS = {"dbias_d" + a: getattr(tex, "dbias_d" + a) for a in _ACTS}

# combo -> kind, which selects the tex entry point and how many inputs are read:
#   plain       tex.quantize(x)              1 input  -> C++ SPECIALIZED cast-only kernel
#   act         tex.<act>(x)                 1 input  -> C++ standard kernel (IS_ACT)
#   dact        tex.<dact>(grad, act_in)     2 inputs -> C++ standard kernel (IS_DACT)
#   dbias       tex.bgrad_quantize(grad)     1 input  -> C++ standard kernel (IS_DBIAS)
#   dbias_dact  tex.dbias_<dact>(grad, act_in) 2 inputs -> standard (IS_DBIAS|IS_DACT)
# Only "plain" hits the specialized kernel (specialized::hasSpec is true solely for
# IS_DBIAS=IS_DACT=IS_ACT=false). Every other combo forces the standard MXFP8 kernel
# that the CuTeDSL backend mirrors — the apples-to-apples comparison.
COMBO_KIND = {"plain": "plain", "dbias": "dbias"}
for _a in _ACTS:
    COMBO_KIND[_a] = "act"                  # fwd activation, no dbias
    COMBO_KIND["d" + _a] = "dact"           # dactivation, no dbias
    COMBO_KIND["dbias_d" + _a] = "dbias_dact"   # dactivation + dbias

_FP8_DTYPES = {
    "e4m3": tex.DType.kFloat8E4M3,
    "e5m2": tex.DType.kFloat8E5M2,
}
_TORCH_IN_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _make_quantizer(rowwise, colwise, fp8_dtype, swizzle):
    q = MXFP8Quantizer(
        fp8_dtype=_FP8_DTYPES[fp8_dtype],
        rowwise=rowwise,
        columnwise=colwise,
    )
    q.internal = True
    if swizzle:
        q.optimize_for_gemm = True
    return q


def make_fn(combo, x, act_input, rowwise, colwise, fp8_dtype="e4m3", swizzle=False):
    """Return a 0-arg callable that quantizes `x` via the standard TE dispatch.

    Whether this lands on the C++ or CuTeDSL kernel is governed by
    NVTE_ENABLE_CUTEDSL_QUANT_BACKEND in the environment — not by this code.
    `act_input` is the second input (fwd activation input) for dact combos.

    Uses the direct `tex.*` pybinds (not `MXFP8Quantizer.__call__`, which wraps
    the result in a Float8Tensor and adds ~15 us of Python overhead that would
    skew warm-cache wall-clock).
    """
    quantizer = _make_quantizer(rowwise, colwise, fp8_dtype, swizzle)
    kind = COMBO_KIND[combo]
    if kind == "plain":
        return lambda: tex.quantize(x, quantizer)
    if kind == "act":
        op = _ACT_FNS[combo]
        return lambda: op(x, quantizer)
    if kind == "dact":
        op = _DACT_FNS[combo]
        return lambda: op(x, act_input, quantizer)        # x is grad
    if kind == "dbias":
        return lambda: tex.bgrad_quantize(x, quantizer)   # x is grad
    if kind == "dbias_dact":
        op = _DBIAS_DACT_FNS[combo]
        return lambda: op(x, act_input, quantizer)        # x is grad
    raise ValueError(f"unknown combo {combo!r}")


# Module-level L2 evict buffer. 256 MB f32 (covers B200's ~60 MB L2 with
# headroom). Allocated lazily, reused to avoid alloc churn between bench runs.
_L2_EVICT_BUF = None


def _l2_evict_buf():
    global _L2_EVICT_BUF
    if _L2_EVICT_BUF is None:
        _L2_EVICT_BUF = torch.empty(
            256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    return _L2_EVICT_BUF


def bench_once(name, fn, warmup, iters, evict_l2=False, single=False):
    """Time `fn()` under one of three modes (see module docstring).

    Returns the average per-iter time in milliseconds. `single` takes precedence
    over `evict_l2`.
    """
    if single:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        evict = _l2_evict_buf()
        nvtx.range_push(f"{name}_single")
        evict.zero_()             # async L2 flush
        torch.cuda.synchronize()  # drain evict; L2 cold AND GPU idle
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        nvtx.range_pop()
        return (t1 - t0) / 1e6

    if evict_l2:
        # GPU time, cold cache: flush L2, sync, then time the kernel with CUDA
        # events (the event pair measures pure on-GPU kernel time per iter).
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        evict = _l2_evict_buf()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_ms = 0.0
        nvtx.range_push(f"{name}_measure")
        for i in range(iters):
            evict.zero_()             # async L2 flush
            torch.cuda.synchronize()  # drain evict; L2 cold, GPU idle
            nvtx.range_push(f"{name}_iter_{i}")
            start.record()
            fn()                      # kernel launch
            end.record()
            torch.cuda.synchronize()  # wait for kernel
            nvtx.range_pop()
            total_ms += start.elapsed_time(end)  # GPU time (ms)
        nvtx.range_pop()
        return total_ms / iters

    # Warm cache, CPU time: tight launch loop, NO sync, NO flush. Kernels queue
    # asynchronously, so this is host dispatch cost / warm-cache throughput.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    nvtx.range_push(f"{name}_warm")
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter_ns()
    nvtx.range_pop()
    return (t1 - t0) / 1e6 / iters


def _workload_bytes(M, N, in_bytes_per_elt, need_act_input, rowwise, colwise):
    """Bytes the quantize moves through HBM: input read + FP8 out + e8m0 scales.

    A lower bound on DRAM traffic (assumes each byte touched once; ignores the
    dbias workspace round-trip). Used to turn a measured kernel time into GB/s.
    """
    bytes_in = M * N * in_bytes_per_elt * (2 if need_act_input else 1)
    bytes_out = bytes_scale = 0
    if rowwise:
        bytes_out += M * N                    # rowwise FP8 data (uint8)
        bytes_scale += M * (N // 32)          # rowwise e8m0 scales
    if colwise:
        bytes_out += M * N                    # colwise FP8 data
        bytes_scale += (M // 32) * N          # colwise e8m0 scales
    return bytes_in + bytes_out + bytes_scale


def bench_shape(M, N, rowwise, colwise, warmup, iters, combo="plain",
                in_dtype="bf16", fp8_dtype="e4m3", swizzle=False,
                evict_l2=False, single=False, gpu_nsys=False):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    in_dt = _TORCH_IN_DTYPES[in_dtype]
    x = torch.randn(M, N, dtype=in_dt, device="cuda")
    backend = backend_label()
    # dact / dbias_dact read a second input (the fwd activation input).
    need_act_input = COMBO_KIND[combo] in ("dact", "dbias_dact")
    act_input = (torch.randn(M, N, dtype=in_dt, device="cuda")
                 if need_act_input else None)

    dir_label = "both" if (rowwise and colwise) else ("row" if rowwise else "col")
    tag = f"{combo}_{in_dtype}_{fp8_dtype}"
    if swizzle:
        tag += "_sw"

    nvtx.range_push(f"shape_{M}x{N}_{dir_label}_{backend}_{tag}")

    fn = make_fn(combo, x, act_input, rowwise, colwise,
                 fp8_dtype=fp8_dtype, swizzle=swizzle)
    total_bytes = _workload_bytes(M, N, x.element_size(), need_act_input,
                                  rowwise, colwise)

    if gpu_nsys:
        # The driver profiles the WHOLE matrix in one nsys run per backend and
        # attributes per-workload kernel time via NVTX ranges (nsys nvtx_kern_sum):
        # the kernel NAME encodes the config but not the shape, so we tag each
        # workload with a same-named QBENCH range (which carries M/N) that wraps
        # ONLY fn(). nvtx_kern_sum reports the real CUPTI kernel durations bucketed
        # per range; warmup + the L2-evict sit OUTSIDE the range (blank range) and
        # are ignored. Emit the byte count so the driver can compute GB/s.
        print(f"NSYS_BYTES backend={backend} tag={tag} combo={combo} "
              f"M={M} N={N} dir={dir_label} bytes={total_bytes} iters={iters}",
              flush=True)
        for _ in range(max(warmup, 1)):                  # warmup OUTSIDE any range
            fn()
        torch.cuda.synchronize()

        evict = _l2_evict_buf()
        rng = f"QBENCH|{tag}|{M}x{N}|{dir_label}"
        for i in range(iters):
            evict.zero_()                                # cold L2, OUTSIDE the range
            torch.cuda.synchronize()
            nvtx.range_push(rng)                         # same name every iter ->
            fn()                                         # nvtx_kern_sum aggregates
            torch.cuda.synchronize()
            nvtx.range_pop()
        ms = None
    else:
        # Warm caches / trigger the CuTeDSL JIT compile once (not counted).
        nvtx.range_push("warm")
        fn()
        torch.cuda.synchronize()
        nvtx.range_pop()
        ms = bench_once(
            f"{backend}_{M}x{N}_{dir_label}_{tag}",
            fn, warmup, iters, evict_l2=evict_l2, single=single,
        )

    nvtx.range_pop()  # close shape_ range

    gbps = (total_bytes / (ms * 1e-3) / 1e9) if ms is not None else None

    return {
        "backend": backend,
        "shape": (M, N),
        "dir": dir_label,
        "combo": combo,
        "tag": tag,
        "ms": ms,
        "gbps": gbps,
        "bytes": total_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=(
            "Shape selection (pick one of --preset / --shapes):\n"
            "  --preset PRESET    named sweep (see --list-presets)\n"
            "  --shapes 'M,N;...' custom list\n"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--direction", choices=["row", "col", "both", "all"],
                        default="all",
                        help="Which direction(s) to benchmark")
    parser.add_argument("--directions", type=str, default=None,
                        help="Comma-separated subset of row,col,both "
                             "(overrides --direction; lets one process cover all).")
    parser.add_argument("--combo", type=str, default="plain",
                        choices=sorted(COMBO_KIND),
                        help=f"Operation: one of {sorted(COMBO_KIND)}")
    parser.add_argument("--combos", type=str, default=None,
                        help="Comma-separated list of combos (overrides --combo)")
    parser.add_argument("--in-dtype", type=str, default="bf16",
                        choices=sorted(_TORCH_IN_DTYPES))
    parser.add_argument("--in-dtypes", type=str, default=None,
                        help="Comma-separated list of input dtypes")
    parser.add_argument("--fp8", type=str, default="e4m3",
                        choices=sorted(_FP8_DTYPES))
    parser.add_argument("--fp8s", type=str, default=None,
                        help="Comma-separated list of fp8 output dtypes")
    parser.add_argument("--swizzle", action="store_true",
                        help="Enable GEMM-swizzled scales (optimize_for_gemm)")
    parser.add_argument("--swizzles", type=str, default=None,
                        help="Comma-separated subset of off,on (overrides "
                             "--swizzle; lets one process cover both).")
    parser.add_argument("--evict-l2", dest="evict_l2", action="store_true",
                        default=True,
                        help="GPU time, cold cache (default): flush L2 before "
                             "each iter, time the kernel with CUDA events.")
    parser.add_argument("--no-evict-l2", dest="evict_l2", action="store_false",
                        help="CPU time, warm cache: tight launch loop, no sync, "
                             "no L2 flush (host dispatch throughput).")
    parser.add_argument("--single", action="store_true",
                        help="One-shot cold-cache wall-clock; overrides --iters "
                             "and takes precedence over --evict-l2.")
    parser.add_argument("--gpu-nsys", dest="gpu_nsys", action="store_true",
                        help="GPU time via nsys: run cold-L2 iters with each fn() "
                             "wrapped in a same-named QBENCH NVTX range and print "
                             "the byte count (no in-process timing). The driver "
                             "reads per-range kernel time from nvtx_kern_sum. Used "
                             "by run_mxfp8_benchmark.py.")
    parser.add_argument("--preset", type=str, default=None,
                        choices=sorted(SHAPE_PRESETS),
                        help=f"Shape preset: one of {sorted(SHAPE_PRESETS)}")
    parser.add_argument("--shapes", type=str, default=None,
                        help="Custom shapes: 'M,N;M,N;...'  Overrides --preset")
    parser.add_argument("--list-presets", action="store_true",
                        help="Print all presets and exit")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results as CSV to this file")
    args = parser.parse_args()

    if args.list_presets:
        for name, shapes in SHAPE_PRESETS.items():
            shapes_str = ", ".join(f"{m}x{n}" for m, n in shapes)
            print(f"  {name:8s} {shapes_str}")
        return 0

    if args.shapes:
        shapes = parse_shapes(args.shapes)
    elif args.preset:
        shapes = SHAPE_PRESETS[args.preset]
    else:
        shapes = SHAPE_PRESETS["default"]

    _DIR_RCW = {"row": (True, False), "col": (False, True), "both": (True, True)}
    if args.directions:
        dir_names = [d.strip() for d in args.directions.split(",") if d.strip()]
    elif args.direction == "all":
        dir_names = ["row", "col", "both"]
    else:
        dir_names = [args.direction]
    for d in dir_names:
        if d not in _DIR_RCW:
            print(f"unknown direction: {d}", file=sys.stderr)
            return 1
    dirs = [(d, *_DIR_RCW[d]) for d in dir_names]

    # Swizzle list: --swizzles 'off,on' overrides the single --swizzle flag so one
    # process can cover both scale layouts.
    if args.swizzles:
        swizzles = [s.strip().lower() == "on"
                    for s in args.swizzles.split(",") if s.strip()]
    else:
        swizzles = [args.swizzle]

    if args.combos:
        combos = [c.strip() for c in args.combos.split(",")]
        for c in combos:
            if c not in COMBO_KIND:
                print(f"unknown combo: {c}", file=sys.stderr)
                return 1
    else:
        combos = [args.combo]

    in_dtypes = ([d.strip() for d in args.in_dtypes.split(",")]
                 if args.in_dtypes else [args.in_dtype])
    fp8s = ([d.strip() for d in args.fp8s.split(",")]
            if args.fp8s else [args.fp8])

    backend = backend_label()
    mode = ("gpu-nsys (kernel time from nsys summary)" if args.gpu_nsys else
            "single (one-shot cold wall-clock)" if args.single else
            "evict-l2 (GPU time, cold cache)" if args.evict_l2 else
            "warm cache (CPU dispatch time, no sync)")
    print(f"Backend: {backend}  "
          f"(NVTE_ENABLE_CUTEDSL_QUANT_BACKEND="
          f"{os.environ.get('NVTE_ENABLE_CUTEDSL_QUANT_BACKEND', '<unset>')})")
    print(f"Benchmarking {len(shapes)} shape(s) × {len(dirs)} direction(s) × "
          f"{len(combos)} combo(s) × {len(in_dtypes)} in-dtype × {len(fp8s)} fp8 "
          f"× {len(swizzles)} swizzle")
    print(f"  mode: {mode}")
    print(f"  warmup={args.warmup} iters={args.iters}")
    print(f"  combos: {combos}  in_dtypes: {in_dtypes}  fp8: {fp8s}  "
          f"dirs: {dir_names}  swizzles: {swizzles}")
    for m, n in shapes:
        print(f"  - {m}x{n}")
    print()

    # --gpu-nsys uses NVTX ranges (no cudaProfiler); otherwise wrap the whole sweep
    # so the nsys timeline is annotated when run under nsys manually.
    if not args.gpu_nsys:
        torch.cuda.profiler.start()  # used with --capture-range=cudaProfilerApi

    results = []
    for combo in combos:
        for in_dtype in in_dtypes:
            for fp8 in fp8s:
                for M, N in shapes:
                    for _, rw, cw in dirs:
                        for sw in swizzles:
                            r = bench_shape(M, N, rw, cw, args.warmup, args.iters,
                                            combo, in_dtype=in_dtype, fp8_dtype=fp8,
                                            swizzle=sw, evict_l2=args.evict_l2,
                                            single=args.single,
                                            gpu_nsys=args.gpu_nsys)
                            results.append(r)

    if not args.gpu_nsys:
        torch.cuda.profiler.stop()

    if args.gpu_nsys:
        # Kernel time is measured by nsys (the driver parses its summary) and the
        # NSYS_BYTES line is emitted by bench_shape up front (before the capture
        # range, since stop-shutdown may end the process). Nothing to print here.
        return 0

    # Summary. Columns are positional so run_nsys_profile.sh can parse them:
    #   backend  tag  shape  dir  us  GB/s
    print()
    print(f"  ({mode})")
    print(f"{'backend':>7}  {'tag':>30}  {'shape':>12}  {'dir':>4}  "
          f"{'us':>9}  {'GB/s':>9}")
    print("-" * 84)
    for r in results:
        M, N = r["shape"]
        mxn = f"{M}x{N}"
        print(f"{r['backend']:>7}  {r['tag']:>30}  {mxn:>12}  {r['dir']:>4}  "
              f"{r['ms']*1000:9.2f}  {r['gbps']:9.1f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["backend", "tag", "combo", "M", "N", "dir",
                        "us", "gbps", "bytes"])
            for r in results:
                M, N = r["shape"]
                w.writerow([r["backend"], r["tag"], r["combo"], M, N, r["dir"],
                            f"{r['ms']*1000:.3f}", f"{r['gbps']:.2f}",
                            r["bytes"]])
        print(f"\nCSV written to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
