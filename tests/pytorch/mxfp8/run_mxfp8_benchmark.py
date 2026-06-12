#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""One-shot driver for the MXFP8 quantize benchmark.

`bench_mxfp8_cutedsl.py` measures ONE backend in ONE timing mode per process
(the backend is latched by NVTE_ENABLE_CUTEDSL_QUANT_BACKEND at import). This
driver runs every {backend} x {timing-mode} combination in its own subprocess
and prints a single merged table, so one command gives you the full picture:

    backend:  cpp = CUDA C++ kernels      (NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=0)
              dsl = CuTeDSL kernels        (NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1)
    mode:     GPU = kernel time, cold L2   (nsys NVTX Range Kernel Summary; the
                                            whole matrix in ONE nsys run per
                                            backend, per-shape via NVTX, needs nsys)
              CPU = host dispatch time     (in-process wall clock, --no-evict-l2)

Default (curated): combos {plain, dbias, gelu, dgelu} x directions {row, col,
both} (both = bidirectional, rowwise+colwise in one pass) x swizzle {off, on} x
bf16 x e4m3 x an LLM-representative shape set (hidden 4096-14336, a few thousand
tokens), for both backends and both modes. Override any axis (--preset/--shapes
for sizes), or use --all for the full matrix.

Usage:
    python run_mxfp8_benchmark.py                         # curated default
    python run_mxfp8_benchmark.py --preset llm --modes gpu
    python run_mxfp8_benchmark.py --combos plain --directions row --swizzle off
    python run_mxfp8_benchmark.py --backends dsl          # CuTeDSL only
    python run_mxfp8_benchmark.py --all --preset tiny     # everything
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).with_name("bench_mxfp8_cutedsl.py")
# Repo root (…/tests/pytorch/mxfp8/run_mxfp8_benchmark.py -> 3 levels up). The bench
# is launched as a script, so its own dir lands on sys.path and `import
# transformer_engine` would resolve to the INSTALLED package — not this checkout —
# meaning the CuTeDSL backend silently runs the installed (often stale / CUDA-
# fallback) kernels. Putting the repo root on PYTHONPATH forces the local TE so the
# `dsl` backend actually exercises the local CuTeDSL kernels.
_REPO_ROOT = Path(__file__).resolve().parents[3]
# CPU mode times host dispatch in-process; GPU mode reads kernel time from nsys.
_CPU_FLAG = "--no-evict-l2"

# Kernel-summary rows whose name matches any of these are the L2-evict op (a
# torch fill / memset), not the quantize — excluded from the measured GPU time.
_EVICT_NAME_PATTERNS = ("memset", "memcpy", "fill", "elementwise_kernel",
                        "vectorized_elementwise")

# Full axes (mirror bench_mxfp8_cutedsl.py) — used to expand "all" / --all.
_ACTS = ["gelu", "silu", "relu", "qgelu", "srelu"]
_ALL_COMBOS = (["plain", "dbias"] + _ACTS
               + ["d" + a for a in _ACTS] + ["dbias_d" + a for a in _ACTS])
_ALL_DTYPES = ["bf16", "fp16", "fp32"]
_ALL_FP8 = ["e4m3", "e5m2"]

# Default shapes (tokens M x hidden N): LLM-representative — hidden dims 4096-
# 14336 (7B/70B hidden + Llama-3 MLP intermediate), a few thousand tokens. All
# multiples of 128 so the swizzled-scale layout applies. Override with --preset
# / --shapes.
_DEFAULT_SHAPES = "4096,4096;4096,8192;8192,8192;4096,14336"


def _expand(val, full):
    """Expand a comma list, turning the literal 'all' into the full axis."""
    if val is None:
        return None
    items = [v.strip() for v in val.split(",") if v.strip()]
    return ",".join(full) if items == ["all"] else val


def _detect_cute_dsl_arch():
    """sm_<major><minor>[a] for the current device (CuTeDSL compile target)."""
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}{'a' if major >= 9 else ''}"
    except Exception:
        return None


def _backend_env(backend):
    """Process env that latches the backend (and CuTeDSL arch) for a bench run."""
    env = dict(os.environ)
    env["NVTE_ENABLE_CUTEDSL_QUANT_BACKEND"] = "1" if backend == "dsl" else "0"
    if backend == "dsl":
        if "CUTE_DSL_ARCH" not in env:
            arch = _detect_cute_dsl_arch()
            if arch:
                env["CUTE_DSL_ARCH"] = arch
        # We EXPECT the CuTeDSL backend to handle every quantize in a `dsl` run.
        # If the dispatcher falls back to CUDA (e.g. the wrong transformer_engine
        # got imported, or the kernel didn't register/compile), this makes it warn
        # loudly instead of silently producing CUDA numbers labelled "dsl".
        env.setdefault("NVTE_WARN_IF_CUTEDSL_BACKEND_NOT_CHOSEN", "1")
    # Force the bench subprocess to import THIS checkout's transformer_engine (so
    # `dsl` runs the local CuTeDSL kernels), not the installed package. Without
    # this, `python bench.py` imports installed TE and `dsl` silently falls back.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _run_cpu(backend, passthrough):
    """CPU mode: bench loops shapes/combos in-process, times host dispatch with a
    wall clock, writes CSV. Returns {(tag, M, N, dir): (us, gbps)}."""
    env = _backend_env(backend)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    cmd = [sys.executable, str(BENCH), _CPU_FLAG, "--csv", csv_path] + passthrough
    print(f"[run] backend={backend:3s} mode=cpu: {' '.join(cmd)}", file=sys.stderr)
    try:
        proc = subprocess.run(env=env, args=cmd, stdout=subprocess.DEVNULL)
        if proc.returncode != 0:
            print(f"[warn] backend={backend} mode=cpu exited {proc.returncode}; "
                  "skipping (is this backend available?)", file=sys.stderr)
            return None
        rows = {}
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                rows[(r["tag"], int(r["M"]), int(r["N"]), r["dir"])] = (
                    float(r["us"]), float(r["gbps"]))
        return rows
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def _parse_nsys_bytes_all(stdout):
    """All NSYS_BYTES lines -> {(tag, M, N, dir): bytes}. The bench emits one per
    workload before its timed loop."""
    out = {}
    for line in stdout.splitlines():
        if line.startswith("NSYS_BYTES"):
            kv = dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok)
            try:
                out[(kv["tag"], int(kv["M"]), int(kv["N"]), kv["dir"])] = int(kv["bytes"])
            except (KeyError, ValueError):
                continue
    return out


def _parse_nvtx_kern_sum(stats_csv, iters):
    """Per-workload per-iter kernel time (ns) from `nsys stats --report
    nvtx_kern_sum --format csv`. Each row is (NVTX range, kernel) with the real
    CUPTI 'Total Time (ns)'. We sum the kernel Total over each QBENCH range and
    divide by its range-instance count (== iters) -> per-iter kernel time, the
    same precision as cuda_gpu_kern_sum but bucketed by workload. The L2-evict
    sits in the blank range (outside QBENCH) and is skipped. Returns
    {(tag, M, N, dir): per_iter_ns}."""
    lines = stats_csv.splitlines()
    hdr = next((i for i, ln in enumerate(lines)
                if "NVTX Range" in ln and "Total Time" in ln), None)
    if hdr is None:
        return {}
    reader = csv.DictReader(lines[hdr:])
    fields = reader.fieldnames or []
    rng_col = next((c for c in fields if c.strip() == "NVTX Range"), None)
    tot_col = next((c for c in fields if "Total Time" in c), None)
    inst_col = next((c for c in fields if c.strip() == "NVTX Inst"), None)
    name_col = next((c for c in fields if "Kernel Name" in c), None)
    if not (rng_col and tot_col and inst_col):
        return {}
    # range key -> [summed kernel Total ns, range instances]
    acc = {}
    for row in reader:
        name = (row.get(rng_col) or "").lstrip(":").strip()
        if not name.startswith("QBENCH|"):
            continue
        kname = (row.get(name_col) or "").lower() if name_col else ""
        if any(p in kname for p in _EVICT_NAME_PATTERNS):
            continue  # defensive; evict is in the blank range anyway
        parts = name.split("|")            # QBENCH | tag | MxN | dir
        if len(parts) != 4:
            continue
        _, tag, mxn, d = parts
        try:
            M, N = (int(v) for v in mxn.lower().split("x"))
            tot = float((row[tot_col] or "0").replace(",", ""))
            inst = int(float((row[inst_col] or "0").replace(",", "")))
        except (TypeError, ValueError):
            continue
        key = (tag, M, N, d)
        cur = acc.setdefault(key, [0.0, inst])
        cur[0] += tot
        cur[1] = inst or cur[1]
    return {k: (tot / (inst or iters)) for k, (tot, inst) in acc.items() if tot > 0}


def _run_gpu_nsys_backend(nsys, env, backend, combos, in_dtypes, fp8s, shapes_str,
                          directions, swizzles, warmup, iters, out_dir=None):
    """Profile the WHOLE matrix for one backend in a SINGLE nsys run; attribute
    per-workload kernel time via NVTX ranges (nvtx_kern_sum). Returns
    {(tag, M, N, dir): (us, gbps)}.

    One process (not one per workload) because the kernel NAME can't separate
    shapes — the QBENCH NVTX range carries M/N. Whole-process profile flushes
    reliably at exit (no capture range needed). If out_dir is set, the raw report,
    the nvtx_kern_sum CSV and the run log are kept there.

    Note CuTeDSL JIT-compiles each distinct config on first use; here that all
    happens once, in this one process, during each workload's warmup."""
    tmp = None if out_dir else tempfile.TemporaryDirectory()
    rep = (os.path.join(out_dir, f"{backend}_matrix") if out_dir
           else os.path.join(tmp.name, "rep"))
    try:
        sw_arg = ",".join("on" if s else "off" for s in swizzles)
        bench_cmd = [sys.executable, str(BENCH), "--gpu-nsys",
                     "--combos", combos, "--in-dtypes", in_dtypes, "--fp8s", fp8s,
                     "--directions", ",".join(directions), "--swizzles", sw_arg,
                     "--shapes", shapes_str, "--warmup", str(warmup),
                     "--iters", str(iters)]
        cmd = [nsys, "profile", "-o", rep, "-f", "true"] + bench_cmd
        print(f"[run] backend={backend} nsys (full matrix in 1 process): "
              f"combos={combos} dirs={','.join(directions)} sw={sw_arg} "
              f"shapes={shapes_str}", file=sys.stderr)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if out_dir:
            with open(rep + ".log", "w") as fh:
                fh.write(f"$ {' '.join(cmd)}\n[exit {proc.returncode}]\n"
                         f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        if proc.returncode != 0:
            print(f"[warn] nsys profile failed ({proc.returncode}) for "
                  f"backend={backend}; skipping.\n{proc.stderr[-800:]}",
                  file=sys.stderr)
            return {}
        bytes_map = _parse_nsys_bytes_all(proc.stdout)
        # --force-export=true: always re-derive the SQLite from the freshly
        # captured .nsys-rep. Otherwise `nsys stats` reuses a stale .sqlite left
        # next to a reused output path (e.g. --nsys-out), silently reporting old data.
        stats = subprocess.run(
            [nsys, "stats", "--force-export=true", "--report", "nvtx_kern_sum",
             "--format", "csv", rep + ".nsys-rep"], capture_output=True, text=True)
        if out_dir:
            with open(rep + ".nvtx_kern_sum.csv", "w") as fh:
                fh.write(stats.stdout)
        if stats.returncode != 0:
            print(f"[warn] nsys stats failed for backend={backend}; skipping."
                  f"\n{stats.stderr[-500:]}", file=sys.stderr)
            return {}
        per_iter = _parse_nvtx_kern_sum(stats.stdout, iters)
        rows = {}
        for key, per_iter_ns in per_iter.items():
            b = bytes_map.get(key)
            if b is None or per_iter_ns <= 0:
                continue
            rows[key] = (per_iter_ns / 1e3, b / per_iter_ns)  # us, GB/s
        if not rows:
            print(f"[warn] no QBENCH ranges parsed from nsys for backend={backend}.",
                  file=sys.stderr)
        elif out_dir:
            print(f"[nsys] saved {rep}.nsys-rep ({len(rows)} workloads)",
                  file=sys.stderr)
        return rows
    finally:
        if tmp is not None:
            # Temp path: the whole dir (report + SQLite) is removed.
            tmp.cleanup()
        else:
            # --nsys-out path: keep the raw .nsys-rep, but delete the generated
            # SQLite so a later run can never read stale data from it.
            sqlite = rep + ".sqlite"
            if os.path.exists(sqlite):
                os.remove(sqlite)


def _parse_shapes(shapes_str):
    """'M,N;M,N;...' -> [(M, N), ...]."""
    out = []
    for tok in shapes_str.split(";"):
        tok = tok.strip()
        if tok:
            m, n = tok.split(",")
            out.append((int(m), int(n)))
    return out


def _preset_shapes(preset):
    """Resolve a bench shape preset to [(M, N), ...] by asking the bench."""
    out = subprocess.run([sys.executable, str(BENCH), "--list-presets"],
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == preset:
            shapes = []
            for tok in " ".join(parts[1:]).split(","):
                tok = tok.strip().lower()
                if "x" in tok:
                    m, n = tok.split("x")
                    shapes.append((int(m), int(n)))
            return shapes
    return []


def _fwd(args_ns, passthrough_keys):
    """Rebuild the forwarded bench CLI flags from parsed args."""
    out = []
    for flag, val in passthrough_keys.items():
        if val is None or val is False:
            continue
        if val is True:
            out.append(flag)
        else:
            out += [flag, str(val)]
    return out


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--backends", default="cpp,dsl",
                    help="Comma-separated: cpp (CUDA), dsl (CuTeDSL). Default both.")
    ap.add_argument("--modes", default="gpu,cpu",
                    help="Comma-separated: gpu (kernel time), cpu (dispatch time). Default both.")
    ap.add_argument("--all", action="store_true",
                    help="Override the curated defaults with EVERY case: all 17 "
                         "combos x row/col/both x all 3 input dtypes x both fp8 "
                         "formats x swizzle on+off. Very heavy — pair with a small "
                         "--preset and modest --iters.")
    # Curated defaults: plain + one act + one dact (+ plain dbias), rowwise,
    # columnwise, and bidirectional (both), swizzle on+off. Override any axis
    # explicitly; 'all' expands an axis.
    ap.add_argument("--combos", default="plain,dbias,gelu,dgelu")
    ap.add_argument("--directions", default="row,col,both",
                    help="Comma-separated subset of row,col,both "
                         "(both = bidirectional, rowwise+colwise in one pass).")
    ap.add_argument("--swizzle", choices=["off", "on", "both"], default="both",
                    help="Swizzled scale layout: off / on / both. Default both.")
    # Shapes: default to an LLM-representative set; --preset / --shapes override.
    ap.add_argument("--preset", default=None)
    # Forwarded to bench_mxfp8_cutedsl.py (see its --help for semantics).
    ap.add_argument("--shapes")
    ap.add_argument("--in-dtypes", dest="in_dtypes")
    ap.add_argument("--fp8s")
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--nsys-out", dest="nsys_out", default=None,
                    help="Keep raw nsys artifacts in this dir (GPU mode): per "
                         "workload a <label>.nsys-rep report, .kern_sum.csv kernel "
                         "summary, and .log. Default: discarded in a temp dir.")
    args = ap.parse_args()

    # Default to the LLM-representative shape set unless the user picked shapes.
    if args.preset is None and args.shapes is None:
        args.shapes = _DEFAULT_SHAPES

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for b in backends:
        assert b in ("cpp", "dsl"), f"unknown backend {b!r}"
    for m in modes:
        assert m in ("gpu", "cpu"), f"unknown mode {m!r}"

    # --all expands every axis; otherwise honor what's given (expanding any
    # explicit "all" literal per axis).
    if args.all:
        combos, in_dtypes, fp8s = (",".join(_ALL_COMBOS),
                                   ",".join(_ALL_DTYPES), ",".join(_ALL_FP8))
        directions, swizzles = ["row", "col", "both"], [False, True]
    else:
        combos = _expand(args.combos, _ALL_COMBOS)
        in_dtypes = _expand(args.in_dtypes, _ALL_DTYPES)
        fp8s = _expand(args.fp8s, _ALL_FP8)
        directions = [d.strip() for d in args.directions.split(",") if d.strip()]
        swizzles = {"off": [False], "on": [True], "both": [False, True]}[args.swizzle]
    for d in directions:
        assert d in ("row", "col", "both"), f"unknown direction {d!r}"

    # CPU mode reuses the bench's internal shape/combo loop + CSV (host timer).
    base = _fwd(args, {
        "--preset": args.preset, "--shapes": args.shapes, "--combos": combos,
        "--in-dtypes": in_dtypes, "--fp8s": fp8s,
        "--warmup": args.warmup, "--iters": args.iters,
    })

    # GPU mode profiles the WHOLE matrix for a backend in ONE nsys run and
    # attributes per-workload kernel time by NVTX range (nvtx_kern_sum), since the
    # kernel name can't separate shapes. Resolve the axes the bench needs as
    # comma-strings; shapes as "M,N;..." (the bench loops dirs/swizzles itself).
    gpu_combos = combos or "plain"
    gpu_in_dtypes = in_dtypes or "bf16"
    gpu_fp8s = fp8s or "e4m3"
    shapes_list = (_parse_shapes(args.shapes) if args.shapes
                   else _preset_shapes(args.preset))
    gpu_shapes = ";".join(f"{m},{n}" for m, n in shapes_list)

    # GPU mode passes warmup/iters explicitly to nsys runs, so resolve None to the
    # bench's own defaults (CPU mode forwards them via _fwd, which omits None).
    gpu_warmup = args.warmup if args.warmup is not None else 10
    gpu_iters = args.iters if args.iters is not None else 100

    if args.nsys_out:
        os.makedirs(args.nsys_out, exist_ok=True)

    nsys = shutil.which("nsys")
    if "gpu" in modes and nsys is None:
        print("[warn] nsys not found on PATH; skipping GPU mode (GPU timing now "
              "comes from nsys). Install Nsight Systems or run --modes cpu.",
              file=sys.stderr)
        modes = [m for m in modes if m != "gpu"]

    # (backend, mode) -> {key: (us, gbps)}; key/tag encodes combo/dtype/fp8/swizzle
    # and direction. Sweep direction + swizzle here (bench takes one of each/run).
    data = {}
    keys = []
    for mode in modes:
        for backend in backends:
            env = _backend_env(backend)
            rows = {}
            if mode == "gpu":
                rows = _run_gpu_nsys_backend(
                    nsys, env, backend, gpu_combos, gpu_in_dtypes, gpu_fp8s,
                    gpu_shapes, directions, swizzles, gpu_warmup, gpu_iters,
                    out_dir=args.nsys_out)
            else:
                for direction in directions:
                    for swizzle in swizzles:
                        passthrough = (base + ["--direction", direction]
                                       + (["--swizzle"] if swizzle else []))
                        r = _run_cpu(backend, passthrough)
                        if r:
                            rows.update(r)
            if rows:
                data[(backend, mode)] = rows
                for k in rows:
                    if k not in keys:
                        keys.append(k)

    if not data:
        print("No results (no backend ran successfully).", file=sys.stderr)
        return 1

    # Merged table: one row per (tag, shape, dir); per mode show cpp/dsl us and
    # the cpp/dsl speedup (>1 == CuTeDSL faster). For GPU mode also show effective
    # HBM bandwidth (GB/s) for each backend — meaningful only for kernel time, not
    # host dispatch, so it's omitted for CPU mode.
    print()
    header = f"{'tag':>28}  {'shape':>11}  {'dir':>4}"
    for mode in modes:
        m = mode.upper()
        header += f"  {m+'_cpp_us':>11}  {m+'_dsl_us':>11}  {m+'_x':>6}"
        if mode == "gpu":
            header += f"  {'cpp_GB/s':>9}  {'dsl_GB/s':>9}"
    print(header)
    print("-" * len(header))
    for tag, M, N, d in keys:
        line = f"{tag:>28}  {f'{M}x{N}':>11}  {d:>4}"
        for mode in modes:
            cpp_us, cpp_bw = data.get(("cpp", mode), {}).get((tag, M, N, d), (None, None))
            dsl_us, dsl_bw = data.get(("dsl", mode), {}).get((tag, M, N, d), (None, None))
            cpp_s = f"{cpp_us:11.2f}" if cpp_us is not None else f"{'-':>11}"
            dsl_s = f"{dsl_us:11.2f}" if dsl_us is not None else f"{'-':>11}"
            spd = f"{cpp_us / dsl_us:6.2f}" if (cpp_us and dsl_us) else f"{'-':>6}"
            line += f"  {cpp_s}  {dsl_s}  {spd}"
            if mode == "gpu":
                cpp_bw_s = f"{cpp_bw:9.1f}" if cpp_bw is not None else f"{'-':>9}"
                dsl_bw_s = f"{dsl_bw:9.1f}" if dsl_bw is not None else f"{'-':>9}"
                line += f"  {cpp_bw_s}  {dsl_bw_s}"
        print(line)
    print("\n  us = microseconds/call; *_x = cpp/dsl speedup (>1 = CuTeDSL faster)")
    print("  GB/s = effective HBM bandwidth (in+out+scale bytes / GPU kernel time)")
    print("  GPU = kernel time from nsys summary (cold L2); CPU = host dispatch time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
