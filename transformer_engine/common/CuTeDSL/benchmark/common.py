# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared machinery of the quantization kernel benchmarks in this folder.

A benchmark script describes itself with a `Benchmark` (its dimensions, command-line options
and how to time one implementation) and calls `main(benchmark)`; this module provides the
rest: CUPTI kernel timing over rotating buffers (cold L2), --compare across git revisions or against the
installed TE's CUDA kernels, --regenerate from earlier results, and csv/raw/png/html output.
"""

import argparse
import contextlib
import datetime
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "benchmark_quantization_template.html"
OUTPUT_FORMATS = ("csv", "png", "html", "raw")
IMPLS = {"cutedsl": "CuTeDSL", "cuda": "CUDA"}
MXFP8_BLOCK = 32
DIRECTIONS = {"rowwise": (True, False), "colwise": (False, True), "both": (True, True)}
SWIZZLES = ("off", "on")  # scales in the GEMM-ready swizzled layout
# --input name -> (torch dtype name, CuTeDSL dtype name, bytes per element)
INPUTS = {
    "bf16": ("bfloat16", "BFloat16", 2),
    "fp16": ("float16", "Float16", 2),
    "fp32": ("float32", "Float32", 4),
}
# Data touched between two uses of a buffer set, in multiples of the L2 size: enough that a set's
# input is certainly out of L2 by the time it is reused.
L2_ROTATION = 3
# Columns every results table ends with; a benchmark's own config columns come first.
RESULT_COLUMNS = ["time_us", "min_us", "rel_std_pct", "GB/s", "kernel"]
CUDA_EXPLANATION = """\
Comparing CUDA (old) vs CuTeDSL (new) {what} kernels:
  old = the CUDA kernels compiled into the currently INSTALLED TE, i.e. whatever
        `import transformer_engine` resolves to (set PYTHONPATH=<repo> to use a checkout):
          package: {package}
          version: {version}
          library: {library}
          source:  {git} (the library may be built from an older commit, see
                   the +<sha> in the version)
  new = the CuTeDSL kernels of {new} in this source tree ({repo}), JIT-compiled from
        source. This is NOT the CuTeDSL code of the installed TE.
"""
# `-P` keeps the cwd off sys.path, so this resolves TE exactly like the CUDA worker does.
INSTALLED_TE_PROBE = """\
import transformer_engine as te
from transformer_engine.common import _get_shared_object_file
print(te.__file__, te.__version__, _get_shared_object_file("core"), sep="\\n")
"""


@dataclass
class Benchmark:
    """What a benchmark script supplies to main().

    Results are one row per config: the `config` columns, then RESULT_COLUMNS. Charts plot
    time against `x` (a config column), one chart per value of every other config column,
    with `split` values shown side by side and the rest as selectors (`selectors`, in order,
    with display labels).
    """

    title: str  # e.g. "MXFP8 quantize kernel benchmark"
    config: list  # config columns, e.g. ["M", "N", "direction", ...]
    selectors: dict  # config column -> label, the html selectors in order
    x: str  # config column on the x axis
    split: str  # config column shown as side-by-side charts
    add_arguments: Callable  # (parser) -> None: benchmark-specific options
    parse_arguments: Callable  # (parser, args) -> None: validate them, set args.* defaults
    worker_arguments: Callable  # (args) -> list[str]: the same options for a compare worker
    run: Callable  # (args, impl) -> DataFrame with `config` + RESULT_COLUMNS (minus impl)
    describe: Callable  # (args) -> list[str]: config.txt lines describing the suite
    heading: Callable  # (chart values: dict, activation: str) -> str, without the split value
    extra: tuple = ()  # informational columns after RESULT_COLUMNS, e.g. a group's member shapes


# ---- timing ----------------------------------------------------------------------------


def l2_cache_bytes(device=None):
    """L2 cache size in bytes of `device` (the current CUDA device by default)."""
    import torch  # pylint: disable=import-outside-toplevel

    device = torch.cuda.current_device() if device is None else device
    return torch.cuda.get_device_properties(device).L2_cache_size


def base_name(kernel):
    """`void ns::quantize_mxfp8_kernel<...>(...)` -> `quantize_mxfp8_kernel`."""
    return re.split(r"[<(]", kernel.removeprefix("void "), maxsplit=1)[0].split("::")[-1]


def short_name(base):
    """CuTeDSL kernel symbols encode the module path and arguments; keep only the class."""
    match = re.search(r"(MXFP8\w*?Quantize\w*?Kernel)", base)
    return match.group(1) if match else base


def _cuda_kernels(prof):
    import torch  # pylint: disable=import-outside-toplevel

    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return sorted(events, key=lambda e: e.time_range.start)


def time_kernels(factories, runs, warmup, keep):
    """Times each config's kernels with CUPTI kernel timestamps (via torch.profiler).

    `factories` yields one zero-argument factory per config (consumed lazily, so buffers are
    created and freed per config); each call builds an independent set of input and output
    buffers and returns launch() on them. Launches rotate over enough sets that more than
    L2_ROTATION x the L2 size is touched before a set comes back, so every launch reads its
    input from DRAM, and run back to back, so each pays (amortized) for writing back the
    dirty outputs earlier launches left in L2, as in a real pipeline. A launch may run
    several kernels: the ones whose base name `keep` accepts are summed, the rest ignored.
    Returns, per config, (list of per-run microseconds, kernel names, number of sets).
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from torch.profiler import ProfilerActivity, profile

    # A tiny kernel with a name nothing else uses marks where each timed launch starts.
    def marker():
        torch.cuda._sleep(1)  # pylint: disable=protected-access

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        marker()
        torch.cuda.synchronize()
    marker_names = [e.name for e in _cuda_kernels(prof)]

    # Per config: [setup, warmups] M run M run ... M run M | next config's setup.
    num_sets = []
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for factory in factories:
            outputs = []  # each set keeps its last output alive, so new ones get fresh memory

            def keep_output(launch, slot):
                def run():
                    outputs[slot] = launch()

                return run

            before = torch.cuda.memory_allocated()
            outputs.append(None)
            sets = [keep_output(factory(), 0)]
            sets[0]()  # JIT compile; also materializes outputs allocated per call
            torch.cuda.synchronize()
            footprint = max(torch.cuda.memory_allocated() - before, 1)
            count = max(2, -(-L2_ROTATION * l2_cache_bytes() // footprint) + 1)
            for slot in range(1, count):
                outputs.append(None)
                sets.append(keep_output(factory(), slot))
            num_sets.append(count)
            for k in range(max(warmup, count)):  # every set touched once before timing
                sets[k % count]()
            for k in range(runs):
                marker()
                sets[k % count]()
            marker()
            del sets, outputs
        torch.cuda.synchronize()

    segments, current, i = [], [], 0
    kernels = _cuda_kernels(prof)
    while i < len(kernels):
        if [e.name for e in kernels[i : i + len(marker_names)]] == marker_names:
            segments.append(current)
            current, i = [], i + len(marker_names)
        else:
            current.append(kernels[i])
            i += 1
    expected = len(num_sets) * (runs + 1)
    if len(segments) != expected:
        raise RuntimeError(
            f"Expected {expected} launch markers in the trace, found {len(segments)}"
        )
    results = []
    for c, count in enumerate(num_sets):
        times, names = [], []
        for seg in segments[c * (runs + 1) + 1 : (c + 1) * (runs + 1)]:
            kept = [e for e in seg if keep(base_name(e.name))]
            if not kept:
                seen = sorted({short_name(base_name(e.name)) for e in seg})
                raise RuntimeError(
                    f"No benchmarked kernel in a timed launch (kernels: {seen}); did it fall back"
                    " to another implementation?"
                )
            times.append(sum(e.device_time_total for e in kept))
            names = list(dict.fromkeys(short_name(base_name(e.name)) for e in kept))
        results.append((times, "+".join(names), count))
    return results


def result_row(times, kernel):
    """The RESULT_COLUMNS (except GB/s) of one config's timed runs."""
    mean = statistics.fmean(times)
    return {
        "time_us": mean,
        "min_us": min(times),
        "rel_std_pct": statistics.stdev(times) / mean * 100 if len(times) > 1 else None,
        "kernel": kernel,
    }


def mxfp8_bytes(numel, input_dtype, direction, num_inputs=1):
    """Bytes an MXFP8 quantize moves: the input(s), plus fp8 data and e8m0 scales out per
    direction (dbias partial sums and scale padding not counted)."""
    num_outputs = sum(DIRECTIONS[direction])
    return num_inputs * numel * INPUTS[input_dtype][2] + num_outputs * (numel + numel // 32)


def cuda_backend(enable):
    """Sets NVTE_ENABLE_CUTEDSL_BACKEND for TE's own dispatch; call before importing TE."""
    os.environ["NVTE_ENABLE_CUTEDSL_BACKEND"] = "1" if enable else "0"


def disable_cutedsl_dispatch():
    """Makes TE's quantize calls run the CUDA kernels, even if TE was imported already."""
    # pylint: disable=import-outside-toplevel
    import ctypes

    from transformer_engine.common import _get_shared_object_file

    core = _get_shared_object_file("core")
    lib = ctypes.CDLL(str(core))
    if hasattr(lib, "nvte_set_cutedsl_backend"):
        lib.nvte_set_cutedsl_backend.argtypes = [ctypes.c_int]
        lib.nvte_set_cutedsl_backend(0)
    return core


# ---- command line shared by the quantize benchmarks ------------------------------------


def parse_list(parser, text, choices, option):
    """`a;b;c` (or `all`) -> list, each value checked against `choices`."""
    if text.strip().lower() == "all":
        return list(choices)
    values = [v.strip().lower() for v in text.split(";") if v.strip()]
    if not values or any(v not in choices for v in values):
        parser.error(
            f"{option} takes all or a ;-separated list of {', '.join(choices)}, got {text!r}"
        )
    return values


def parse_ints(parser, text, option):
    """`8;16` -> [8, 16]."""
    try:
        values = [int(v) for v in text.split(";") if v.strip()]
    except ValueError:
        values = []
    if not values:
        parser.error(f"{option} takes a ;-separated list of integers, got {text!r}")
    return values


def parse_shapes(text, block=MXFP8_BLOCK):
    """Parse "MxN;MxN;..." into [(M, N), ...]; M and N must be multiples of `block`."""
    shapes = []
    for item in filter(None, (s.strip() for s in text.split(";"))):
        match = re.fullmatch(r"(\d+)\s*[xX]\s*(\d+)", item)
        if match is None:
            raise ValueError(f"bad shape {item!r}, expected MxN")
        m, n = int(match[1]), int(match[2])
        if m % block or n % block:
            raise ValueError(f"{item}: M and N must be divisible by {block}")
        shapes.append((m, n))
    return shapes


def add_quantize_arguments(parser, fusions):
    """Options shared by the MXFP8 quantize benchmarks."""
    parser.add_argument(
        "--mode",
        type=str.lower,
        choices=["default", "manual", "debug"],
        default="default",
        help=(
            "default: built-in shape suite; manual: shapes from --shapes; debug: a few small"
            " shapes and few runs, to check the output quickly (default: default)"
        ),
    )
    parser.add_argument(
        "--direction", nargs="+", default=list(DIRECTIONS), choices=list(DIRECTIONS)
    )
    parser.add_argument(
        "--fusion",
        default="none",
        metavar="all|F;F;...",
        help=f"all, or a ;-separated subset of {';'.join(fusions)} (default: none)",
    )
    parser.add_argument(
        "--swizzle",
        type=str.lower,
        choices=["off", "on", "both"],
        default="off",
        help="scales in the GEMM-ready swizzled layout (default: off)",
    )
    parser.add_argument(
        "--input",
        default="bf16",
        metavar="DTYPE;...",
        help=f"input dtypes, a ;-separated subset of {';'.join(INPUTS)} (default: bf16)",
    )


def parse_quantize_arguments(parser, args, fusions):
    """Validates add_quantize_arguments' options; sets args.fusions/swizzles/inputs."""
    if (args.mode == "manual") != (args.shapes is not None):
        parser.error("--shapes is required by, and only valid with, --mode manual")
    args.runs = args.runs or (3 if args.mode == "debug" else 20)
    args.warmup = args.warmup or (1 if args.mode == "debug" else 5)
    args.fusions = parse_list(parser, args.fusion, fusions, "--fusion")
    args.swizzles = list(SWIZZLES) if args.swizzle == "both" else [args.swizzle]
    args.inputs = parse_list(parser, args.input, list(INPUTS), "--input")


def quantize_worker_arguments(args):
    """add_quantize_arguments' options for a compare worker."""
    worker = ["--mode", args.mode, *(["--shapes", args.shapes] if args.shapes else [])]
    worker += ["--direction", *args.direction, "--fusion", ";".join(args.fusions)]
    worker += ["--swizzle", args.swizzle, "--input", ";".join(args.inputs)]
    return worker


def quantize_heading(values, activation):
    """`bf16 · dact(gelu) · both · swizzled scale` for the shared quantize dimensions."""
    fusion = values.get("fusion", "none")
    parts = [values.get("input", "bf16"), f"{fusion}({activation})" if "act" in fusion else fusion]
    parts.append(values["direction"])
    if values.get("swizzle") == "on":
        parts.append("swizzled scale")
    return " · ".join(parts)


# ---- git, compare and regenerate -------------------------------------------------------


def git(repo, *args, check=True):
    """Runs `git -C repo args...`, capturing its output."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True, text=True
    )


def installed_te():
    """Where the installed TE, and so the CUDA kernels, come from."""
    probe = subprocess.run(
        [sys.executable, "-P", "-c", INSTALLED_TE_PROBE], check=True, capture_output=True, text=True
    )
    init, version, library = probe.stdout.strip().splitlines()[-3:]
    package = Path(init).parent
    sha = git(package, "rev-parse", "--short", "HEAD", check=False)
    branch = git(package, "branch", "--show-current", check=False).stdout.strip()
    source = (
        f"{branch or 'detached'} @ {sha.stdout.strip()}"
        if sha.returncode == 0
        else "not a checkout"
    )
    return {"package": package, "version": version, "library": library, "git": source}


@contextlib.contextmanager
def _workers(script, repo, worker_args):
    """Yields run(label, impl) -> DataFrame, which runs the benchmark in a subprocess.

    The subprocesses run a copy of the script and this module, so every run uses the same
    benchmark code, and it survives checking out a revision that does not contain them.
    """
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    tmpdir = Path(tempfile.mkdtemp(prefix="bench_quantize_"))
    copy = tmpdir / Path(script).name
    shutil.copy(script, copy)
    shutil.copy(__file__, tmpdir)

    def run(label, impl):
        csv = tmpdir / f"{label}.csv"
        cmd = [sys.executable, str(copy), "--repo-root", str(repo)]
        cmd += ["--impl", impl, *worker_args, "--worker-csv", str(csv)]
        subprocess.run(cmd, check=True)
        return pd.read_csv(csv).assign(revision=label)

    try:
        yield run
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def with_delta(frames, keys, columns, base="old", cand="new"):
    """Concatenates both runs and adds `delta`: % time reduction of `cand` vs `base`
    (positive = `cand` is faster), the same value on both rows of a config."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    df = pd.concat(frames, ignore_index=True)
    table = df.pivot_table(index=keys, columns="revision", values="time_us")[[base, cand]]
    table["delta"] = (table[base] - table[cand]) / table[base] * 100
    df = df.merge(table["delta"].reset_index(), on=keys)
    return df[["revision", *columns, "delta"]]


@contextlib.contextmanager
def _in_place_checkouts(repo):
    """Yields checkout(sha), which checks out a revision in the working tree, and checks out
    the original branch again on exit. Refuses to run with uncommitted changes, which would
    otherwise leak into every revision."""
    # Ignored: submodules (checkout leaves them alone) and the benchmark's own files (the runs
    # use a copy made before any checkout); neither is part of the benchmarked code.
    status = ["status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all", "--"]
    status.append(".")
    if HERE.is_relative_to(repo):
        status.append(f":!{HERE.relative_to(repo)}")
    dirty = git(repo, *status).stdout.rstrip()
    if dirty:
        sys.exit(f"--compare checks out revisions in place; commit or stash these first:\n{dirty}")
    head = git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    head = head or git(repo, "rev-parse", "HEAD").stdout.strip()

    def checkout(sha):
        result = git(repo, "checkout", "--quiet", sha, check=False)
        if result.returncode != 0:
            sys.exit(f"git checkout {sha[:12]} failed:\n{result.stderr.strip()}")

    try:
        yield checkout
    finally:
        git(repo, "checkout", "--quiet", head)
        print(f"Restored {head}")


def _compare(bench, script, repo, args, worker_args):
    """Runs the benchmark on args.old (a revision, or "CUDA") and args.new (a revision)."""
    old, new = args.old, args.new
    shas = {
        rev: git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").stdout.strip()
        for rev in (old, new)
        if rev != "CUDA"
    }
    explanation = ""
    frames = []
    with _in_place_checkouts(repo) as checkout, _workers(script, repo, worker_args) as run:
        if old == "CUDA":
            what = bench.title.removesuffix(" benchmark")
            explanation = CUDA_EXPLANATION.format(what=what, repo=repo, new=new, **installed_te())
            print(explanation, flush=True)
        for label, rev in (("old", old), ("new", new)):
            if rev == "CUDA":
                # Before any checkout: the installed TE may be this very source tree.
                print(f"=== {label}: CUDA (installed TE) ===", flush=True)
                frames.append(run(label, "CUDA"))
            else:
                print(f"=== {label}: CuTeDSL @ {rev} ({shas[rev][:12]}) ===", flush=True)
                checkout(shas[rev])
                frames.append(run(label, "CuTeDSL"))
    print(explanation)
    return with_delta(frames, bench.config, _columns(bench))


def _columns(bench):
    return ["impl", *bench.config, *RESULT_COLUMNS, *bench.extra]


def _load_results(folder):
    """(results, config fields, config text) of a folder written by an earlier run."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    folder = Path(folder)
    if not (folder / "results.csv").is_file():
        sys.exit(f"{folder} has no results.csv; pass a result folder written by this script")
    config = folder / "config.txt"
    text = config.read_text(encoding="utf-8") if config.is_file() else ""
    fields = {}
    for key, value in re.findall(r"^(\w+):\s+(.*)$", text, re.M):
        fields.setdefault(key, value)  # the top-level entry, not an embedded source config
    df = pd.read_csv(folder / "results.csv")
    if "swizzle" not in df.columns:  # written before --swizzle existed
        df.insert(df.columns.get_loc("fusion") + 1, "swizzle", "off")
    if "input" not in df.columns:  # written before --input existed
        df.insert(df.columns.get_loc("swizzle") + 1, "input", "bf16")
    return df, fields, text


def _regenerate_inputs(bench, args):
    """Loads earlier results instead of benchmarking. Returns (df, labels, activation, commit,
    sources) where sources lists (role, folder, config text) for config.txt."""
    if args.compare:
        runs = {}
        for role, folder in (("old", args.old), ("new", args.new)):
            df, fields, text = _load_results(folder)
            if "revision" in df.columns:
                sys.exit(f"--{role} {folder} holds a comparison; pass single-run result folders")
            series = fields.get("series", df["impl"].iloc[0])
            commit = fields.get("commit", "unknown").split()[0]
            label = f"{role}: {series} @ {commit} ({Path(folder).resolve().name})"
            runs[role] = (df.assign(revision=role), fields, text, label)
        df = with_delta([runs["old"][0], runs["new"][0]], bench.config, _columns(bench))
        labels = {role: run[3] for role, run in runs.items()}
        fields = runs["old"][1]
        sources = [(role, f, runs[role][2]) for role, f in (("old", args.old), ("new", args.new))]
    else:
        df, fields, text = _load_results(args.regenerate)
        series = fields.get("series", "").split(" | ")
        if "revision" in df.columns:
            labels = dict(zip(["old", "new"], series)) if len(series) == 2 else {}
            labels = labels or {"old": "old", "new": "new"}
        else:
            labels = {df["impl"].iloc[0]: series[0] or df["impl"].iloc[0]}
        sources = [("data", args.regenerate, text)]
    commit = fields.get("commit", "unknown").split()[0]
    return df, labels, fields.get("activation", "gelu"), commit, sources


# ---- output ----------------------------------------------------------------------------


def build_charts(bench, df, labels, activation="gelu"):
    """Renderer-agnostic chart specs: one line chart of time vs bench.x per value of every other
    config column.

    `labels` maps each series key (the `revision` column when comparing, else the `impl`
    column) to its legend name, in drawing order. Returns JSON-serializable dicts, so the
    same specs can feed matplotlib (png) or a web chart library (html). A series' `role`
    ("old" or "new"; a single run is "new") decides its look in every renderer.
    """
    key_col = "revision" if "revision" in df.columns else "impl"
    point_cols = [bench.x, "time_us", "min_us", "rel_std_pct", "GB/s", "kernel"]
    group_cols = [c for c in bench.config if c != bench.x]
    charts = []
    for values, rows in df.groupby(group_cols, sort=False):
        values = dict(zip(group_cols, (v.item() if hasattr(v, "item") else v for v in values)))
        rows = rows.sort_values(bench.x)
        series = []
        for key, name in labels.items():
            points = rows[rows[key_col] == key][point_cols]
            points = points.astype(object).where(points.notna(), None)  # NaN is not JSON
            records = [{"x": p[bench.x], **p} for p in points.to_dict("records")]
            series.append(
                {
                    "name": name,
                    "role": "old" if key == "old" else "new",
                    "x": points[bench.x].tolist(),
                    "y": points["time_us"].tolist(),
                    "points": records,
                }
            )
        subtitle = next(iter(labels.values())) if len(labels) == 1 else ""
        if "delta" in rows.columns:
            delta = rows.drop_duplicates(bench.x)["delta"]
            word = "faster" if delta.mean() >= 0 else "slower"
            subtitle = (
                f"new is {abs(delta.mean()):.1f}% {word} than old on average"
                f" (per {bench.x}: {delta.min():+.1f}% to {delta.max():+.1f}%, + = faster)"
            )
        heading = bench.heading(values, activation)
        split = f"{bench.split}={values[bench.split]}"
        charts.append(
            {
                **values,
                "id": re.sub(r"[^A-Za-z0-9=+_-]+", "_", f"{heading}_{split}").strip("_"),
                "heading": heading,
                "title": f"{heading} · {split}",
                "subtitle": subtitle,
                "x_label": bench.x,
                "y_label": "time (µs)",
                "x_scale": "log2",
                "y_scale": "log",
                "series": series,
            }
        )
    return charts


# Default palette's first two categorical slots (validated light-mode pair) plus its chrome.
CHART_STYLE = {
    "new": {"color": "#2a78d6", "linestyle": "-"},
    "old": {"color": "#eb6834", "linestyle": "--"},
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


def render_png(charts, out_dir):
    """Writes one png per chart spec from build_charts()."""
    # pylint: disable=import-outside-toplevel
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, LogLocator, NullLocator, ScalarFormatter

    st = CHART_STYLE
    out_dir.mkdir(parents=True, exist_ok=True)
    for chart in charts:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        fig.patch.set_facecolor(st["surface"])
        ax.set_facecolor(st["surface"])
        xs = sorted({x for s in chart["series"] for x in s["x"]})
        # Spread the end-of-line labels apart when the lines end close together.
        ends = sorted((s["y"][-1], i) for i, s in enumerate(chart["series"]) if s["y"])
        close = len(ends) > 1 and ends[-1][0] / ends[0][0] < 1.1
        nudge = {
            i: (rank - (len(ends) - 1) / 2) * 9 if close else 0 for rank, (_, i) in enumerate(ends)
        }
        for i, series in enumerate(chart["series"]):
            look = st[series["role"]]
            ax.plot(
                series["x"],
                series["y"],
                color=look["color"],
                linestyle=look["linestyle"],
                linewidth=2,
                marker="o",
                markersize=5,
                label=series["name"],
            )
            if series["x"] and len(chart["series"]) > 1:  # direct label at the line end
                ax.annotate(
                    series["role"],
                    (series["x"][-1], series["y"][-1]),
                    xytext=(6, nudge[i]),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=st["ink_secondary"],
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_locator(
            LogLocator(base=10, subs=(1, 2, 3, 4, 5, 6, 7, 8, 9), numticks=12)
        )
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_title(chart["title"], loc="left", fontsize=11, color=st["ink"], pad=18)
        ax.text(
            0,
            1.02,
            chart["subtitle"],
            transform=ax.transAxes,
            fontsize=8,
            color=st["ink_secondary"],
        )
        ax.set_xlabel(chart["x_label"], color=st["muted"], fontsize=9)
        ax.set_ylabel(chart["y_label"], color=st["muted"], fontsize=9)
        ax.tick_params(colors=st["muted"], labelsize=8, which="both")
        ax.grid(True, which="major", color=st["grid"], linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(st["axis"])
        if len(chart["series"]) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=st["ink_secondary"])
        fig.tight_layout()
        fig.savefig(out_dir / f"{chart['id']}.png", facecolor=st["surface"])
        plt.close(fig)


def render_html(charts, out_dir, meta):
    """Writes data.json (the specs from build_charts() plus `meta`) and index.html, an
    interactive page that also embeds the data so it opens straight from disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"meta": meta, "charts": charts})
    (out_dir / "data.json").write_text(data, encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/", data.replace("</", "<\\/"))
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def _gpu():
    """Name and driver of the GPU the benchmark runs on (first visible device)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    query = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    if visible.isdigit():
        query += ["-i", visible]
    try:
        out = subprocess.run(query, check=True, capture_output=True, text=True).stdout
        name, driver = out.splitlines()[0].split(", ")
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return "unknown"
    return f"{name} (driver {driver})"


def _write_config(bench, out_dir, args, repo, commit, labels, files):
    """Writes config.txt: what was benchmarked, on which commit and GPU."""

    def rev(name):
        sha = git(repo, "rev-parse", "--short", f"{name}^{{commit}}", check=False).stdout.strip()
        return f"CuTeDSL kernels @ {name} ({sha})"

    def installed():
        te = installed_te()
        return f"CUDA kernels of the installed TE: {te['library']} (version {te['version']})"

    branch = git(repo, "branch", "--show-current", check=False).stdout.strip() or "detached"
    lines = [
        bench.title,
        "",
        f"date:       {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"command:    {shlex.join(['python', *sys.argv])}",
        f"commit:     {commit} on {branch} ({repo})",
        f"gpu:        {_gpu()}",
    ]
    if args.compare:
        lines += [
            f"old:        {installed() if args.old == 'CUDA' else rev(args.old)}",
            f"new:        {rev(args.new)}",
        ]
    else:
        lines.append(f"impl:       {installed() if args.impl == 'CUDA' else rev('HEAD')}")
    lines.append(f"series:     {' | '.join(labels.values())}")
    lines += bench.describe(args)
    lines += [
        f"runs:       {args.runs} timed + {args.warmup} warm-up launches per config",
        (
            "timing:     CUPTI kernel timestamps (torch.profiler); back-to-back launches rotate"
            f" over buffer sets touching >{L2_ROTATION}x L2 between reuses (cold inputs, output"
            " write-backs included); clocks as set by the user"
        ),
        f"files:      {', '.join(files)}",
    ]
    (out_dir / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_regenerated_config(bench, out_dir, labels, activation, commit, sources, files):
    """Writes config.txt for --regenerate: where the data came from, plus the source configs."""
    lines = [
        f"{bench.title} (regenerated from existing results, nothing re-run)",
        "",
        f"date:       {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"command:    {shlex.join(['python', *sys.argv])}",
        f"commit:     {commit}",
        f"series:     {' | '.join(labels.values())}",
        f"activation: {activation}",
    ]
    lines += [f"{role + ':':11} {Path(folder).resolve()}" for role, folder, _ in sources]
    lines.append(f"files:      {', '.join(files)}")
    for role, folder, text in sources:
        lines += ["", f"----- {role}: config.txt of {Path(folder).resolve()} -----", text.rstrip()]
    (out_dir / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---- entry point -----------------------------------------------------------------------


def main(bench, script, doc):
    """Command-line entry point of a benchmark script (`script` is its __file__)."""
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    bench.add_arguments(parser)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run the shapes on --old and --new and compare them (clean tree needed)",
    )
    parser.add_argument(
        "--impl", type=str.lower, choices=list(IMPLS), help="when not comparing (default: cutedsl)"
    )
    parser.add_argument(
        "--new", metavar="REV|DIR", help="CuTeDSL revision to evaluate (default: HEAD)"
    )
    parser.add_argument("--old", metavar="REV|CUDA|DIR", help="baseline (default: main)")
    parser.add_argument(
        "--regenerate",
        nargs="?",
        const="",
        metavar="DIR",
        help=(
            "re-render earlier results in --format instead of benchmarking: --regenerate DIR,"
            " or --regenerate --compare --old DIR --new DIR to compare two single-run folders"
            " (e.g. two CUDA builds); benchmark options are ignored"
        ),
    )
    parser.add_argument("--runs", type=int, help="timed runs per config (default: 20; debug: 3)")
    parser.add_argument(
        "--warmup", type=int, help="untimed launches per config (default: 5; debug: 1)"
    )
    parser.add_argument(
        "--format",
        type=str.lower,
        choices=OUTPUT_FORMATS,
        default="csv",
        help=(
            "csv file, folder of png charts, interactive html page, or raw: print the table"
            " to the console and write nothing (default: csv)"
        ),
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        type=Path,
        metavar="DIR",
        help=(
            "directory to write into; each run creates a timestamped folder in it"
            " (default: .generated/ next to this script)"
        ),
    )
    parser.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-csv", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.regenerate is not None:
        if args.compare and (args.regenerate or not (args.old and args.new)):
            parser.error(
                "--regenerate --compare takes the two result folders as --old DIR --new DIR"
            )
        if not args.compare and not args.regenerate:
            parser.error("--regenerate needs a result folder (or --compare --old DIR --new DIR)")
    else:
        bench.parse_arguments(parser, args)
        if not args.compare and (args.old or args.new):
            parser.error("--old and --new are only valid with --compare")
        if args.compare:
            args.new = args.new or "HEAD"
        if args.compare and args.impl:
            parser.error(
                "--impl is not used when comparing: pass --old CUDA to compare against CUDA"
            )
        if args.new and args.new.lower() == "cuda":
            parser.error("CUDA can only be --old: other revisions of it would need a TE rebuild")
        args.old = "CUDA" if (args.old or "").lower() == "cuda" else (args.old or "main")
        args.impl = IMPLS[args.impl or "cutedsl"]
    if args.format == "raw" and args.output_path:
        parser.error("--format raw writes nothing, so --output_path is not used")

    repo = (args.repo_root or HERE.parents[3]).resolve()
    out_dir = None
    if args.format != "raw" and not args.worker_csv:
        base = args.output_path or HERE / ".generated"
        out_dir = base / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        while out_dir.exists():
            out_dir = out_dir.with_name(out_dir.name + "_")
        print(f"Results will be written to {out_dir.resolve()}/", flush=True)
    if args.regenerate is not None:
        df, labels, activation, sha, sources = _regenerate_inputs(bench, args)
    else:
        # Taken before comparing checks anything out.
        sha = git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
        status = ["status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"]
        if git(repo, *status).stdout.strip():
            sha += "-dirty"
        activation = getattr(args, "activation", None) or "gelu"
        if args.compare:
            old = "CUDA (installed TE)" if args.old == "CUDA" else f"CuTeDSL @ {args.old}"
            labels = {"old": f"old: {old}", "new": f"new: CuTeDSL @ {args.new}"}
            worker_args = bench.worker_arguments(args)
            worker_args += ["--runs", str(args.runs), "--warmup", str(args.warmup)]
            df = _compare(bench, script, repo, args, worker_args)
        else:
            labels = {args.impl: args.impl}
            # The CUDA kernels must come from the installed TE, so leave sys.path alone for them.
            if args.impl == "CuTeDSL":
                sys.path.insert(0, str(repo))
            df = bench.run(args, args.impl)
            df.insert(0, "impl", args.impl)

    if args.worker_csv:
        df.to_csv(args.worker_csv, index=False)
        return
    is_comparison = "revision" in df.columns
    if args.format == "raw":
        if is_comparison:
            table = df.pivot_table(index=bench.config, columns="revision", values="time_us")
            table = table[["old", "new"]]
            table["delta"] = df.groupby(bench.config)["delta"].first()
            table = table.rename(columns=lambda c: c if c == "delta" else f"{c}_us")
            print(table.to_string(float_format="%.2f"))
        else:
            print(df.to_string(index=False))
        return

    out_dir.mkdir(parents=True)
    if args.output_path is None:  # keep results out of git
        (out_dir.parent / ".gitignore").write_text("*\n", encoding="utf-8")
    files = ["results.csv"]
    if args.format in ("png", "html"):
        charts = build_charts(bench, df, labels, activation)
        if args.format == "png":
            render_png(charts, out_dir)
            files.append(f"{len(charts)} png charts")
        else:
            meta = {
                "title": bench.title,
                "description": " vs ".join(labels.values()),
                "compare": is_comparison,
                "commit": sha,
                "command": shlex.join(["python", *sys.argv]),
                "selectors": [{"key": k, "label": v} for k, v in bench.selectors.items()],
                "split": bench.split,
            }
            render_html(charts, out_dir, meta)
            files += ["index.html (open in a browser)", "data.json"]
    df.to_csv(out_dir / "results.csv", index=False)
    if args.regenerate is not None:
        _write_regenerated_config(
            bench, out_dir, labels, activation, sha, sources, files + ["config.txt"]
        )
    else:
        _write_config(bench, out_dir, args, repo, sha, labels, files + ["config.txt"])
    print(f"Results written to {out_dir.resolve()}/")
