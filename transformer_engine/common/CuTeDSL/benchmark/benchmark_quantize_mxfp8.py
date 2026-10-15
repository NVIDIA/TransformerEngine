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

import argparse
import contextlib
import ctypes
import datetime
import functools
import itertools
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# Hidden sizes of current frontier models: gpt-oss (2880, also the non-128-aligned case),
# Qwen3-235B (4096), Llama 4 / Qwen3-32B / GLM-4.5 (5120), DeepSeek-V3 / Kimi K2 (7168),
# Llama 3 70B (8192).
DEFAULT_N = [2880, 4096, 5120, 7168, 8192]
DEFAULT_M = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_SHAPES = list(itertools.product(DEFAULT_M, DEFAULT_N))
# Smallest set that still exercises every output: two M (a line per chart), two N (one of
# them not 128-aligned).
DEBUG_SHAPES = list(itertools.product([1024, 4096], [2880, 4096]))
MXFP8_BLOCK = 32
DIRECTIONS = {"rowwise": (True, False), "colwise": (False, True), "both": (True, True)}
OUTPUT_FORMATS = ("csv", "png", "html", "raw")
IMPLS = {"cutedsl": "CuTeDSL", "cuda": "CUDA"}
FUSIONS = ("none", "act", "dact", "dbias", "dactbias")
SWIZZLES = ("off", "on")  # scales in the GEMM-ready swizzled layout
# --input name -> (torch dtype name, CuTeDSL dtype name, bytes per element)
INPUTS = {
    "bf16": ("bfloat16", "BFloat16", 2),
    "fp16": ("float16", "Float16", 2),
    "fp32": ("float32", "Float32", 4),
}
ACTIVATIONS = ("gelu", "relu", "silu", "qgelu", "srelu")
COLUMNS = [
    "impl",
    "M",
    "N",
    "direction",
    "fusion",
    "swizzle",
    "input",
    "time_us",
    "min_us",
    "rel_std_pct",
    "GB/s",
    "kernel",
]
CUDA_EXPLANATION = """\
Comparing CUDA (old) vs CuTeDSL (new) MXFP8 quantize kernels:
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


def _installed_te():
    """Where the installed TE, and so the CUDA kernels, come from."""
    probe = subprocess.run(
        [sys.executable, "-P", "-c", INSTALLED_TE_PROBE], check=True, capture_output=True, text=True
    )
    init, version, library = probe.stdout.strip().splitlines()[-3:]
    package = Path(init).parent
    sha = _git(package, "rev-parse", "--short", "HEAD", check=False)
    branch = _git(package, "branch", "--show-current", check=False).stdout.strip()
    git = (
        f"{branch or 'detached'} @ {sha.stdout.strip()}"
        if sha.returncode == 0
        else "not a checkout"
    )
    return {"package": package, "version": version, "library": library, "git": git}


def l2_cache_bytes(device=None):
    """L2 cache size in bytes of `device` (the current CUDA device by default)."""
    import torch  # pylint: disable=import-outside-toplevel

    device = torch.cuda.current_device() if device is None else device
    return torch.cuda.get_device_properties(device).L2_cache_size


def make_l2_flush(device=None, multiple=2):
    """Returns flush(), which evicts L2 by reading a buffer `multiple` times its size; call it
    before each timed launch so the kernel reads its inputs from DRAM.

    It must be a read: a write (e.g. zero_) leaves L2 full of dirty lines, and the next kernel
    pays for writing them back (measured +10% on some MXFP8 quantize kernels).
    """
    import torch  # pylint: disable=import-outside-toplevel

    device = torch.cuda.current_device() if device is None else device
    buffer = torch.ones(multiple * l2_cache_bytes(device) // 4, dtype=torch.float32, device=device)
    return lambda: buffer.sum()


def _round_up(x, y):
    return (x + y - 1) // y * y


def gbps(t_us, m, n, direction, fusion, input_dtype):
    """Bytes moved (input(s), fp8 + e8m0 scales out per direction) per ns, i.e. GB/s."""
    num_inputs = 2 if "dact" in fusion else 1
    num_outputs = sum(DIRECTIONS[direction])
    nbytes = num_inputs * m * n * INPUTS[input_dtype][2] + num_outputs * (m * n + m * n // 32)
    return nbytes / (t_us * 1e3)


def _cutedsl_launcher(activation):
    """Returns setup(x, act_input, direction, fusion, swizzle) -> launch(), calling the compiled CuTeDSL
    kernel directly."""
    # pylint: disable=import-outside-toplevel
    import torch

    # Not needed for the direct call below, but keeps TE's own dispatch consistent with --impl.
    os.environ["NVTE_ENABLE_CUTEDSL_BACKEND"] = "1"
    from transformer_engine.common.CuTeDSL import get_mxfp8_quantization_function

    # Resolve the defining module instead of naming it: it was renamed between revisions.
    qc = sys.modules[get_mxfp8_quantization_function.__module__]

    print(f"CuTeDSL kernels from {qc.__file__}")

    @functools.lru_cache(maxsize=None)
    def compiled(dtype, direction, fusion, swizzle):
        rowwise, colwise = DIRECTIONS[direction]
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
        rowwise, colwise = DIRECTIONS[direction]
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
        dtype = next(v[1] for v in INPUTS.values() if getattr(torch, v[0]) == x.dtype)
        fn = compiled(dtype, direction, fusion, swizzle)
        stream = torch.cuda.current_stream().cuda_stream
        outs = (o_row, s_row, o_col, s_col)
        return lambda: fn(x, *outs, None, None, act_input, workspace, stream)

    return setup


def _cuda_launcher(activation):
    """Returns setup(x, act_input, direction, fusion, swizzle) -> launch(), running the installed TE's
    CUDA kernel."""
    # pylint: disable=import-outside-toplevel
    os.environ["NVTE_ENABLE_CUTEDSL_BACKEND"] = "0"
    import transformer_engine.pytorch  # pylint: disable=unused-import
    import transformer_engine_torch as tex
    from transformer_engine.common import _get_shared_object_file
    from transformer_engine.pytorch import MXFP8Quantizer

    core = _get_shared_object_file("core")
    print(f"CUDA kernels from {core}")
    lib = ctypes.CDLL(str(core))
    if hasattr(lib, "nvte_set_cutedsl_backend"):  # in case TE was imported before the env var
        lib.nvte_set_cutedsl_backend.argtypes = [ctypes.c_int]
        lib.nvte_set_cutedsl_backend(0)

    def setup(x, act_input, direction, fusion, swizzle):
        rowwise, colwise = DIRECTIONS[direction]
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


def _base_name(kernel):
    """`void ns::quantize_mxfp8_kernel<...>(...)` -> `quantize_mxfp8_kernel`."""
    return re.split(r"[<(]", kernel.removeprefix("void "), maxsplit=1)[0].split("::")[-1]


def _short_name(base):
    """CuTeDSL kernel symbols encode the module path and arguments; keep only the class."""
    match = re.search(r"(MXFP8Quantize\w*?Kernel)", base)
    return match.group(1) if match else base


def run_suite(shapes, inputs, directions, fusions, swizzles, activation, runs, warmup, impl):
    """Times the `impl` quantize kernel for every shape, direction, fusion and swizzle with
    CUPTI kernel timestamps (via torch.profiler); returns a DataFrame."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import torch
    from torch.profiler import ProfilerActivity, profile

    setup = (_cuda_launcher if impl == "CUDA" else _cutedsl_launcher)(activation)
    # Helper kernels such as scale-padding zeroing or the dbias reduction are not timed.
    prefix = "quantize" if impl == "CUDA" else "kernel_cutlass"

    @functools.lru_cache(maxsize=1)
    def make_inputs(m, n, input_dtype):
        """The input (or incoming gradient) and, for dact, the forward activation input."""
        dtype = getattr(torch, INPUTS[input_dtype][0])
        return [torch.randn((m, n), dtype=dtype, device="cuda") for _ in range(2)]

    flush_l2 = make_l2_flush()
    configs = [
        (m, n, inp, d, f, sw)
        for (m, n) in shapes
        for inp in inputs
        for d in directions
        for f in fusions
        for sw in swizzles
    ]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for m, n, input_dtype, direction, fusion, swizzle in configs:
            launch = setup(*make_inputs(m, n, input_dtype), direction, fusion, swizzle)
            for _ in range(warmup):  # JIT compile and first-launch costs stay out of the timing
                launch()
            for _ in range(runs):
                flush_l2()
                launch()
        torch.cuda.synchronize()

    kernels = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    kernels = [(base, e.device_time_total) for e in kernels if (base := _base_name(e.name))]
    kernels = [(base, t) for base, t in kernels if base.startswith(prefix)]
    per_config = warmup + runs
    if len(kernels) != per_config * len(configs):
        names = sorted({_short_name(base) for base, _ in kernels})
        raise RuntimeError(
            f"Expected {per_config} {impl} quantize kernels per config, got"
            f" {len(kernels) / len(configs):.2f} (kernels: {names}); did it fall back to"
            " another implementation or launch extra kernels?"
        )
    rows = []
    for i, (m, n, input_dtype, direction, fusion, swizzle) in enumerate(configs):
        timed = kernels[i * per_config + warmup : (i + 1) * per_config]
        times = [t for _, t in timed]
        mean = statistics.fmean(times)
        rows.append(
            {
                "impl": impl,
                "M": m,
                "N": n,
                "direction": direction,
                "fusion": fusion,
                "swizzle": swizzle,
                "input": input_dtype,
                "time_us": mean,
                "min_us": min(times),
                "rel_std_pct": statistics.stdev(times) / mean * 100 if runs > 1 else None,
                "GB/s": gbps(mean, m, n, direction, fusion, input_dtype),
                "kernel": _short_name(timed[-1][0]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True, text=True
    )


@contextlib.contextmanager
def _workers(repo, worker_args):
    """Yields run(label, impl) -> DataFrame, which runs the benchmark in a subprocess.

    The subprocesses run a copy of this script, so every run uses the same benchmark code,
    and it survives checking out a revision that does not contain this file.
    """
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    tmpdir = Path(tempfile.mkdtemp(prefix="bench_mxfp8_"))
    script = tmpdir / Path(__file__).name
    shutil.copy(__file__, script)

    def run(label, impl):
        csv = tmpdir / f"{label}.csv"
        cmd = [sys.executable, str(script), "--repo-root", str(repo)]
        cmd += ["--impl", impl, *worker_args, "--worker-csv", str(csv)]
        subprocess.run(cmd, check=True)
        return pd.read_csv(csv).assign(revision=label)

    try:
        yield run
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _with_delta(frames, base, cand):
    """Concatenates both runs and adds `delta`: % time reduction of `cand` vs `base`
    (positive = `cand` is faster), the same value on both rows of a config."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd

    keys = ["M", "N", "direction", "fusion", "swizzle", "input"]
    df = pd.concat(frames, ignore_index=True)
    table = df.pivot_table(index=keys, columns="revision", values="time_us")[[base, cand]]
    table["delta"] = (table[base] - table[cand]) / table[base] * 100
    df = df.merge(table["delta"].reset_index(), on=keys)
    return df[["revision", *COLUMNS, "delta"]]


@contextlib.contextmanager
def _in_place_checkouts(repo):
    """Yields checkout(sha), which checks out a revision in the working tree, and checks out
    the original branch again on exit. Refuses to run with uncommitted changes, which would
    otherwise leak into every revision."""
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.rstrip()
    if dirty:
        sys.exit(f"--compare checks out revisions in place; commit or stash these first:\n{dirty}")
    head = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    head = head or _git(repo, "rev-parse", "HEAD").stdout.strip()

    def checkout(sha):
        result = _git(repo, "checkout", "--quiet", sha, check=False)
        if result.returncode != 0:
            sys.exit(f"git checkout {sha[:12]} failed:\n{result.stderr.strip()}")

    try:
        yield checkout
    finally:
        _git(repo, "checkout", "--quiet", head)
        print(f"Restored {head}")


def compare(repo, old, new, worker_args):
    """Run the benchmark on `old` (a revision, or "CUDA") and `new` (a revision)."""
    shas = {
        rev: _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").stdout.strip()
        for rev in (old, new)
        if rev != "CUDA"
    }
    explanation = ""
    frames = []
    with _in_place_checkouts(repo) as checkout, _workers(repo, worker_args) as run:
        if old == "CUDA":
            explanation = CUDA_EXPLANATION.format(repo=repo, new=new, **_installed_te())
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
    return _with_delta(frames, "old", "new")


def build_charts(df, labels, activation="gelu"):
    """Renderer-agnostic chart specs: one line chart of time vs M per (fusion, direction,
    swizzle, input, N).

    `labels` maps each series key (the `revision` column when comparing, else the `impl`
    column) to its legend name, in drawing order. Returns JSON-serializable dicts, so the
    same specs can feed matplotlib (png) or a web chart library (html). A series' `role`
    ("old" or "new"; a single run is "new") decides its look in every renderer.
    """
    key_col = "revision" if "revision" in df.columns else "impl"
    point_cols = ["M", "time_us", "min_us", "rel_std_pct", "GB/s", "kernel"]
    charts = []
    for fusion, direction, swizzle, input_dtype in itertools.product(
        FUSIONS, DIRECTIONS, SWIZZLES, INPUTS
    ):
        group = df[
            (df["fusion"] == fusion)
            & (df["direction"] == direction)
            & (df["swizzle"] == swizzle)
            & (df["input"] == input_dtype)
        ]
        for n in sorted(group["N"].unique()):
            rows = group[group["N"] == n].sort_values("M")
            series = []
            for key, name in labels.items():
                points = rows[rows[key_col] == key][point_cols]
                points = points.astype(object).where(points.notna(), None)  # NaN is not JSON
                series.append(
                    {
                        "name": name,
                        "role": "old" if key == "old" else "new",
                        "x": points["M"].tolist(),
                        "y": points["time_us"].tolist(),
                        "points": points.to_dict("records"),
                    }
                )
            subtitle = next(iter(labels.values())) if len(labels) == 1 else ""
            if "delta" in rows.columns:
                delta = rows.drop_duplicates("M")["delta"]
                word = "faster" if delta.mean() >= 0 else "slower"
                subtitle = (
                    f"new is {abs(delta.mean()):.1f}% {word} than old on average"
                    f" (per M: {delta.min():+.1f}% to {delta.max():+.1f}%, + = faster)"
                )
            charts.append(
                {
                    "id": (
                        f"{input_dtype}_{fusion}_{direction}"
                        f"{'_swizzled' if swizzle == 'on' else ''}_N{n}"
                    ),
                    "fusion": fusion,
                    "direction": direction,
                    "swizzle": swizzle,
                    "input": input_dtype,
                    "N": int(n),
                    "title": " · ".join(
                        [input_dtype, f"{fusion}({activation})" if "act" in fusion else fusion]
                        + [direction]
                        + (["swizzled scale"] if swizzle == "on" else [])
                        + [f"N={n}"]
                    ),
                    "subtitle": subtitle,
                    "x_label": "M",
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
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
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
    template = Path(__file__).with_name("benchmark_quantization_template.html")
    template = template.read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/", data.replace("</", "<\\/"))
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def _gpu():
    """Name and driver of the GPU the benchmark runs on (first visible device)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    query = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    if visible.isdigit():
        query += ["-i", visible]
    try:
        name, driver = (
            subprocess.run(query, check=True, capture_output=True, text=True)
            .stdout.splitlines()[0]
            .split(", ")
        )
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return "unknown"
    return f"{name} (driver {driver})"


def _write_config(out_dir, args, fusions, swizzles, inputs, shapes, repo, commit, labels, files):
    """Writes config.txt: what was benchmarked, on which commit and GPU."""

    def rev(name):
        sha = _git(repo, "rev-parse", "--short", f"{name}^{{commit}}", check=False).stdout.strip()
        return f"CuTeDSL kernels @ {name} ({sha})"

    def installed():
        te = _installed_te()
        return f"CUDA kernels of the installed TE: {te['library']} (version {te['version']})"

    if args.mode != "default":
        shape_text = f"{args.mode}: " + "; ".join(f"{m}x{n}" for m, n in shapes)
    else:
        shape_text = (
            f"default suite, M = {', '.join(map(str, DEFAULT_M))}"
            f" x N = {', '.join(map(str, DEFAULT_N))} ({len(DEFAULT_SHAPES)} shapes)"
        )
    branch = _git(repo, "branch", "--show-current", check=False).stdout.strip() or "detached"
    lines = [
        "MXFP8 quantize kernel benchmark",
        "",
        f"date:       {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"command:    {shlex.join(['python', *sys.argv])}",
        f"commit:     {commit} on {branch} ({repo})",
        f"gpu:        {_gpu()}",
        f"mode:       {args.mode}",
    ]
    if args.compare:
        lines += [
            f"old:        {installed() if args.old == 'CUDA' else rev(args.old)}",
            f"new:        {rev(args.new)}",
        ]
    else:
        lines.append(f"impl:       {installed() if args.impl == 'CUDA' else rev('HEAD')}")
    lines += [
        f"series:     {' | '.join(labels.values())}",
        f"shapes:     {shape_text}",
        f"directions: {', '.join(args.direction)}",
        f"fusions:    {', '.join(fusions)}",
        f"swizzle:    {', '.join(swizzles)}",
        f"input:      {', '.join(inputs)}",
    ]
    if {"act", "dact", "dactbias"} & set(fusions):
        lines.append(f"activation: {args.activation or 'gelu'}")
    lines += [
        f"runs:       {args.runs} profiled + {args.warmup} warm-up launches per config",
        (
            "timing:     CUPTI kernel timestamps (torch.profiler), L2 flushed (read) before each"
            " run, clocks as set by the user"
        ),
        f"files:      {', '.join(files)}",
    ]
    (out_dir / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_results(folder):
    """(results, config fields, config text) of a folder written by an earlier run."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd

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


def _regenerate_inputs(args):
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
        df = _with_delta([runs["old"][0], runs["new"][0]], "old", "new")
        labels = {role: run[3] for role, run in runs.items()}
        fields = runs["old"][1]
        sources = [
            (role, folder, runs[role][2]) for role, folder in (("old", args.old), ("new", args.new))
        ]
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


def _write_regenerated_config(out_dir, labels, activation, commit, sources, files):
    """Writes config.txt for --regenerate: where the data came from, plus the source configs."""
    lines = [
        "MXFP8 quantize kernel benchmark (regenerated from existing results, nothing re-run)",
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


def parse_shapes(text):
    """Parse "MxN;MxN;..." into [(M, N), ...]."""
    shapes = []
    for item in filter(None, (s.strip() for s in text.split(";"))):
        match = re.fullmatch(r"(\d+)\s*[xX]\s*(\d+)", item)
        if match is None:
            raise ValueError(f"bad shape {item!r}, expected MxN")
        m, n = int(match[1]), int(match[2])
        if m % MXFP8_BLOCK or n % MXFP8_BLOCK:
            raise ValueError(
                f"{item}: PyTorch only quantizes MXFP8 tensors with M and N divisible by"
                f" {MXFP8_BLOCK}"
            )
        shapes.append((m, n))
    return shapes


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        type=str.lower,
        choices=["default", "manual", "debug"],
        default="default",
        help=(
            "default: built-in shape suite; manual: shapes from --shapes; debug: 4 small shapes"
            " and 1 run, to check the output quickly (default: default)"
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run the shapes on --old and --new and compare them (clean tree needed)",
    )
    parser.add_argument("--shapes", metavar='"MxN;..."', help="M, N multiples of 32")
    parser.add_argument(
        "--impl",
        type=str.lower,
        choices=list(IMPLS),
        help="implementation when not comparing (default: cutedsl)",
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
    parser.add_argument(
        "--direction", nargs="+", default=list(DIRECTIONS), choices=list(DIRECTIONS)
    )
    parser.add_argument(
        "--fusion",
        default="none",
        metavar="all|F;F;...",
        help=f"all, or a ;-separated subset of {';'.join(FUSIONS)} (default: none)",
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
    parser.add_argument(
        "--activation", choices=ACTIVATIONS, help="only with act/dact/dactbias (default: gelu)"
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
        if (args.mode == "manual") != (args.shapes is not None):
            parser.error("--shapes is required by, and only valid with, --mode manual")
        try:
            if args.mode == "manual":
                shapes = parse_shapes(args.shapes)
            else:
                shapes = DEBUG_SHAPES if args.mode == "debug" else DEFAULT_SHAPES
        except ValueError as e:
            parser.error(str(e))
        args.runs = args.runs or (3 if args.mode == "debug" else 20)
        args.warmup = args.warmup or (1 if args.mode == "debug" else 5)
        if args.fusion.strip().lower() == "all":
            fusions = list(FUSIONS)
        else:
            fusions = [f.strip().lower() for f in args.fusion.split(";") if f.strip()]
        if not fusions or any(f not in FUSIONS for f in fusions):
            parser.error(
                f"--fusion takes all or a ;-separated list of {', '.join(FUSIONS)}, got"
                f" {args.fusion!r}"
            )
        swizzles = list(SWIZZLES) if args.swizzle == "both" else [args.swizzle]
        inputs = [i.strip().lower() for i in args.input.split(";") if i.strip()]
        if not inputs or any(i not in INPUTS for i in inputs):
            parser.error(
                f"--input takes a ;-separated list of {', '.join(INPUTS)}, got {args.input!r}"
            )
        if args.activation and not {"act", "dact", "dactbias"} & set(fusions):
            parser.error("--activation only applies to the act, dact and dactbias fusions")
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

    repo = (args.repo_root or Path(__file__).resolve().parents[4]).resolve()
    out_dir = None
    if args.format != "raw" and not args.worker_csv:
        base = args.output_path or Path(__file__).resolve().parent / ".generated"
        out_dir = base / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        while out_dir.exists():
            out_dir = out_dir.with_name(out_dir.name + "_")
        print(f"Results will be written to {out_dir.resolve()}/", flush=True)
    if args.regenerate is not None:
        df, labels, activation, sha, sources = _regenerate_inputs(args)
    else:
        # Taken before comparing checks anything out.
        sha = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
        if _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
            sha += "-dirty"
        activation = args.activation or "gelu"
        if args.compare:
            old = "CUDA (installed TE)" if args.old == "CUDA" else f"CuTeDSL @ {args.old}"
            labels = {"old": f"old: {old}", "new": f"new: CuTeDSL @ {args.new}"}
            worker_args = ["--mode", args.mode, *(["--shapes", args.shapes] if args.shapes else [])]
            worker_args += ["--runs", str(args.runs), "--warmup", str(args.warmup)]
            worker_args += ["--direction", *args.direction]
            worker_args += ["--fusion", ";".join(fusions), "--swizzle", args.swizzle]
            worker_args += ["--input", ";".join(inputs)]
            worker_args += ["--activation", args.activation] if args.activation else []
            df = compare(repo, args.old, args.new, worker_args)
        else:
            labels = {args.impl: args.impl}
            # The CUDA kernels must come from the installed TE, so leave sys.path alone for them.
            if args.impl == "CuTeDSL":
                sys.path.insert(0, str(repo))
            df = run_suite(
                shapes,
                inputs,
                args.direction,
                fusions,
                swizzles,
                args.activation or "gelu",
                args.runs,
                args.warmup,
                args.impl,
            )

    if args.worker_csv:
        df.to_csv(args.worker_csv, index=False)
        return
    is_comparison = "revision" in df.columns
    if args.format == "raw":
        if is_comparison:
            keys = ["M", "N", "direction", "fusion", "swizzle", "input"]
            table = df.pivot_table(index=keys, columns="revision", values="time_us")[["old", "new"]]
            table["delta"] = df.groupby(keys)["delta"].first()
            print(
                table.rename(columns=lambda c: c if c == "delta" else f"{c}_us").to_string(
                    float_format="%.2f"
                )
            )
        else:
            print(df.to_string(index=False))
        return

    out_dir.mkdir(parents=True)
    if args.output_path is None:  # keep results out of git
        (out_dir.parent / ".gitignore").write_text("*\n", encoding="utf-8")
    files = ["results.csv"]
    if args.format in ("png", "html"):
        charts = build_charts(df, labels, activation)
        if args.format == "png":
            render_png(charts, out_dir)
            files.append(f"{len(charts)} png charts (<fusion>_<direction>_N<N>.png)")
        else:
            meta = {
                "title": "MXFP8 quantize kernel benchmark",
                "description": " vs ".join(labels.values()),
                "compare": is_comparison,
                "activation": activation,
                "commit": sha,
                "command": shlex.join(["python", *sys.argv]),
            }
            render_html(charts, out_dir, meta)
            files += ["index.html (open in a browser)", "data.json"]
    df.to_csv(out_dir / "results.csv", index=False)
    if args.regenerate is not None:
        _write_regenerated_config(out_dir, labels, activation, sha, sources, files + ["config.txt"])
    else:
        _write_config(
            out_dir,
            args,
            fusions,
            swizzles,
            inputs,
            shapes,
            repo,
            sha,
            labels,
            files + ["config.txt"],
        )
    print(f"Results written to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
