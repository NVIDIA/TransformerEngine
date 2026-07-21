# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Autotune the CuTeDSL MXFP8 quantize kernels via ``cutlass.cute.testing.autotune_jit``.

Usage:
    python -m transformer_engine.common.CuTeDSL.cast.mxfp8.autotune bidim
    Use CUTE_DSL_LOG_AUTOTUNE=1 to see per-candidate timings while sweeping
"""

import argparse
import itertools
import os
import sys

import torch
from cuda.bindings.driver import CUstream

import cutlass
from cutlass import cute
from cutlass.cute import testing

from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernel,
    MXFP8QuantizeSpecializedRowwiseKernel,
    MXFP8QuantizeSpecializedBidimensionalKernel,
)

SHAPES = (
    # Square
    (4096, 4096),
    (8192, 8192),
    # Activations (tokens = batch * seq, hidden)
    (16384, 4096),
    (65536, 8192),
    # Weights / wgrad inputs (hidden, ffn), Llama3-8B / 70B MLP and wide-FFN
    (4096, 14336),
    (8192, 28672),
    (4096, 32768),
    # Small token count (decode / MoE expert slice)
    (512, 4096),
)
WARMUP_ITERATIONS = 10
ITERATIONS = 100


def make_entry(kernel_class, cfg, candidates):
    """Build the autotuned entry; the knobs enter as one Constexpr dict."""

    @testing.autotune_jit(
        params_dict={"tuneable_cfgs": candidates},
        update_on_change=["M", "N"],
        warmup_iterations=WARMUP_ITERATIONS,
        iterations=ITERATIONS,
    )
    @cute.jit
    def entry(
        mX,
        mO_row,
        mS_row,
        mO_col,
        mS_col,
        mAmax,
        mNoop,
        mDActInput,
        mWorkspace,
        stream,
        M,  # tuning-cache key only
        N,  # tuning-cache key only
        # cute.jit re-evaluates defaults outside the closure: literal only.
        # The tuner always supplies tuneable_cfgs; the default exists for arg binding.
        tuneable_cfgs: cutlass.Constexpr = None,
    ):
        kernel = kernel_class(cfg, tuneable_cfgs)  # trace-time construction
        kernel(mX, mO_row, mS_row, mO_col, mS_col, mAmax, mNoop, mDActInput, mWorkspace, stream)

    return entry


def _roundup(x, m):
    return (x + m - 1) // m * m


def make_args(cfg, M, N):
    """Argument list for one shape, mirroring TE's (padded) allocation contract."""
    in_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.DTYPE_STR]
    out_dtype = torch.float8_e4m3fn if cfg.FP8_DTYPE == "e4m3" else torch.float8_e5m2
    x = torch.randn(M, N, dtype=in_dtype, device="cuda")
    o_row = s_row = o_col = s_col = amax = act_in = workspace = None
    if cfg.ROWWISE:
        o_row = torch.empty(M, N, dtype=out_dtype, device="cuda")
        s_row = torch.zeros(
            _roundup(M, 128), _roundup(N // 32, 4), dtype=torch.float8_e8m0fnu, device="cuda"
        )
    if cfg.COLWISE:
        o_col = torch.empty(M, N, dtype=out_dtype, device="cuda")
        s_col = torch.zeros(
            _roundup(M // 32, 4), _roundup(N, 128), dtype=torch.float8_e8m0fnu, device="cuda"
        )
    if cfg.WITH_AMAX:
        amax = torch.zeros(1, dtype=torch.float32, device="cuda")
    if cfg.WITH_DACT:
        act_in = torch.randn(M, N, dtype=in_dtype, device="cuda")
    if cfg.WITH_DBIAS:
        # Workspace geometry depends on the candidate's tiling; allocate for the
        # chunk_rows == 32 lower bound so every candidate fits in a prefix.
        workspace = torch.zeros(_roundup(M, 32) // 32, N, dtype=torch.float32, device="cuda")
    stream = CUstream(torch.cuda.current_stream().cuda_stream)
    return [x, o_row, s_row, o_col, s_col, amax, None, act_in, workspace, stream]


def _cfg(direction="both", swizzle=False, combo="plain"):
    return MXFP8QuantizeConfig(
        dtype="bf16",
        fp8_dtype="e4m3",
        rowwise=direction in ("row", "both"),
        colwise=direction in ("col", "both"),
        with_gemm_swizzled_scales=swizzle,
        with_amax=False,
        with_dbias=combo == "dbias",
        with_dact=combo.startswith("d") and combo != "dbias",
        with_act=combo not in ("plain", "dbias") and not combo.startswith("d"),
        activation="none" if combo in ("plain", "dbias") else combo,
    )


def _sweep_case(kernel_class, cfg, candidates, case_name, winners):
    """Autotune one (kernel, cfg) case over all SHAPES; record winners."""
    print(f"\n### {case_name}: {cfg} ({len(candidates)} candidates)")
    entry = make_entry(kernel_class, cfg, candidates)
    for M, N in SHAPES:
        entry(*make_args(cfg, M, N), M, N)  # first call per (M, N) autotunes
        torch.cuda.synchronize()
        best = entry._best_config[(M, N)]["tuneable_cfgs"]
        print(f"== {case_name} {M}x{N}: {best}")
        winners[(case_name, M, N)] = best


# ---------------------------------------------------------------------------
# Suites: one per kernel. Candidate dicts must be complete and jointly valid
# (an invalid one that raises during compile ABORTS the whole sweep).
# ---------------------------------------------------------------------------


def run_general_suite():
    """The general kernel serves every fused/swizzled combo. The standard-tile
    and dbias-only-tile knobs are disjoint (the cfg picks one pair), so each
    case only sweeps the pair it actually uses, keeping the other at default."""
    standard_candidates = [
        {
            "_NUM_STAGES": stages,
            "_NUM_TILES_STANDARD": tiles,
            "_NUM_TILES_DBIAS_ONLY": 4,
            "_THREADS_PER_CTA_STANDARD": threads,
            "_THREADS_PER_CTA_DBIAS_ONLY": 128,
        }
        for stages, tiles, threads in itertools.product((2, 3), (2, 4, 8), (64, 128))
    ]
    dbias_candidates = [
        {
            "_NUM_STAGES": stages,
            "_NUM_TILES_STANDARD": 2,
            "_NUM_TILES_DBIAS_ONLY": tiles,
            "_THREADS_PER_CTA_STANDARD": 64,
            "_THREADS_PER_CTA_DBIAS_ONLY": threads,
        }
        for stages, tiles, threads in itertools.product((2, 3), (2, 4, 8), (64, 128))
    ]
    cases = [
        ("cast_2d_swizzled", _cfg(swizzle=True), standard_candidates),
        ("cast_colwise", _cfg(direction="col"), standard_candidates),
        ("gelu_2d", _cfg(combo="gelu"), standard_candidates),
        ("dgelu_2d", _cfg(combo="dgelu"), standard_candidates),
        ("dbias_2d", _cfg(combo="dbias"), dbias_candidates),
    ]
    winners = {}
    for case_name, cfg, candidates in cases:
        _sweep_case(MXFP8QuantizeKernel, cfg, candidates, case_name, winners)
    return winners


def run_rowwise_suite():
    """The rowwise specialized kernel only serves plain non-swizzled row-only cast."""
    candidates = [
        {"_TILE_ROWS": rows, "_TILE_COLS": cols, "_STASH_SCALE_TO_SMEM": stash}
        for rows, cols, stash in itertools.product((4, 8, 16), (512, 1024, 2048), (True, False))
        if 32 <= rows * cols // 32 <= 1024  # one thread per scale block, CTA-size limits
    ]
    winners = {}
    _sweep_case(
        MXFP8QuantizeSpecializedRowwiseKernel,
        _cfg(direction="row"),
        candidates,
        "cast_rowwise",
        winners,
    )
    return winners


def run_bidim_suite():
    """The bidim specialized kernel only serves plain non-swizzled rowwise+colwise cast."""
    candidates = [
        {"_NUM_TILES_X": x, "_NUM_TILES_Y": y, "_NUM_STAGES": stages}
        for x, y, stages in itertools.product((2, 4, 8, 16), (1, 2, 4), (2, 3, 4, 5))
        if x * y >= stages  # a span shallower than the pipeline wastes stages
    ]
    winners = {}
    _sweep_case(
        MXFP8QuantizeSpecializedBidimensionalKernel, _cfg(), candidates, "cast_2d", winners
    )
    return winners


SUITES = {
    "general": run_general_suite,
    "rowwise": run_rowwise_suite,
    "bidim": run_bidim_suite,
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", choices=sorted(SUITES))
    args = p.parse_args()

    dev = torch.cuda.get_device_properties(0)
    print(f"# device={dev.name} cc={dev.major}.{dev.minor}"
          f" arch={os.environ.get('CUTE_DSL_ARCH', '<auto>')}")
    print(f"# preset={args.preset}")

    winners = SUITES[args.preset]()

    print("\n# winners:")
    for (case_name, M, N), best in winners.items():
        print(f"#   {case_name} {M}x{N}: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
