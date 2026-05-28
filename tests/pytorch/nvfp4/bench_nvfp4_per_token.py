# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bench NVFP4 per-token K1+K2 quant vs per-tensor RHT+SR baseline.

Quant-only (no GEMM). bf16, M % 128 == 0, K % 128 == 0.

Modes:
  * default: 2-way composite (per-token vs per-tensor). Ratio = pt / pten.
  * ``--rht``: 3-way composite (adds per-token + col-wise 16-pt RHT). Ratio =
    per-token (+rht) / per-tensor.
  * ``--k1-only``: K1 in isolation. Without ``--rht``: pt_K1 vs prod_K1.
    With ``--rht``: (A) pt_K1 vs pt_K1+RHT (apples-to-apples) and
    (B) pt_K1+RHT vs prod_K1 (NOT apples-to-apples; output shapes differ).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

# Import transformer_engine first so libtransformer_engine.so is dlopen'd
# before transformer_engine_torch tries to resolve its typeinfo symbols.
import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer


def cuda_time_ms(fn: Callable[[], None], *, warmup: int = 5, iters: int = 50) -> float:
    """Median wall time of fn over iters invocations, in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    return statistics.median(samples)


def cuda_graph_time_ms(fn: Callable[[], object], *, warmup: int = 5, iters: int = 50) -> float:
    """Median g.replay() wall time of fn captured into a CUDA Graph (kernel-only floor).

    Returns nan if capture fails.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
    except Exception as e:
        print(f"  [graph capture skipped: {type(e).__name__}: {e}]", file=sys.stderr)
        return float("nan")

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    return statistics.median(samples)


def _make_baseline_quantizer() -> NVFP4Quantizer:
    """Per-tensor baseline quantizer: RHT + SR + random sign mask."""
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=True,
        with_random_sign_mask=True,
    )


def _has_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


@dataclass
class ShapeBench:
    M: int
    K: int
    t_pt: float  # per-token full K1+K2, no RHT (Eager pybind, ms)
    t_pt_rht: float  # per-token full K1+K2, +RHT col-wise (Eager pybind, ms)
    t_pten: float  # per-tensor full K1+K2 with RHT+SR (Eager pybind, ms)
    t_pt_g: float  # per-token under CUDA Graphs replay (ms)
    t_pt_rht_g: float  # per-token+RHT under CUDA Graphs replay (ms)
    t_pten_g: float  # per-tensor under CUDA Graphs replay (ms)


@dataclass
class K1ShapeBench:
    M: int
    K: int
    # K1-only timings: 3 paths x 2 modes (Eager + CUDA Graphs).
    t_pt: float  # per-token K1, no RHT  (rowwise+columnwise amax vectors)
    t_pt_rht: float  # per-token K1, +RHT on col direction
    t_prod: float  # prod K1 hadamard_transform_amax (per-tensor scalar amax)
    t_pt_g: float
    t_pt_rht_g: float
    t_prod_g: float


# Default mask seed; matches prod's `te-nvfp4-build-overrides.mdc` convention.
_RHT_MASK_DEFAULT: int = 0xACE1


def _bench_shape(
    M: int, K: int, *, device: torch.device, with_rht: bool = False, mask_t: int = _RHT_MASK_DEFAULT
) -> ShapeBench:
    """Composite K1+K2 timing at one (M, K) shape.
    pt = per-token (no RHT), pt_rht = per-token + col-wise 16-pt RHT
    (NaN unless with_rht=True), pten = per-tensor + RHT + SR (prod baseline).
    """
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    quantizer = _make_baseline_quantizer()
    dst_a = quantizer.make_empty(a.shape, dtype=torch.bfloat16, device=device)

    # Per-token A-side buffers reused across no-RHT and +RHT paths.
    BLOCK_K = 16
    ra_a = torch.empty((M,), dtype=torch.float32, device=device)
    ca_a = torch.empty((K,), dtype=torch.float32, device=device)
    q_row_a = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    s_dec_row_a = torch.empty((M, K // BLOCK_K), dtype=torch.uint8, device=device)
    q_col_a = torch.empty((K, M // 2), dtype=torch.uint8, device=device)
    s_dec_col_a = torch.empty((K, M // BLOCK_K), dtype=torch.uint8, device=device)

    def _baseline_quant_fn():
        tex.quantize(a, quantizer, dst_a, None)

    def _pt_full_quant_fn():
        tex.nvfp4_per_token_quantize(
            a,
            q_row_a,
            s_dec_row_a,
            ra_a,
            q_col_a,
            s_dec_col_a,
            ca_a,
            True,
            True,
            with_rht=False,
            random_sign_mask_t=0,
        )

    t_pten = cuda_time_ms(_baseline_quant_fn)
    t_pt = cuda_time_ms(_pt_full_quant_fn)
    t_pten_g = cuda_graph_time_ms(_baseline_quant_fn)
    t_pt_g = cuda_graph_time_ms(_pt_full_quant_fn)

    if with_rht:

        def _pt_full_quant_rht_fn():
            tex.nvfp4_per_token_quantize(
                a,
                q_row_a,
                s_dec_row_a,
                ra_a,
                q_col_a,
                s_dec_col_a,
                ca_a,
                True,
                True,
                with_rht=True,
                random_sign_mask_t=mask_t,
            )

        t_pt_rht = cuda_time_ms(_pt_full_quant_rht_fn)
        t_pt_rht_g = cuda_graph_time_ms(_pt_full_quant_rht_fn)
    else:
        t_pt_rht = float("nan")
        t_pt_rht_g = float("nan")

    return ShapeBench(
        M=M,
        K=K,
        t_pt=t_pt,
        t_pt_rht=t_pt_rht,
        t_pten=t_pten,
        t_pt_g=t_pt_g,
        t_pt_rht_g=t_pt_rht_g,
        t_pten_g=t_pten_g,
    )


def _bench_shape_k1_only(
    M: int, K: int, *, device: torch.device, with_rht: bool = False, mask_t: int = _RHT_MASK_DEFAULT
) -> K1ShapeBench:
    """K1-only timing. pt = per-token (no RHT), pt_rht = per-token + col RHT
    (NaN unless with_rht=True), prod = hadamard_transform_amax (scalar amax;
    NOT apples-to-apples but the closest prod K1 reference).
    """
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    # Per-token K1 amax buffers (vectors).
    ra_pt = torch.empty((M,), dtype=torch.float32, device=device)
    ca_pt = torch.empty((K,), dtype=torch.float32, device=device)

    # prod K1 amax buffers (scalars).
    ra_prod = torch.empty((1,), dtype=torch.float32, device=device)
    ca_prod = torch.empty((1,), dtype=torch.float32, device=device)

    def _pt_k1_fn():
        tex.nvfp4_per_token_amax(
            a,
            ra_pt,
            ca_pt,
            True,
            True,
            with_rht=False,
            random_sign_mask_t=0,
        )

    def _prod_k1_fn():
        # row pre-RHT + col post-RHT scalar amax; both numel=1 buffers.
        tex.hadamard_transform_amax(a, ra_prod, ca_prod, mask_t)

    t_pt = cuda_time_ms(_pt_k1_fn)
    t_prod = cuda_time_ms(_prod_k1_fn)
    t_pt_g = cuda_graph_time_ms(_pt_k1_fn)
    t_prod_g = cuda_graph_time_ms(_prod_k1_fn)

    if with_rht:
        ra_pt_rht = torch.empty((M,), dtype=torch.float32, device=device)
        ca_pt_rht = torch.empty((K,), dtype=torch.float32, device=device)

        def _pt_k1_rht_fn():
            tex.nvfp4_per_token_amax(
                a,
                ra_pt_rht,
                ca_pt_rht,
                True,
                True,
                with_rht=True,
                random_sign_mask_t=mask_t,
            )

        t_pt_rht = cuda_time_ms(_pt_k1_rht_fn)
        t_pt_rht_g = cuda_graph_time_ms(_pt_k1_rht_fn)
    else:
        t_pt_rht = float("nan")
        t_pt_rht_g = float("nan")

    return K1ShapeBench(
        M=M,
        K=K,
        t_pt=t_pt,
        t_pt_rht=t_pt_rht,
        t_prod=t_prod,
        t_pt_g=t_pt_g,
        t_pt_rht_g=t_pt_rht_g,
        t_prod_g=t_prod_g,
    )


# 6x3 sweep matching bench_nvfp4_per_token_group.py: M in {1024..32768}, K in {2048,4096,8192}.
_M_VALUES: Tuple[int, ...] = (1024, 2048, 4096, 8192, 16384, 32768)
_K_VALUES: Tuple[int, ...] = (2048, 4096, 8192)
_DEFAULT_SHAPES: Tuple[Tuple[int, int], ...] = tuple((m, k) for m in _M_VALUES for k in _K_VALUES)


def _parse_shape(s: str) -> Tuple[int, int]:
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be MxK, got '{s}'")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _ratio(num: float, den: float) -> float:
    if den <= 0 or math.isnan(num) or math.isnan(den):
        return float("nan")
    return num / den


def _print_composite_table_2way(records: List[ShapeBench]) -> None:
    """2-way composite (no RHT). ratio = per-token / per-tensor (< 1.0 wins)."""
    w_pt, w_pten, w_ratio = 14, 15, 8
    block_w = w_pt + 1 + w_pten + 1 + w_ratio
    header1 = f"{'':>7} {'':>6} |{'Eager, unit (ms)':^{block_w}} |{'Graph, unit (ms)':^{block_w}}"
    header2 = (
        f"{'M':>7} {'K':>6}"
        " |"
        f"{'per-token':>{w_pt}} {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
        " |"
        f"{'per-token':>{w_pt}} {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
    )
    print(header1)
    print(header2)
    print("-" * len(header2))
    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        ratio = _ratio(rec.t_pt, rec.t_pten)
        ratio_g = _ratio(rec.t_pt_g, rec.t_pten_g)

        def _fmt(r: float) -> str:
            return "nan" if math.isnan(r) else f"{r:.2f}x"

        print(
            f"{rec.M:>7} {rec.K:>6}"
            " |"
            f"{rec.t_pt:>{w_pt}.4f} {rec.t_pten:>{w_pten}.4f} {_fmt(ratio):>{w_ratio}}"
            " |"
            f"{rec.t_pt_g:>{w_pt}.4f} {rec.t_pten_g:>{w_pten}.4f} {_fmt(ratio_g):>{w_ratio}}"
        )


def _print_composite_table(records: List[ShapeBench]) -> None:
    """3-way composite (--rht). ratio = per-token (+rht) / per-tensor."""
    w_pt, w_pt_rht, w_pten, w_ratio = 12, 12, 13, 8
    block_w = w_pt + 1 + w_pt_rht + 1 + w_pten + 1 + w_ratio
    header1 = f"{'':>7} {'':>6} |{'Eager, unit (ms)':^{block_w}} |{'Graph, unit (ms)':^{block_w}}"
    header2 = (
        f"{'M':>7} {'K':>6}"
        " |"
        f"{'per-token':>{w_pt}} {'per-token':>{w_pt_rht}}"
        f" {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
        " |"
        f"{'per-token':>{w_pt}} {'per-token':>{w_pt_rht}}"
        f" {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
    )
    header3 = (
        f"{'':>7} {'':>6}"
        " |"
        f"{'':>{w_pt}} {'(+rht)':>{w_pt_rht}}"
        f" {'':>{w_pten}} {'':>{w_ratio}}"
        " |"
        f"{'':>{w_pt}} {'(+rht)':>{w_pt_rht}}"
        f" {'':>{w_pten}} {'':>{w_ratio}}"
    )
    print(header1)
    print(header2)
    print(header3)
    print("-" * len(header2))
    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        ratio = _ratio(rec.t_pt_rht, rec.t_pten)
        ratio_g = _ratio(rec.t_pt_rht_g, rec.t_pten_g)

        def _fmt(r: float) -> str:
            return "nan" if math.isnan(r) else f"{r:.2f}x"

        print(
            f"{rec.M:>7} {rec.K:>6}"
            " |"
            f"{rec.t_pt:>{w_pt}.4f} {rec.t_pt_rht:>{w_pt_rht}.4f}"
            f" {rec.t_pten:>{w_pten}.4f} {_fmt(ratio):>{w_ratio}}"
            " |"
            f"{rec.t_pt_g:>{w_pt}.4f} {rec.t_pt_rht_g:>{w_pt_rht}.4f}"
            f" {rec.t_pten_g:>{w_pten}.4f} {_fmt(ratio_g):>{w_ratio}}"
        )


def _print_k1_2way_table(records: List[K1ShapeBench]) -> None:
    """2-way K1 (default --k1-only). pt_K1 vs prod_K1; NOT apples-to-apples
    (per-token K1 outputs M+K floats, prod outputs 2 scalars).
    """
    print("K1-only: pt vs prod (NOT apples-to-apples; output shapes differ).")
    header = (
        f"{'M':>7} {'K':>6}"
        " |"
        f"{'pt_K1':>9} {'prod_K1':>9} {'ratio':>8}"
        " |"
        f"{'pt_K1(Graph)':>14} {'prod_K1(Graph)':>16} {'ratio(Graph)':>13}"
    )
    print(header)
    print("-" * len(header))
    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        ratio = _ratio(rec.t_pt, rec.t_prod)
        ratio_g = _ratio(rec.t_pt_g, rec.t_prod_g)
        ratio_s = "nan" if math.isnan(ratio) else f"{ratio:.2f}x"
        ratio_g_s = "nan" if math.isnan(ratio_g) else f"{ratio_g:.2f}x"
        print(
            f"{rec.M:>7} {rec.K:>6}"
            " |"
            f"{rec.t_pt:>9.4f} {rec.t_prod:>9.4f} {ratio_s:>8}"
            " |"
            f"{rec.t_pt_g:>14.4f} {rec.t_prod_g:>16.4f} {ratio_g_s:>13}"
        )


def _print_k1_rht_cost_table(records: List[K1ShapeBench]) -> None:
    """Table A: pt_K1 vs pt_K1+RHT (apples-to-apples; same output shapes)."""
    print("Table A -- K1-only RHT cost (pt = per-token, +RHT = col-wise FHT).")
    header = (
        f"{'M':>7} {'K':>6}"
        " |"
        f"{'pt_K1':>9} {'pt_K1+RHT':>11} {'ratio':>8}"
        " |"
        f"{'pt_K1(Graph)':>14} {'pt_K1+RHT(Graph)':>18} {'ratio(Graph)':>13}"
    )
    print(header)
    print("-" * len(header))
    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        ratio = _ratio(rec.t_pt_rht, rec.t_pt)
        ratio_g = _ratio(rec.t_pt_rht_g, rec.t_pt_g)
        ratio_s = "nan" if math.isnan(ratio) else f"{ratio:.2f}x"
        ratio_g_s = "nan" if math.isnan(ratio_g) else f"{ratio_g:.2f}x"
        print(
            f"{rec.M:>7} {rec.K:>6}"
            " |"
            f"{rec.t_pt:>9.4f} {rec.t_pt_rht:>11.4f} {ratio_s:>8}"
            " |"
            f"{rec.t_pt_g:>14.4f} {rec.t_pt_rht_g:>18.4f} {ratio_g_s:>13}"
        )


def _print_k1_vs_prod_table(records: List[K1ShapeBench]) -> None:
    """Table B: pt_K1+RHT vs prod_K1 (NOT apples-to-apples; output shapes
    differ -- 2 scalars vs M+K floats). Fast-floor reference only.
    """
    print("Table B -- K1-only vs prod (NOT apples-to-apples; output shapes differ).")
    header = (
        f"{'M':>7} {'K':>6}"
        " |"
        f"{'pt_K1+RHT':>11} {'prod_K1':>9} {'ratio':>8}"
        " |"
        f"{'pt_K1+RHT(Graph)':>18} {'prod_K1(Graph)':>16} {'ratio(Graph)':>13}"
    )
    print(header)
    print("-" * len(header))
    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        ratio = _ratio(rec.t_pt_rht, rec.t_prod)
        ratio_g = _ratio(rec.t_pt_rht_g, rec.t_prod_g)
        ratio_s = "nan" if math.isnan(ratio) else f"{ratio:.2f}x"
        ratio_g_s = "nan" if math.isnan(ratio_g) else f"{ratio_g:.2f}x"
        print(
            f"{rec.M:>7} {rec.K:>6}"
            " |"
            f"{rec.t_pt_rht:>11.4f} {rec.t_prod:>9.4f} {ratio_s:>8}"
            " |"
            f"{rec.t_pt_rht_g:>18.4f} {rec.t_prod_g:>16.4f} {ratio_g_s:>13}"
        )


def _print_composite_legend(*, with_rht: bool, rht_mask: int) -> None:
    """Prose legend mapping table labels to their C++ entry points."""
    print()
    print("Legend:")
    if with_rht:
        print("  per-token (ms)         = tex.nvfp4_per_token_quantize(a, ..., rowwise+colwise,")
        print("                           with_rht=False)")
        print("                           = K1 fused amax + K2 fused cast (2 launches), no RHT.")
        print(
            "  per-token (+rht) (ms)  = same, but with_rht=True +"
            f" random_sign_mask_t=0x{rht_mask:04X}."
        )
        print("                           Applies a 16-point RHT along the columnwise direction in")
        print("                           BOTH K1 amax and K2 cast; rowwise stays raw. Length-16")
        print("                           matches the 1x16 inner-SF block of NVFP4, so each scale")
        print("                           window is decorrelated.")
        print("  per-tensor (ms)        = tex.quantize(a, NVFP4Quantizer(rht+sr), ...)")
        print("                           = nvte_quantize_with_hadamard_transform")
        print(
            "                           (1 fused launch: rowwise quant + col-wise RHT + col quant,"
        )
        print("                           prod baseline).")
        print("  ratio                  = per-token (+rht) / per-tensor")
        print("                           ** < 1.0 = this PR wins vs prod baseline **")
    else:
        print(
            "  per-token (ms)  = tex.nvfp4_per_token_quantize(a, ..., rowwise+colwise,"
            " with_rht=False)"
        )
        print("                    = K1 fused amax + K2 fused cast (2 launches), no RHT.")
        print("  per-tensor (ms) = tex.quantize(a, NVFP4Quantizer(rht+sr), ...)")
        print("                    = nvte_quantize_with_hadamard_transform")
        print("                    (1 fused launch: rowwise quant + col-wise RHT + col quant,")
        print("                    prod baseline).")
        print(
            "  ratio           = per-token / per-tensor   ** < 1.0 = per-token wins vs prod"
            " baseline **"
        )
    print("  (Graph) suffix    = same under CUDA Graphs replay (Python + alloc elided).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 per-token K1+K2 quant vs per-tensor production NVFP4."
    )
    parser.add_argument(
        "--shapes",
        type=_parse_shape,
        nargs="+",
        default=None,
        help=(
            "Shapes to bench, in MxK form (e.g. 4096x4096). "
            "Default: an internally-chosen production-shape sweep."
        ),
    )
    parser.add_argument(
        "--rht",
        action="store_true",
        help=(
            "Also time the per-token + RHT path (col-wise 16-pt RHT in K1 + K2). "
            "Default OFF: prints a 2-way table (per-token vs per-tensor). With "
            "--rht: prints a 3-way table with one ratio "
            "(per-token (+rht) / per-tensor)."
        ),
    )
    parser.add_argument(
        "--k1-only",
        action="store_true",
        help=(
            "K1-only mode (no K2 cast). Without --rht: 2-way table (pt_K1 "
            "vs prod_K1). With --rht: two tables back-to-back -- (A) RHT cost "
            "pt_K1 vs pt_K1+RHT (apples-to-apples) and (B) pt_K1+RHT vs prod_K1 "
            "(context only; output shapes differ)."
        ),
    )
    parser.add_argument(
        "--rht-mask",
        type=lambda s: int(s, 0),
        default=_RHT_MASK_DEFAULT,
        help=(
            "16-bit random sign mask for the RHT path (only matters with --rht). "
            f"Default 0x{_RHT_MASK_DEFAULT:04X}; accepts hex (0x...) or decimal."
        ),
    )
    args = parser.parse_args()

    if not _has_sm100():
        print("SKIP: NVFP4 per-token quant requires SM100 (Blackwell).", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    shapes = list(args.shapes) if args.shapes else list(_DEFAULT_SHAPES)
    mask = args.rht_mask & 0xFFFF

    if args.k1_only:
        records_k1: List[K1ShapeBench] = [
            _bench_shape_k1_only(M, K, device=device, with_rht=args.rht, mask_t=mask)
            for (M, K) in shapes
        ]
        if args.rht:
            _print_k1_rht_cost_table(records_k1)
            print()
            _print_k1_vs_prod_table(records_k1)
        else:
            _print_k1_2way_table(records_k1)
    else:
        records: List[ShapeBench] = [
            _bench_shape(M, K, device=device, with_rht=args.rht, mask_t=mask) for (M, K) in shapes
        ]
        if args.rht:
            _print_composite_table(records)
        else:
            _print_composite_table_2way(records)
        _print_composite_legend(with_rht=args.rht, rht_mask=mask)

    return 0


if __name__ == "__main__":
    sys.exit(main())
