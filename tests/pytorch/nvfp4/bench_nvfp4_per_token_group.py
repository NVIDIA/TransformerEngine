"""Bench NVFP4 per-token grouped K1+K2 quant vs per-tensor RHT+SR baseline.

Modes:
  * default: 2-way (per-token vs per-tensor). Ratio = pt / pten.
  * ``--rht``: 3-way (adds per-token + col-wise 16-pt RHT). Ratio =
    per-token (+rht) / per-tensor.

Default sweep: N=8 equal splits, sum_M in {1024..32768} x K in {2048,4096,8192}.
Requires bf16, K % 128 == 0, every split % 128 == 0, num_splits <= 64.

CLI:
  --shapes SUMMxK ...   custom shapes (default: 18-row sweep)
  --num-splits N        equal splits per shape (default 8)
  --rht                 enable 3-way RHT comparison
  --rht-mask 0x...      16-bit RHT sign pattern (default 0xACE1)
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from typing import Callable, List, Tuple

import torch

# Import transformer_engine first so libtransformer_engine.so is dlopen'd
# before transformer_engine_torch tries to resolve its typeinfo symbols.
import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex  # type: ignore  # noqa: F401

from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_per_token_group import (
    nvfp4_per_token_group_quantize,
)


def _make_baseline_quantizer_list(num_splits: int) -> List[NVFP4Quantizer]:
    """Per-tensor RHT+SR baseline: one quantizer instance shared across N splits."""
    q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=True,
        with_random_sign_mask=True,
    )
    return [q] * num_splits


def cuda_graph_time_ms(fn: Callable[[], object], *, warmup: int = 5, iters: int = 50) -> float:
    """Median g.replay() time of fn under CUDA Graphs, in ms (nan on capture failure)."""
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
    return statistics.median(starts[i].elapsed_time(ends[i]) for i in range(iters))


# Default RHT mask seed; matches te-nvfp4-build-overrides.mdc convention.
_RHT_MASK_DEFAULT: int = 0xACE1


def _time_grouped(
    x_concat,
    split_sections,
    rowwise,
    columnwise,
    *,
    with_rht: bool = False,
    mask: int = _RHT_MASK_DEFAULT,
    n_iters: int = 20,
    n_warmup: int = 5,
) -> float:
    """Per-token grouped via the BULK Python wrapper. Allocation in-loop."""
    for _ in range(n_warmup):
        _ = nvfp4_per_token_group_quantize(
            x_concat,
            split_sections,
            rowwise=rowwise,
            columnwise=columnwise,
            with_rht=with_rht,
            random_sign_mask_t=mask,
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        _ = nvfp4_per_token_group_quantize(
            x_concat,
            split_sections,
            rowwise=rowwise,
            columnwise=columnwise,
            with_rht=with_rht,
            random_sign_mask_t=mask,
        )
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / n_iters  # ms


def _time_split_quantize(x_concat, split_sections, quantizer_list, n_iters=20, n_warmup=5):
    """Per-tensor grouped baseline: tex.split_quantize, allocation in-binding."""
    for _ in range(n_warmup):
        _ = tex.split_quantize(x_concat, split_sections, quantizer_list)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        _ = tex.split_quantize(x_concat, split_sections, quantizer_list)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / n_iters  # ms


def _time_split_quantize_graph(x_concat, split_sections, quantizer_list, n_iters=20, n_warmup=5):
    """Per-tensor grouped under CUDA Graphs replay."""

    def fn() -> None:
        _ = tex.split_quantize(x_concat, split_sections, quantizer_list)

    return cuda_graph_time_ms(fn, warmup=n_warmup, iters=n_iters)


def _time_grouped_graph(
    x_concat,
    split_sections,
    rowwise,
    columnwise,
    *,
    with_rht: bool = False,
    mask: int = _RHT_MASK_DEFAULT,
    n_iters: int = 20,
    n_warmup: int = 5,
) -> float:
    """Per-token grouped under CUDA Graphs replay."""

    def fn() -> None:
        _ = nvfp4_per_token_group_quantize(
            x_concat,
            split_sections,
            rowwise=rowwise,
            columnwise=columnwise,
            with_rht=with_rht,
            random_sign_mask_t=mask,
        )

    return cuda_graph_time_ms(fn, warmup=n_warmup, iters=n_iters)


# Default sweep: N = 8 equal splits (MoE-typical), sum_M in {1024..32768},
# K in {2048..8192}. Override either via the CLI flags below.
_DEFAULT_NUM_SPLITS: int = 8
_DEFAULT_SUM_M_VALUES: Tuple[int, ...] = (1024, 2048, 4096, 8192, 16384, 32768)
_DEFAULT_K_VALUES: Tuple[int, ...] = (2048, 4096, 8192)


def _parse_shape(s: str) -> Tuple[int, int]:
    """Parse a `sum_MxK` CLI argument."""
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be sum_MxK, got '{s}'")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _build_bench_cases(
    shapes: List[Tuple[int, int]], num_splits: int
) -> List[Tuple[List[int], int]]:
    """Turn (sum_M, K) pairs into (split_sections, K) cases; each split
    must be a multiple of 128.
    """
    cases: List[Tuple[List[int], int]] = []
    for sum_M, K in shapes:
        if sum_M % num_splits != 0:
            raise argparse.ArgumentTypeError(
                f"sum_M={sum_M} not divisible by num_splits={num_splits}"
            )
        M_i = sum_M // num_splits
        if M_i % 128 != 0:
            raise argparse.ArgumentTypeError(
                f"sum_M={sum_M} / num_splits={num_splits} = M_i={M_i} must be a "
                "multiple of 128 (NVFP4 per-token kernel constraint)"
            )
        if K % 128 != 0:
            raise argparse.ArgumentTypeError(f"K={K} must be a multiple of 128")
        cases.append(([M_i] * num_splits, K))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bench NVFP4 per-token grouped K1+K2 quant. Three-way: "
            "per-token (no RHT) / per-token+RHT / per-tensor (RHT+SR)."
        )
    )
    parser.add_argument(
        "--shapes",
        type=_parse_shape,
        nargs="+",
        default=None,
        help=(
            "Shapes to bench, in sum_MxK form (e.g. 8192x4096). "
            "Default: a 6x3 = 18-row internally-chosen sweep."
        ),
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=_DEFAULT_NUM_SPLITS,
        help=(
            f"Number of equal splits per shape (default {_DEFAULT_NUM_SPLITS}; "
            "<= 64). M_i = sum_M / num_splits must be a multiple of 128."
        ),
    )
    parser.add_argument(
        "--rht",
        action="store_true",
        help=(
            "Enable 3-way table with per-token + col-wise 16-pt RHT path. "
            "Default OFF prints 2-way (per-token vs per-tensor)."
        ),
    )
    parser.add_argument(
        "--rht-mask",
        type=lambda s: int(s, 0),
        default=_RHT_MASK_DEFAULT,
        help=(
            f"16-bit RHT sign mask (default 0x{_RHT_MASK_DEFAULT:04X}; accepts "
            "hex/dec). Only affects per-token+RHT; per-tensor uses its own mask."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping bench.")
        return 1
    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        print(f"NVFP4 per-token requires SM100+ (got SM{cap[0]}.{cap[1]}); skipping.")
        return 1
    if args.num_splits <= 0 or args.num_splits > 64:
        print(f"--num-splits must be in [1, 64], got {args.num_splits}")
        return 2

    if args.shapes is not None:
        shapes_in = [tuple(s) for s in args.shapes]
    else:
        shapes_in = [(sm, k) for sm in _DEFAULT_SUM_M_VALUES for k in _DEFAULT_K_VALUES]
    bench_cases = _build_bench_cases(shapes_in, args.num_splits)
    rht_mask: int = args.rht_mask & 0xFFFF
    with_rht: bool = args.rht

    device = torch.device("cuda")
    print(f"# Device: {torch.cuda.get_device_name(0)}  (cap {cap[0]}.{cap[1]})")
    print(f"# Split structure: N={args.num_splits} equal splits, M_i = sum_M / {args.num_splits}")
    if with_rht:
        print(
            f"# RHT mask: 0x{rht_mask:04X}  (per-token+RHT col-wise; per-tensor uses its own"
            " internal mask)"
        )
    else:
        print(
            "# RHT: disabled (pass --rht to enable 3-way per-token / per-token (+rht) / per-tensor"
            " table)"
        )
    print()

    # Per-tensor baseline quantizer is fixed to row+col, so both enabled.
    rowwise = True
    columnwise = True

    def _fmt(r: float) -> str:
        return "nan" if math.isnan(r) else f"{r:.2f}x"

    def _ratio(num: float, den: float) -> float:
        if den <= 0 or math.isnan(num) or math.isnan(den):
            return float("nan")
        return num / den

    # Multi-line header: section label + column names (+ `(+rht)` sub-label
    # row in 3-way mode), then separator + data rows.
    if with_rht:
        w_pt, w_pt_rht, w_pten, w_ratio = 12, 12, 13, 8
        block_w = w_pt + 1 + w_pt_rht + 1 + w_pten + 1 + w_ratio
        header1 = (
            f"{'':>6} {'':>5} |{'Eager, unit (ms)':^{block_w}} |{'Graph, unit (ms)':^{block_w}}"
        )
        header2 = (
            f"{'sum_M':>6} {'K':>5}"
            " |"
            f"{'per-token':>{w_pt}} {'per-token':>{w_pt_rht}}"
            f" {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
            " |"
            f"{'per-token':>{w_pt}} {'per-token':>{w_pt_rht}}"
            f" {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
        )
        header3 = (
            f"{'':>6} {'':>5}"
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
    else:
        w_pt, w_pten, w_ratio = 14, 15, 8
        block_w = w_pt + 1 + w_pten + 1 + w_ratio
        header1 = (
            f"{'':>6} {'':>5} |{'Eager, unit (ms)':^{block_w}} |{'Graph, unit (ms)':^{block_w}}"
        )
        header2 = (
            f"{'sum_M':>6} {'K':>5}"
            " |"
            f"{'per-token':>{w_pt}} {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
            " |"
            f"{'per-token':>{w_pt}} {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
        )
        print(header1)
        print(header2)
    print("-" * len(header2))

    prev_sum_M = None
    for split_sections, K in bench_cases:
        sum_M = sum(split_sections)
        num_splits = len(split_sections)

        # Blank line between sum_M groups for readability.
        if prev_sum_M is not None and sum_M != prev_sum_M:
            print()
        prev_sum_M = sum_M

        x_concat = (torch.randn((sum_M, K), dtype=torch.bfloat16, device=device) * 3.0).contiguous()
        quantizer_list = _make_baseline_quantizer_list(num_splits)

        t_pt = _time_grouped(x_concat, split_sections, rowwise, columnwise, with_rht=False)
        t_pten = _time_split_quantize(x_concat, split_sections, quantizer_list)
        t_pt_g = _time_grouped_graph(
            x_concat,
            split_sections,
            rowwise,
            columnwise,
            with_rht=False,
        )
        t_pten_g = _time_split_quantize_graph(
            x_concat,
            split_sections,
            quantizer_list,
        )

        if with_rht:
            t_pt_rht = _time_grouped(
                x_concat, split_sections, rowwise, columnwise, with_rht=True, mask=rht_mask
            )
            t_pt_rht_g = _time_grouped_graph(
                x_concat,
                split_sections,
                rowwise,
                columnwise,
                with_rht=True,
                mask=rht_mask,
            )

            ratio_eager = _ratio(t_pt_rht, t_pten)
            ratio_graph = _ratio(t_pt_rht_g, t_pten_g)

            print(
                f"{sum_M:>6d} {K:>5d}"
                " |"
                f"{t_pt:>{w_pt}.4f} {t_pt_rht:>{w_pt_rht}.4f}"
                f" {t_pten:>{w_pten}.4f} {_fmt(ratio_eager):>{w_ratio}}"
                " |"
                f"{t_pt_g:>{w_pt}.4f} {t_pt_rht_g:>{w_pt_rht}.4f}"
                f" {t_pten_g:>{w_pten}.4f} {_fmt(ratio_graph):>{w_ratio}}"
            )
        else:
            ratio_eager = _ratio(t_pt, t_pten)
            ratio_graph = _ratio(t_pt_g, t_pten_g)
            print(
                f"{sum_M:>6d} {K:>5d}"
                " |"
                f"{t_pt:>{w_pt}.4f} {t_pten:>{w_pten}.4f} {_fmt(ratio_eager):>{w_ratio}}"
                " |"
                f"{t_pt_g:>{w_pt}.4f} {t_pten_g:>{w_pten}.4f} {_fmt(ratio_graph):>{w_ratio}}"
            )

        del x_concat, quantizer_list
        torch.cuda.empty_cache()

    print()
    print("Legend:")
    if with_rht:
        print("  per-token (ms)         = nvfp4_per_token_group_quantize(x, splits,")
        print("                           rowwise+colwise, with_rht=False)")
        print("                           = K1 fused amax + K2 fused cast (2 launches), no RHT.")
        print(
            "  per-token (+rht) (ms)  = same, but with_rht=True +"
            f" random_sign_mask_t=0x{rht_mask:04X}."
        )
        print("                           Applies a 16-point RHT along the columnwise direction in")
        print("                           BOTH K1 amax and K2 cast; rowwise stays raw. Length-16")
        print("                           matches the 1x16 inner-SF block of NVFP4, so each scale")
        print("                           window is decorrelated.")
        print(
            "  per-tensor (ms)        = tex.split_quantize(x, splits, [NVFP4Quantizer(rht+sr)]*N)"
        )
        print("                           = nvte_group_hadamard_transform_amax")
        print("                           + nvte_group_hadamard_transform_cast_fusion")
        print("                           (2 launches, prod baseline).")
        print("  ratio                  = per-token (+rht) / per-tensor")
        print("                           ** < 1.0 = this PR wins vs prod baseline **")
    else:
        print(
            "  per-token (ms)  = nvfp4_per_token_group_quantize(x, splits, rowwise+colwise,"
            " with_rht=False)"
        )
        print("                    = K1 fused amax + K2 fused cast (2 launches), no RHT.")
        print("  per-tensor (ms) = tex.split_quantize(x, splits, [NVFP4Quantizer(rht+sr)]*N)")
        print("                    = nvte_group_hadamard_transform_amax")
        print(
            "                    + nvte_group_hadamard_transform_cast_fusion (2 launches, prod"
            " baseline)."
        )
        print(
            "  ratio           = per-token / per-tensor   ** < 1.0 = per-token wins vs prod"
            " baseline **"
        )
    print("  (Graph) suffix    = same under CUDA Graphs replay (Python + alloc elided).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
