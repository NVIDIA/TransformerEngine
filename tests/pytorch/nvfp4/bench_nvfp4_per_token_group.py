"""Bench: NVFP4 per-token grouped (K1+K2 fused) vs per-tensor+RHT baseline.

18-row sweep at fixed N=8 splits: sum_M in {1024..32768} x K in {2048,4096,8192}.
Both eager and CUDA Graphs columns reported on every row (ratio < 1.0 wins).
Requires bf16, K % 128 == 0, every split % 128 == 0, num_splits <= 64.
"""

from __future__ import annotations

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
    """Median g.replay() time of fn captured into a CUDA Graph, in ms.

    Returns nan if capture fails (e.g. some C-API does an incompatible sync).
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
    return statistics.median(starts[i].elapsed_time(ends[i]) for i in range(iters))


def _time_grouped(x_concat, split_sections, rowwise, columnwise, n_iters=20, n_warmup=5):
    """Per-token grouped via the BULK Python wrapper. Allocation in-loop."""
    for _ in range(n_warmup):
        _ = nvfp4_per_token_group_quantize(
            x_concat, split_sections, rowwise=rowwise, columnwise=columnwise
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        _ = nvfp4_per_token_group_quantize(
            x_concat, split_sections, rowwise=rowwise, columnwise=columnwise
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


def _time_grouped_graph(x_concat, split_sections, rowwise, columnwise, n_iters=20, n_warmup=5):
    """Per-token grouped under CUDA Graphs replay."""

    def fn() -> None:
        _ = nvfp4_per_token_group_quantize(
            x_concat, split_sections, rowwise=rowwise, columnwise=columnwise
        )

    return cuda_graph_time_ms(fn, warmup=n_warmup, iters=n_iters)


# N = 8 equal splits (MoE-typical), sum_M in {1024..32768}, K in {2048..8192}.
_NUM_SPLITS: int = 8

_SUM_M_VALUES: List[int] = [1024, 2048, 4096, 8192, 16384, 32768]
_K_VALUES: List[int] = [2048, 4096, 8192]

_BENCH_CASES: List[Tuple[List[int], int]] = []
for _sum_M in _SUM_M_VALUES:
    _M_i = _sum_M // _NUM_SPLITS
    for _K in _K_VALUES:
        _BENCH_CASES.append(([_M_i] * _NUM_SPLITS, _K))


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping bench.")
        return
    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        print(f"NVFP4 per-token requires SM100+ (got SM{cap[0]}.{cap[1]}); skipping.")
        return

    device = torch.device("cuda")
    print(f"# Device: {torch.cuda.get_device_name(0)}  (cap {cap[0]}.{cap[1]})")
    print(f"# Split structure: N={_NUM_SPLITS} equal splits, M_i = sum_M / {_NUM_SPLITS}")
    print()

    # Per-tensor baseline quantizer is fixed to row+col, so both enabled.
    rowwise = True
    columnwise = True

    header = (
        f"{'sum_M':>6} {'K':>5}"
        " |"
        f"{'per-token':>10} {'per-tensor':>10} {'ratio':>8}"
        " |"
        f"{'per-token(Graph)':>17} {'per-tensor(Graph)':>17} {'ratio(Graph)':>13}"
    )
    print(header)
    print("-" * len(header))

    prev_sum_M = None
    for split_sections, K in _BENCH_CASES:
        sum_M = sum(split_sections)
        num_splits = len(split_sections)

        # Blank line between sum_M groups for readability.
        if prev_sum_M is not None and sum_M != prev_sum_M:
            print()
        prev_sum_M = sum_M

        x_concat = (torch.randn((sum_M, K), dtype=torch.bfloat16, device=device) * 3.0).contiguous()
        quantizer_list = _make_baseline_quantizer_list(num_splits)

        t_pt = _time_grouped(x_concat, split_sections, rowwise, columnwise)
        t_pten = _time_split_quantize(x_concat, split_sections, quantizer_list)
        ratio = t_pt / t_pten if t_pten > 0 else float("nan")

        t_pt_g = _time_grouped_graph(x_concat, split_sections, rowwise, columnwise)
        t_pten_g = _time_split_quantize_graph(x_concat, split_sections, quantizer_list)
        if math.isnan(t_pt_g) or math.isnan(t_pten_g) or t_pten_g <= 0:
            ratio_g = float("nan")
            graph_cells = f"{t_pt_g:>17.4f} {t_pten_g:>17.4f} {'nan':>13}"
        else:
            ratio_g = t_pt_g / t_pten_g
            graph_cells = f"{t_pt_g:>17.4f} {t_pten_g:>17.4f} {ratio_g:>12.2f}x"

        print(f"{sum_M:>6d} {K:>5d} |{t_pt:>10.4f} {t_pten:>10.4f} {ratio:>7.2f}x |{graph_cells}")

        del x_concat, quantizer_list
        torch.cuda.empty_cache()

    print()
    print("Legend:")
    print("  per-token        = nvfp4_per_token_group_quantize(x, splits, rowwise+colwise)")
    print("                     = K1 fused amax + K2 fused cast (2 launches), this PR")
    print("  per-tensor       = tex.split_quantize(x, splits, [NVFP4Quantizer(rht+sr)]*N)")
    print("                     = nvte_group_hadamard_transform_amax")
    print("                     + nvte_group_hadamard_transform_cast_fusion (2 launches)")
    print("  ratio            = per-token / per-tensor   ** < 1.0 = this PR wins **")
    print("  (Graph) suffix   = same under CUDA Graphs replay (Python + alloc elided,")
    print("                     pure kernel-level wall time, ALL rows)")


if __name__ == "__main__":
    main()
