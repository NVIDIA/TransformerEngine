# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bench NVFP4 per-token K1+K2 quant vs per-tensor RHT+SR baseline.

Quant-only (no GEMM). Both sides time the K1 (amax) + K2 (cast) composite on
activation A, rowwise+columnwise. Requires bf16 input, M % 128 == 0, K % 128 == 0.
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
    t_pt: float  # per-token full K1+K2 (eager pybind, ms)
    t_pten: float  # per-tensor full K1+K2 (eager pybind, ms)
    t_pt_g: float  # per-token under CUDA Graphs replay (ms)
    t_pten_g: float  # per-tensor under CUDA Graphs replay (ms)


def _bench_shape(M: int, K: int, *, device: torch.device) -> ShapeBench:
    """Time per-tensor vs per-token K1+K2 quant at one (M, K) shape."""
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    # Per-tensor quantizer + A output tensor.
    quantizer = _make_baseline_quantizer()
    dst_a = quantizer.make_empty(a.shape, dtype=torch.bfloat16, device=device)

    # Per-token A-side buffers: BLOCK_K=16 (1x16 e4m3 inner SF).
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
        )

    t_pten = cuda_time_ms(_baseline_quant_fn)
    t_pt = cuda_time_ms(_pt_full_quant_fn)
    t_pten_g = cuda_graph_time_ms(_baseline_quant_fn)
    t_pt_g = cuda_graph_time_ms(_pt_full_quant_fn)

    return ShapeBench(M=M, K=K, t_pt=t_pt, t_pten=t_pten, t_pt_g=t_pt_g, t_pten_g=t_pten_g)


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
    args = parser.parse_args()

    if not _has_sm100():
        print("SKIP: NVFP4 per-token quant requires SM100 (Blackwell).", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    shapes = list(args.shapes) if args.shapes else list(_DEFAULT_SHAPES)

    records: List[ShapeBench] = [_bench_shape(M, K, device=device) for (M, K) in shapes]

    header = (
        f"{'M':>7} {'K':>6}"
        " |"
        f"{'per-token':>10} {'per-tensor':>11} {'ratio':>8}"
        " |"
        f"{'per-token(Graph)':>17} {'per-tensor(Graph)':>18} {'ratio(Graph)':>13}"
    )
    print(header)
    print("-" * len(header))
    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        ratio = _ratio(rec.t_pt, rec.t_pten)
        ratio_g = _ratio(rec.t_pt_g, rec.t_pten_g)
        ratio_s = "nan" if math.isnan(ratio) else f"{ratio:.2f}x"
        ratio_g_s = "nan" if math.isnan(ratio_g) else f"{ratio_g:.2f}x"
        print(
            f"{rec.M:>7} {rec.K:>6}"
            " |"
            f"{rec.t_pt:>10.4f} {rec.t_pten:>11.4f} {ratio_s:>8}"
            " |"
            f"{rec.t_pt_g:>17.4f} {rec.t_pten_g:>18.4f} {ratio_g_s:>13}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
