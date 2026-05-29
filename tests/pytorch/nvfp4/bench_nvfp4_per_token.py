# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bench NVFP4 per-token K1+K2 quant vs per-tensor RHT+SR baseline.

bf16, M % 128 == 0, K % 128 == 0.

Modes:
  * default: 2-way quant-only (per-token vs per-tensor). Ratio = pt / pten.
  * ``--rht``: 3-way quant-only (adds per-token + col-wise 16-pt RHT).
  * ``--swizzle``: 3-way END-TO-END (quant + swizzle + cuBLAS LT NVFP4 GEMM).
    Compares per-token (separate swizzle launch) vs per-token (fused
    swizzle in K2) vs per-tensor. Ratio = per-token (+swizzle) / per-tensor.
  * ``--qs``: 2-way K1+K2 + standalone rowwise swizzle. NO GEMM.
      - default (solo, 1 tensor): K1+K2(A) + swizzle(A); apples-to-apples
        with --composite (which is also 1-tensor) -- the delta vs --composite
        is the pure marginal swizzle cost.
      - ``--pair`` (2 tensors): K1+K2(A) + K1+K2(B) + swizzle(A) + swizzle(B);
        matches prod NVFP4 GEMM's per-call quant+swizzle pipeline (1 swizzle
        per operand). Use this when you want "one GEMM call's worth of
        non-GEMM cost".
      - ``--fuse``: also bench per-token with fused-swizzle K2 (K2 directly
        writes the rowwise SF in cuBLAS LT swizzled layout; no separate
        swizzle launch). Prints a 3-way table: per-token / per-token(fuse) /
        per-tensor. The (fuse) column saves 1 swizzle launch/operand vs the
        non-fuse column.
    Ratio = per-token / per-tensor (3-way mode adds a per-token(fuse) column).
  * ``--k1-only``: K1 in isolation (orthogonal to --swizzle / --qs).
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


@dataclass
class E2EShapeBench:
    """End-to-end (quant + GEMM) timing for --swizzle mode. N is bound to M."""

    M: int
    K: int
    t_pt: float  # per-token (no fused swizzle): quant + ext swizzle + GEMM
    t_pt_swz: float  # per-token (fused swizzle): quant_with_swizzle=True + GEMM
    t_pten: float  # per-tensor: NVFP4Quantizer + cuBLAS LT GEMM
    t_pt_g: float
    t_pt_swz_g: float
    t_pten_g: float


@dataclass
class QSShapeBench:
    """K1+K2 + rowwise swizzle, no GEMM. solo=3 launches, --pair=6,
    --fuse adds per-token-fused column (K2 emits swizzled SF in 1 launch)."""

    M: int
    K: int
    t_pt: float  # per-token K1+K2 + ext swizzle (1 or 2 operands depending on pair)
    t_pten: float  # per-tensor K1+K2 + ext swizzle (matching operand count)
    t_pt_g: float
    t_pten_g: float
    t_pt_swz: float = float("nan")  # per-token K1+K2 with fused swizzle (no ext swz launch)
    t_pt_swz_g: float = float("nan")


# Default mask seed; matches prod's `te-nvfp4-build-overrides.mdc` convention.
_RHT_MASK_DEFAULT: int = 0xACE1


def _bench_shape(
    M: int, K: int, *, device: torch.device, with_rht: bool = False, mask_t: int = _RHT_MASK_DEFAULT
) -> ShapeBench:
    """Composite K1+K2 at (M, K). pt = per-token (no RHT); pt_rht = +col-wise
    16-pt RHT (NaN unless with_rht=True); pten = per-tensor + RHT + SR."""
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


def _bench_shape_e2e_swizzle(
    M: int,
    K: int,
    *,
    device: torch.device,
    with_rht: bool = False,
    mask_t: int = _RHT_MASK_DEFAULT,
) -> E2EShapeBench:
    """E2E (quant + cuBLAS LT NVFP4 GEMM) for --swizzle, square N=M.
    pt: ext swizzle; pt_swz: fused-swizzle K2 (no internal swz launch);
    pten: NVFP4Quantizer + nvfp4_per_tensor_gemm baseline."""
    from transformer_engine.pytorch.cpp_extensions.gemm import get_cublas_workspace

    N = M  # square; cuBLAS LT NVFP4 is TN-only -- A: (M, K), B: (N, K)
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    d = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    # torch.device("cuda").index is None (no explicit device index); resolve to
    # an actual GPU index via the allocated tensor so get_cublas_workspace
    # creates the workspace on the right CUDA device instead of CPU.
    workspace = get_cublas_workspace(a.device.index, ub=False, grouped_gemm=False)

    # Per-token quant produces row + col directions on every call (matches the
    # per-tensor baseline below which does both in one kernel). GEMM consumes
    # only the rowwise side; the col allocation is realistic prod overhead.
    BLOCK_K = 16

    def _alloc_pt(R, C):
        return (
            torch.empty((R, C // 2), dtype=torch.uint8, device=device),
            torch.empty((R, C // BLOCK_K), dtype=torch.uint8, device=device),
            torch.empty((R,), dtype=torch.float32, device=device),
            torch.empty((C, R // 2), dtype=torch.uint8, device=device),
            torch.empty((C, R // BLOCK_K), dtype=torch.uint8, device=device),
            torch.empty((C,), dtype=torch.float32, device=device),
        )

    a_qr, a_sr, a_ra, a_qc, a_sc, a_ca = _alloc_pt(M, K)
    b_qr, b_sr, b_ra, b_qc, b_sc, b_ca = _alloc_pt(N, K)

    def _pt_quant(t, qr, sr, ra_buf, qc, sc, ca_buf, *, fused_swizzle: bool):
        tex.nvfp4_per_token_quantize(
            t,
            qr,
            sr,
            ra_buf,
            qc,
            sc,
            ca_buf,
            True,
            True,  # rowwise + columnwise (apples-to-apples vs per-tensor)
            with_rht=with_rht,
            random_sign_mask_t=mask_t if with_rht else 0,
            with_swizzle=fused_swizzle,
        )

    def _pt_e2e_ext_swizzle():
        _pt_quant(a, a_qr, a_sr, a_ra, a_qc, a_sc, a_ca, fused_swizzle=False)
        _pt_quant(b, b_qr, b_sr, b_ra, b_qc, b_sc, b_ca, fused_swizzle=False)
        tex.nvfp4_per_token_gemm(
            a_qr,
            b_qr,
            a_sr.reshape(-1),
            b_sr.reshape(-1),
            a_ra,
            b_ra,
            d,
            workspace,
            M,
            N,
            K,
            1.0,
            0.0,
            a_sf_swizzled=False,
            b_sf_swizzled=False,
        )

    def _pt_e2e_fused_swizzle():
        _pt_quant(a, a_qr, a_sr, a_ra, a_qc, a_sc, a_ca, fused_swizzle=True)
        _pt_quant(b, b_qr, b_sr, b_ra, b_qc, b_sc, b_ca, fused_swizzle=True)
        tex.nvfp4_per_token_gemm(
            a_qr,
            b_qr,
            a_sr.reshape(-1),
            b_sr.reshape(-1),
            a_ra,
            b_ra,
            d,
            workspace,
            M,
            N,
            K,
            1.0,
            0.0,
            a_sf_swizzled=True,
            b_sf_swizzled=True,
        )

    # Per-tensor path: NVFP4Quantizer (RHT+SR) + bench-only nvfp4_per_tensor_gemm.
    quantizer = _make_baseline_quantizer()
    dst_a = quantizer.make_empty(a.shape, dtype=torch.bfloat16, device=device)
    dst_b = quantizer.make_empty(b.shape, dtype=torch.bfloat16, device=device)

    def _pten_e2e():
        tex.quantize(a, quantizer, dst_a, None)
        tex.quantize(b, quantizer, dst_b, None)
        tex.nvfp4_per_tensor_gemm(
            dst_a._rowwise_data,
            dst_b._rowwise_data,
            dst_a._rowwise_scale_inv,
            dst_b._rowwise_scale_inv,
            dst_a._amax_rowwise,
            dst_b._amax_rowwise,
            d,
            workspace,
            M,
            N,
            K,
            1.0,
            0.0,
        )

    t_pt = cuda_time_ms(_pt_e2e_ext_swizzle)
    t_pt_swz = cuda_time_ms(_pt_e2e_fused_swizzle)
    t_pten = cuda_time_ms(_pten_e2e)
    t_pt_g = cuda_graph_time_ms(_pt_e2e_ext_swizzle)
    t_pt_swz_g = cuda_graph_time_ms(_pt_e2e_fused_swizzle)
    t_pten_g = cuda_graph_time_ms(_pten_e2e)

    return E2EShapeBench(
        M=M,
        K=K,
        t_pt=t_pt,
        t_pt_swz=t_pt_swz,
        t_pten=t_pten,
        t_pt_g=t_pt_g,
        t_pt_swz_g=t_pt_swz_g,
        t_pten_g=t_pten_g,
    )


def _bench_shape_qs(
    M: int,
    K: int,
    *,
    device: torch.device,
    with_rht: bool = False,
    mask_t: int = _RHT_MASK_DEFAULT,
    pair: bool = False,
    fuse: bool = False,
) -> QSShapeBench:
    """K1+K2 + standalone rowwise swizzle, no GEMM. solo=3 launches/operand,
    --pair=6 (A+B). Swizzle binding identical across pt/pten -- only K1+K2 differs."""
    N = M  # square; matches --swizzle's apples-to-apples convention.
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    BLOCK_K = 16

    def _alloc_pt(R, C):
        return (
            torch.empty((R, C // 2), dtype=torch.uint8, device=device),
            torch.empty((R, C // BLOCK_K), dtype=torch.uint8, device=device),
            torch.empty((R,), dtype=torch.float32, device=device),
            torch.empty((C, R // 2), dtype=torch.uint8, device=device),
            torch.empty((C, R // BLOCK_K), dtype=torch.uint8, device=device),
            torch.empty((C,), dtype=torch.float32, device=device),
        )

    a_qr, a_sr, a_ra, a_qc, a_sc, a_ca = _alloc_pt(M, K)
    a_sr_swz = torch.empty(a_sr.numel(), dtype=torch.uint8, device=device)

    # B-side allocation only when --pair (avoids spurious HBM pressure in solo).
    if pair:
        b = torch.randn((N, K), dtype=torch.bfloat16, device=device)
        b_qr, b_sr, b_ra, b_qc, b_sc, b_ca = _alloc_pt(N, K)
        b_sr_swz = torch.empty(b_sr.numel(), dtype=torch.uint8, device=device)

    def _pt_quant(t, qr, sr, ra_buf, qc, sc, ca_buf):
        tex.nvfp4_per_token_quantize(
            t,
            qr,
            sr,
            ra_buf,
            qc,
            sc,
            ca_buf,
            True,
            True,
            with_rht=with_rht,
            random_sign_mask_t=mask_t if with_rht else 0,
            with_swizzle=False,  # explicit external swizzle, see below
        )

    if pair:

        def _pt_qs():
            _pt_quant(a, a_qr, a_sr, a_ra, a_qc, a_sc, a_ca)
            _pt_quant(b, b_qr, b_sr, b_ra, b_qc, b_sc, b_ca)
            tex.nvfp4_per_token_swizzle_rowwise_sf(a_qr, a_sr.reshape(-1), a_sr_swz)
            tex.nvfp4_per_token_swizzle_rowwise_sf(b_qr, b_sr.reshape(-1), b_sr_swz)

    else:

        def _pt_qs():
            _pt_quant(a, a_qr, a_sr, a_ra, a_qc, a_sc, a_ca)
            tex.nvfp4_per_token_swizzle_rowwise_sf(a_qr, a_sr.reshape(-1), a_sr_swz)

    # Per-tensor baseline path: NVFP4Quantizer (RHT+SR), reuse internal storage.
    quantizer = _make_baseline_quantizer()
    dst_a = quantizer.make_empty(a.shape, dtype=torch.bfloat16, device=device)
    pten_a_sr_swz = torch.empty(dst_a._rowwise_scale_inv.numel(), dtype=torch.uint8, device=device)
    if pair:
        dst_b = quantizer.make_empty(b.shape, dtype=torch.bfloat16, device=device)
        pten_b_sr_swz = torch.empty(
            dst_b._rowwise_scale_inv.numel(), dtype=torch.uint8, device=device
        )

    if pair:

        def _pten_qs():
            tex.quantize(a, quantizer, dst_a, None)
            tex.quantize(b, quantizer, dst_b, None)
            tex.nvfp4_per_token_swizzle_rowwise_sf(
                dst_a._rowwise_data, dst_a._rowwise_scale_inv.reshape(-1), pten_a_sr_swz
            )
            tex.nvfp4_per_token_swizzle_rowwise_sf(
                dst_b._rowwise_data, dst_b._rowwise_scale_inv.reshape(-1), pten_b_sr_swz
            )

    else:

        def _pten_qs():
            tex.quantize(a, quantizer, dst_a, None)
            tex.nvfp4_per_token_swizzle_rowwise_sf(
                dst_a._rowwise_data, dst_a._rowwise_scale_inv.reshape(-1), pten_a_sr_swz
            )

    t_pt = cuda_time_ms(_pt_qs)
    t_pten = cuda_time_ms(_pten_qs)
    t_pt_g = cuda_graph_time_ms(_pt_qs)
    t_pten_g = cuda_graph_time_ms(_pten_qs)

    t_pt_swz = float("nan")
    t_pt_swz_g = float("nan")
    if fuse:
        # Fused-swizzle K2: writes rowwise SF directly in swizzled layout
        # (same numel as compact, just byte-permuted). No external swizzle
        # launch -- K1+K2 alone is the full pipeline.
        a_qr_f, a_sr_f, a_ra_f, a_qc_f, a_sc_f, a_ca_f = _alloc_pt(M, K)
        if pair:
            b_qr_f, b_sr_f, b_ra_f, b_qc_f, b_sc_f, b_ca_f = _alloc_pt(N, K)

        def _pt_quant_fused(t, qr, sr, ra_buf, qc, sc, ca_buf):
            tex.nvfp4_per_token_quantize(
                t,
                qr,
                sr,
                ra_buf,
                qc,
                sc,
                ca_buf,
                True,
                True,
                with_rht=with_rht,
                random_sign_mask_t=mask_t if with_rht else 0,
                with_swizzle=True,  # <-- fused: K2 emits swizzled rowwise SF
            )

        if pair:

            def _pt_qs_fused():
                _pt_quant_fused(a, a_qr_f, a_sr_f, a_ra_f, a_qc_f, a_sc_f, a_ca_f)
                _pt_quant_fused(b, b_qr_f, b_sr_f, b_ra_f, b_qc_f, b_sc_f, b_ca_f)

        else:

            def _pt_qs_fused():
                _pt_quant_fused(a, a_qr_f, a_sr_f, a_ra_f, a_qc_f, a_sc_f, a_ca_f)

        t_pt_swz = cuda_time_ms(_pt_qs_fused)
        t_pt_swz_g = cuda_graph_time_ms(_pt_qs_fused)

    return QSShapeBench(
        M=M,
        K=K,
        t_pt=t_pt,
        t_pten=t_pten,
        t_pt_g=t_pt_g,
        t_pten_g=t_pten_g,
        t_pt_swz=t_pt_swz,
        t_pt_swz_g=t_pt_swz_g,
    )


def _print_qs_table(records: List[QSShapeBench], *, fuse: bool) -> None:
    """K1+K2 + rowwise swizzle (no GEMM). 2-way default, 3-way w/ --fuse.
    Ratio = per-token(fuse if --fuse else plain) / per-tensor."""

    def _fmt(r: float) -> str:
        return "nan" if math.isnan(r) else f"{r:.2f}x"

    if not fuse:
        w_pt, w_pten, w_ratio = 14, 15, 8
        block_w = w_pt + 1 + w_pten + 1 + w_ratio
        header1 = (
            f"{'':>7} {'':>6} |{'Eager, unit (ms)':^{block_w}} |{'Graph, unit (ms)':^{block_w}}"
        )
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
            print(
                f"{rec.M:>7} {rec.K:>6}"
                " |"
                f"{rec.t_pt:>{w_pt}.4f} {rec.t_pten:>{w_pten}.4f} {_fmt(ratio):>{w_ratio}}"
                " |"
                f"{rec.t_pt_g:>{w_pt}.4f} {rec.t_pten_g:>{w_pten}.4f} {_fmt(ratio_g):>{w_ratio}}"
            )
        return

    # 3-way with fuse column
    w_pt, w_swz, w_pten, w_ratio = 12, 14, 13, 8
    block_w = w_pt + 1 + w_swz + 1 + w_pten + 1 + w_ratio
    header1 = f"{'':>7} {'':>6} |{'Eager, unit (ms)':^{block_w}} |{'Graph, unit (ms)':^{block_w}}"
    header2 = (
        f"{'M':>7} {'K':>6}"
        " |"
        f"{'per-token':>{w_pt}} {'per-token':>{w_swz}}"
        f" {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
        " |"
        f"{'per-token':>{w_pt}} {'per-token':>{w_swz}}"
        f" {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
    )
    header3 = (
        f"{'':>7} {'':>6}"
        " |"
        f"{'':>{w_pt}} {'(fuse)':>{w_swz}}"
        f" {'':>{w_pten}} {'':>{w_ratio}}"
        " |"
        f"{'':>{w_pt}} {'(fuse)':>{w_swz}}"
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
        # 3-way ratio uses the fused-swizzle column vs per-tensor.
        ratio = _ratio(rec.t_pt_swz, rec.t_pten)
        ratio_g = _ratio(rec.t_pt_swz_g, rec.t_pten_g)
        print(
            f"{rec.M:>7} {rec.K:>6}"
            " |"
            f"{rec.t_pt:>{w_pt}.4f} {rec.t_pt_swz:>{w_swz}.4f}"
            f" {rec.t_pten:>{w_pten}.4f} {_fmt(ratio):>{w_ratio}}"
            " |"
            f"{rec.t_pt_g:>{w_pt}.4f} {rec.t_pt_swz_g:>{w_swz}.4f}"
            f" {rec.t_pten_g:>{w_pten}.4f} {_fmt(ratio_g):>{w_ratio}}"
        )


def _print_qs_legend(*, with_rht: bool, rht_mask: int, pair: bool, fuse: bool) -> None:
    print()
    n_tensors = 2 if pair else 1
    n_launches_ext = 3 * n_tensors  # K1+K2+swz per tensor
    n_launches_fused = 2 * n_tensors  # K1+K2 only per tensor (swizzle folded into K2)
    mode_tag = "--pair, 2 operands" if pair else "default solo, 1 operand"
    n_kernels_tag = f"ext-swz pipeline {n_launches_ext} launches" + (
        f" / fused pipeline {n_launches_fused} launches" if fuse else ""
    )
    print(f"Legend (K1+K2 + rowwise swizzle; NO GEMM; mode = {mode_tag}; {n_kernels_tag}):")
    rht_suffix = (
        f"with_rht=True + random_sign_mask_t=0x{rht_mask:04X}" if with_rht else "with_rht=False"
    )
    print(
        f"  per-token (ms)        = {n_tensors} x nvfp4_per_token_quantize({rht_suffix})"
        "  # K1+K2 each"
    )
    print(
        f"                        + {n_tensors} x nvfp4_per_token_swizzle_rowwise_sf"
        "       # 1 swz each"
    )
    print("                          K1 = nvfp4_per_token_amax (per-row/per-col vec amax)")
    print("                          K2 = nvfp4_per_token_encode (cast + e4m3 SF + optional RHT)")
    if fuse:
        print(
            f"  per-token (fuse) (ms) = {n_tensors} x nvfp4_per_token_quantize(..., "
            "with_swizzle=True)"
        )
        print("                          # K1+K2 each; K2 directly emits the swizzled rowwise")
        print("                          # SF in cuBLAS LT layout (no separate swizzle launch).")
    print(
        f"  per-tensor (ms)       = {n_tensors} x tex.quantize(NVFP4Quantizer(rht+sr))"
        "  # K1+K2 each"
    )
    print(
        f"                        + {n_tensors} x nvfp4_per_token_swizzle_rowwise_sf"
        "       # 1 swz each"
    )
    print("                          K1 = nvte_hadamard_transform_amax (post-RHT scalar amax)")
    print("                          K2 = nvte_quantize_with_hadamard_transform")
    print("                               (RHT + SR + cast fusion, rowwise + columnwise)")
    if fuse:
        print("  The (fuse) column saves 1 swizzle launch/operand vs the non-fuse column;")
        print("  the K2 byte-output is identical (verified by pytest byte-equality test).")
    if not pair:
        print("  solo mode is apples-to-apples with --composite (also 1 operand): the delta")
        print("  per-token(--qs) - per-token(--composite) ~= one nvte_swizzle launch.")
    else:
        print("  --pair mode = one prod NVFP4 GEMM call's quant+swizzle pipeline (1 swz/operand).")
    if fuse:
        print("  ratio                 = per-token(fuse) / per-tensor")
    else:
        print("  ratio                 = per-token / per-tensor")
    print("                          ** < 1.0 = this PR wins vs prod K1+K2+swizzle path **")
    print("  (Graph) suffix        = same under CUDA Graphs replay (Python + alloc elided).")


def _print_e2e_swizzle_table(records: List[E2EShapeBench]) -> None:
    """3-way end-to-end (--swizzle). ratio = per-token (+swizzle) / per-tensor."""
    w_pt, w_swz, w_pten, w_ratio = 12, 14, 13, 8
    block_w = w_pt + 1 + w_swz + 1 + w_pten + 1 + w_ratio
    header1 = f"{'':>7} {'':>6} |{'Eager, unit (ms)':^{block_w}} |{'Graph, unit (ms)':^{block_w}}"
    header2 = (
        f"{'M':>7} {'K':>6}"
        " |"
        f"{'per-token':>{w_pt}} {'per-token':>{w_swz}}"
        f" {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
        " |"
        f"{'per-token':>{w_pt}} {'per-token':>{w_swz}}"
        f" {'per-tensor':>{w_pten}} {'ratio':>{w_ratio}}"
    )
    header3 = (
        f"{'':>7} {'':>6}"
        " |"
        f"{'':>{w_pt}} {'(+swizzle)':>{w_swz}}"
        f" {'':>{w_pten}} {'':>{w_ratio}}"
        " |"
        f"{'':>{w_pt}} {'(+swizzle)':>{w_swz}}"
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
        ratio = _ratio(rec.t_pt_swz, rec.t_pten)
        ratio_g = _ratio(rec.t_pt_swz_g, rec.t_pten_g)

        def _fmt(r: float) -> str:
            return "nan" if math.isnan(r) else f"{r:.2f}x"

        print(
            f"{rec.M:>7} {rec.K:>6}"
            " |"
            f"{rec.t_pt:>{w_pt}.4f} {rec.t_pt_swz:>{w_swz}.4f}"
            f" {rec.t_pten:>{w_pten}.4f} {_fmt(ratio):>{w_ratio}}"
            " |"
            f"{rec.t_pt_g:>{w_pt}.4f} {rec.t_pt_swz_g:>{w_swz}.4f}"
            f" {rec.t_pten_g:>{w_pten}.4f} {_fmt(ratio_g):>{w_ratio}}"
        )


def _print_e2e_swizzle_legend(*, with_rht: bool, rht_mask: int) -> None:
    print()
    print("Legend (end-to-end quant + cuBLAS LT NVFP4 GEMM, square N=M):")
    rht_suffix = (
        f"with_rht=True + random_sign_mask_t=0x{rht_mask:04X}" if with_rht else "with_rht=False"
    )
    print(f"  per-token (ms)            = nvfp4_per_token_quantize({rht_suffix}) +")
    print("                              nvfp4_per_token_gemm(sf_swizzled=False)")
    print("                              -> K1 + K2 + 2 swizzle launches + cuBLAS LT GEMM")
    print("                                + per-token post-scale.")
    print(f"  per-token (+swizzle) (ms) = nvfp4_per_token_quantize({rht_suffix},")
    print("                                                       with_swizzle=True) +")
    print("                              nvfp4_per_token_gemm(sf_swizzled=True)")
    print("                              -> K1 + K2 (fused swizzle) + cuBLAS LT GEMM")
    print("                                + per-token post-scale. (2 launches saved.)")
    print("  per-tensor (ms)           = tex.quantize(a, NVFP4Quantizer(rht+sr)) +")
    print("                              nvfp4_per_tensor_gemm (cuBLAS LT NVFP4)")
    print("                              -> fused RHT+quant + 2 swizzle launches + GEMM.")
    print("  ratio                     = per-token (+swizzle) / per-tensor")
    print("                              ** < 1.0 = this PR wins vs prod baseline **")
    print("  (Graph) suffix            = same under CUDA Graphs replay (Python + alloc elided).")


def _bench_shape_k1_only(
    M: int, K: int, *, device: torch.device, with_rht: bool = False, mask_t: int = _RHT_MASK_DEFAULT
) -> K1ShapeBench:
    """K1-only. pt = per-token (no RHT); pt_rht = +col RHT (NaN unless --rht);
    prod = hadamard_transform_amax (scalar amax; not apples-to-apples)."""
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
    """2-way K1 (default --k1-only). pt_K1 vs prod_K1 (not apples-to-apples:
    per-token outputs M+K floats, prod outputs 2 scalars)."""
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
    """Table B: pt_K1+RHT vs prod_K1 (not apples-to-apples; 2 scalars
    vs M+K floats). Fast-floor reference only."""
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
        "--swizzle",
        action="store_true",
        help=(
            "End-to-end mode: quant + cuBLAS LT NVFP4 GEMM (square N=M). "
            "Prints a 3-way table: per-token (external swizzle) vs per-token "
            "(fused swizzle in K2, sf_swizzled=True) vs per-tensor. Ratio = "
            "per-token (+swizzle) / per-tensor. --rht composes (adds 16-pt "
            "col-wise RHT to the per-token paths)."
        ),
    )
    parser.add_argument(
        "--qs",
        action="store_true",
        help=(
            "K1+K2 + standalone rowwise swizzle. NO GEMM. 2-way table: "
            "per-token vs per-tensor. Default solo (1 operand, 3 launches) is "
            "apples-to-apples with --composite; add --pair for 2-operand "
            "(6 launches, matches prod NVFP4 GEMM's per-call pipeline). "
            "--rht composes."
        ),
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help=(
            "Modifier for --qs: bench the 2-operand (A + B) pipeline, matching "
            "what prod NVFP4 GEMM does per call (1 K1+K2 + 1 swizzle per "
            "operand). Default (no --pair) is solo (1 operand)."
        ),
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help=(
            "Modifier for --qs: also bench per-token with fused-swizzle K2 "
            "(K2 directly emits the rowwise SF in cuBLAS LT swizzled layout; "
            "no separate swizzle launch). Adds a 'per-token(fuse)' column to "
            "the table, and the ratio switches to per-token(fuse) / per-tensor."
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

    # --pair / --fuse are modifiers for --qs; auto-imply --qs if either is set
    # alone, so we don't silently fall through to --composite default and bake
    # a confusing "looks-like-the-modifier-worked-but-didnt" table.
    if (args.pair or args.fuse) and not args.qs:
        modifiers = []
        if args.pair:
            modifiers.append("--pair")
        if args.fuse:
            modifiers.append("--fuse")
        print(
            f"INFO: {' / '.join(modifiers)} implies --qs; running --qs "
            f"{' '.join(modifiers)} (K1+K2 + swizzle, no GEMM).",
            file=sys.stderr,
        )
        args.qs = True

    exclusive = sum(int(x) for x in (args.k1_only, args.swizzle, args.qs))
    if exclusive > 1:
        print("ERROR: --k1-only, --swizzle, and --qs are mutually exclusive.", file=sys.stderr)
        return 2

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
    elif args.swizzle:
        records_e2e: List[E2EShapeBench] = [
            _bench_shape_e2e_swizzle(M, K, device=device, with_rht=args.rht, mask_t=mask)
            for (M, K) in shapes
        ]
        _print_e2e_swizzle_table(records_e2e)
        _print_e2e_swizzle_legend(with_rht=args.rht, rht_mask=mask)
    elif args.qs:
        records_qs: List[QSShapeBench] = [
            _bench_shape_qs(
                M,
                K,
                device=device,
                with_rht=args.rht,
                mask_t=mask,
                pair=args.pair,
                fuse=args.fuse,
            )
            for (M, K) in shapes
        ]
        _print_qs_table(records_qs, fuse=args.fuse)
        _print_qs_legend(with_rht=args.rht, rht_mask=mask, pair=args.pair, fuse=args.fuse)
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
