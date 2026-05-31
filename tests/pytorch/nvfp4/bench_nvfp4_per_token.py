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
  * ``--gemm-only``: 3-way NVFP4 GEMM in isolation. Inputs are pre-quantized
    + pre-swizzled before timing. Compares per-token Route 1
    (``nvfp4_per_token_gemm`` = cuBLASLt + bf16 post-scale, 2 launches) vs
    Route 2 (``nvfp4_cutlass_per_token_gemm`` = fused-EVT CUTLASS, 1 launch)
    vs prod (``nvfp4_per_tensor_gemm`` = cuBLASLt + alpha-fold). Headline
    ratio ``lp/cf`` selects the per-token dispatcher's winning route.
  * ``--e2e-fwd``: 2-way E2E forward (quant + GEMM in the timing loop, N=K).
    Compares per-token quant (with_swizzle=True) + fused-EVT CUTLASS GEMM
    vs NVFP4Quantizer (RHT+SR) + prod ``nvfp4_per_tensor_gemm``.
  * ``--e2e-bwd``: 2-way E2E backward, real prod bwd lifecycle (N=K).
    Timing loop = 1 x dY quant + dgrad GEMM + wgrad GEMM. X, W are
    pre-quantized OUTSIDE the loop (mirrors prod's reuse of fwd-saved
    QuantizedTensorStorage; bwd never re-quantizes W or X). Compares
    per-token (dY dual K1+K2 + fused-EVT dgrad/wgrad) vs REAL-SHIP grad
    quantizer (RHT cols + SR) + general_gemm NN/NT. cf/pten is the
    actual per-step bwd cost in real training.
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
class GemmOnlyShapeBench:
    """NVFP4 GEMM in isolation. Inputs pre-quantized + pre-swizzled outside the
    timed window; only the GEMM kernel call is timed. N = K.

    Three NVFP4 GEMM paths timed side-by-side, exposing the per-token
    dispatcher's Route 1 vs Route 2 crossover and the absolute gap to prod:
      pten_gemm = cuBLAS LT NVFP4 + alpha-fold (1 launch, no post-scale).
                  CURRENT prod per-tensor baseline.
      lt_post   = cuBLAS LT NVFP4 (alpha=1 / amax pinned to 1) + standalone
                  bf16 per-row*per-col post-scale kernel. 2 launches; D round-
                  trips through HBM once. "Route 1" per-token path.
      ct_fused  = forked CUTLASS NVFP4 GEMM with per-row * per-col rescale
                  fused into the EVT epilogue. 1 launch; D never round-trips.
                  "Route 2" per-token path (current ship target).

    Headline ratio for the dispatcher decision:
      lp/cf = lt_post / ct_fused.  < 1.0 ⇒ Route 1 wins this shape (use
      cuBLASLt + post_scale); > 1.0 ⇒ Route 2 wins (use CUTLASS-fused).
    """

    M: int
    K: int
    N: int
    t_pten: float  # nvfp4_per_tensor_gemm: cuBLAS LT + alpha-fold (prod baseline)
    t_lp: float  # nvfp4_per_token_gemm: cuBLAS LT + per-row*per-col post-scale (Route 1)
    t_clf: float  # nvfp4_cutlass_per_token_gemm: per-row*per-col fused EVT (Route 2)
    t_pten_g: float
    t_lp_g: float
    t_clf_g: float


@dataclass
class E2EForwardShapeBench:
    """E2E forward (quant + GEMM): per-token CUTLASS fused-EVT vs prod per-tensor
    cuBLASLt. N = K. Ratio cf/pten < 1.0 = shippable (per-token wins E2E).
    """

    M: int
    K: int
    N: int
    t_pten: float  # NVFP4Quantizer (RHT+SR) + nvfp4_per_tensor_gemm
    t_cf: float  # nvfp4_per_token_quantize(with_swizzle=True) + nvfp4_cutlass_per_token_gemm
    t_pten_g: float
    t_cf_g: float


@dataclass
class E2EBackwardShapeBench:
    """E2E backward, real prod nn.Linear.bwd lifecycle (N = K).
    Timing loop = 1 x dY quant + dgrad GEMM + wgrad GEMM. X, W are
    pre-quantized OUTSIDE the loop (mirrors prod's reuse of fwd-saved
    QuantizedTensorStorage; bwd flips usage flags only, never re-quantizes).
    cf/pten < 1.0 = per-token bwd faster than real-ship prod per-tensor.
    Prod path uses real NVFP4BlockScaling defaults (grad RHT+SR for dY);
    per-token currently has no RHT/SR.
    """

    M: int
    K: int
    N: int
    t_pten: float  # dY quant (grad_q, RHT+SR) + general_gemm NN dgrad + NT wgrad
    t_cf: float  # dY quant (per-token dual) + fused-EVT dgrad + wgrad
    t_pten_g: float
    t_cf_g: float


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


def _bench_shape_gemm_only(
    M: int,
    K: int,
    *,
    device: torch.device,
    with_rht: bool = False,
    mask_t: int = _RHT_MASK_DEFAULT,
) -> GemmOnlyShapeBench:
    """NVFP4 GEMM in isolation (N = K). Quant + swizzle run once before timing;
    only the GEMM kernel call is timed.

    Three paths timed (per-token Route 1 vs Route 2 vs prod per-tensor):
      - pten_gemm: cuBLAS LT NVFP4 per-tensor (current prod NVFP4 GEMM,
                   alpha-folded, no post-scale).
      - lt_post  : cuBLAS LT NVFP4 (amax pinned to 1.0) + standalone bf16
                   per-row * per-col post-scale kernel. Per-token Route 1
                   (2 launches, D round-trips HBM once).
      - ct_fused : forked CUTLASS NVFP4 GEMM with per-row * per-col rescale
                   fused into the EVT epilogue. Per-token Route 2 (1 launch,
                   D never round-trips).
    """
    from transformer_engine.pytorch.cpp_extensions.gemm import get_cublas_workspace

    N = K
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    d = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    workspace = get_cublas_workspace(a.device.index, ub=False, grouped_gemm=False)

    BLOCK_K = 16

    # Per-token rowwise quant for the fused CUTLASS path. Pre-swizzled SF so
    # the timed window only covers the GEMM kernel (no swizzle launch).
    a_qr = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    a_sr = torch.empty((M, K // BLOCK_K), dtype=torch.uint8, device=device)
    a_ra = torch.empty((M,), dtype=torch.float32, device=device)
    b_qr = torch.empty((N, K // 2), dtype=torch.uint8, device=device)
    b_sr = torch.empty((N, K // BLOCK_K), dtype=torch.uint8, device=device)
    b_ra = torch.empty((N,), dtype=torch.float32, device=device)
    empty_u8 = torch.empty(0, dtype=torch.uint8, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    tex.nvfp4_per_token_quantize(
        a,
        a_qr,
        a_sr,
        a_ra,
        empty_u8,
        empty_u8,
        empty_f32,
        True,
        False,
        with_rht=with_rht,
        random_sign_mask_t=mask_t if with_rht else 0,
        with_swizzle=True,
    )
    tex.nvfp4_per_token_quantize(
        b,
        b_qr,
        b_sr,
        b_ra,
        empty_u8,
        empty_u8,
        empty_f32,
        True,
        False,
        with_rht=with_rht,
        random_sign_mask_t=mask_t if with_rht else 0,
        with_swizzle=True,
    )
    a_sr_flat = a_sr.reshape(-1)
    b_sr_flat = b_sr.reshape(-1)

    # Per-tensor: NVFP4Quantizer (RHT+SR) -> pre-swizzle SF once so prod GEMM
    # call doesn't pay 2 swizzle launches inside the timed window either.
    quantizer = _make_baseline_quantizer()
    dst_a = quantizer.make_empty(a.shape, dtype=torch.bfloat16, device=device)
    dst_b = quantizer.make_empty(b.shape, dtype=torch.bfloat16, device=device)
    tex.quantize(a, quantizer, dst_a, None)
    tex.quantize(b, quantizer, dst_b, None)

    pten_a_sr_flat = dst_a._rowwise_scale_inv.reshape(-1)
    pten_b_sr_flat = dst_b._rowwise_scale_inv.reshape(-1)
    pten_a_sr_swz = torch.empty(pten_a_sr_flat.numel(), dtype=torch.uint8, device=device)
    pten_b_sr_swz = torch.empty(pten_b_sr_flat.numel(), dtype=torch.uint8, device=device)
    tex.nvfp4_per_token_swizzle_rowwise_sf(dst_a._rowwise_data, pten_a_sr_flat, pten_a_sr_swz)
    tex.nvfp4_per_token_swizzle_rowwise_sf(dst_b._rowwise_data, pten_b_sr_flat, pten_b_sr_swz)

    def _pten_gemm():
        tex.nvfp4_per_tensor_gemm(
            dst_a._rowwise_data,
            dst_b._rowwise_data,
            pten_a_sr_swz,
            pten_b_sr_swz,
            dst_a._amax_rowwise,
            dst_b._amax_rowwise,
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

    # Route 1 per-token: cuBLAS LT NVFP4 (operand amaxes pinned to 1.0) + a
    # standalone bf16 per-row * per-col post-scale kernel. 2 launches; D
    # round-trips through HBM once (post-scale is HBM-bound elementwise).
    # Reuses the per-token-quantized A/B above (same a_ra/b_ra); pre-swizzled
    # SF skips the in-binding swizzle so the timed window is exactly
    # GEMM + post_scale.
    d_lp = torch.empty_like(d)

    def _lt_post():
        tex.nvfp4_per_token_gemm(
            a_qr,
            b_qr,
            a_sr_flat,
            b_sr_flat,
            a_ra,
            b_ra,
            d_lp,
            workspace,
            M,
            N,
            K,
            1.0,
            0.0,
            a_sf_swizzled=True,
            b_sf_swizzled=True,
            skip_post_scale=False,
        )

    # Route 2 per-token: forked CUTLASS NVFP4 GEMM with per-row * per-col
    # rescale fused INTO the epilogue (EVT). Single launch; the M*N output
    # never round-trips through HBM.
    d_clf = torch.empty_like(d)

    def _cutlass_fused():
        tex.nvfp4_cutlass_per_token_gemm(
            a_qr,
            b_qr,
            a_sr_flat,
            b_sr_flat,
            a_ra,
            b_ra,
            d_clf,
            M,
            N,
            K,
            a_sf_swizzled=True,
            b_sf_swizzled=True,
        )

    t_pten = cuda_time_ms(_pten_gemm)
    t_lp = cuda_time_ms(_lt_post)
    t_clf = cuda_time_ms(_cutlass_fused)
    t_pten_g = cuda_graph_time_ms(_pten_gemm)
    t_lp_g = cuda_graph_time_ms(_lt_post)
    t_clf_g = cuda_graph_time_ms(_cutlass_fused)

    return GemmOnlyShapeBench(
        M=M,
        K=K,
        N=N,
        t_pten=t_pten,
        t_lp=t_lp,
        t_clf=t_clf,
        t_pten_g=t_pten_g,
        t_lp_g=t_lp_g,
        t_clf_g=t_clf_g,
    )


def _bench_shape_e2e_fwd(
    M: int,
    K: int,
    *,
    device: torch.device,
) -> E2EForwardShapeBench:
    """E2E forward (quant + GEMM in the timing loop, N = K). Two paths:
      - pten: NVFP4Quantizer (input RHT 1D / weight no-RHT 2D) + general_gemm
              (the real prod nn.Linear fwd dispatch).
      - cf:   nvfp4_per_token_quantize(with_swizzle=True) + fused-EVT GEMM.

    Kernel pipeline per fwd call (Y = X @ W^T):

      pten (prod NVFP4BlockScaling defaults):
        X bf16 ─→ tex.quantize(input_q  RHT-cols 1D no-SR)    ~4 launches
                   (K1 row amax | K2 row cast | K1 col amax+RHT | K2 col cast)
        W bf16 ─→ tex.quantize(weight_q 2D no-RHT no-SR)      ~2 launches
                   (2D fused cast | amax)
        X_q, W_q ─→ general_gemm(W_q, X_q, layout="NN")
                    (swizzle X SF | swizzle W SF | compute_α 1×1 | cuBLASLt GEMM)
                                                              4 launches
                                                          ──────────────
                                                          ~10 launches
        Y = bf16(α · X @ W^T)
            α = per-tensor scalar (one fp32) -- FREE in cuBLASLt epilogue.

      cf (per-token):
        X bf16 ─→ tex.nvfp4_per_token_quantize(with_swizzle=True)
                   (K1 amax row+col | K2 encode + swizzle fused)
                                                              2 launches
        W bf16 ─→ tex.nvfp4_per_token_quantize(with_swizzle=True)
                                                              2 launches
        X_q, W_q ─→ nvfp4_cutlass_per_token_gemm
                    (single CUTLASS NVFP4 GEMM with EVT epilogue)
                                                              1 launch
                                                          ──────────────
                                                          5 launches
        Y[i,j] = bf16(a_ra[i] · b_ra[j] · (X @ W^T)[i,j])
                 α = per-row × per-col vector, fused into CUTLASS EVT
                 (zero extra HBM traffic; D never round-trips).
    """
    N = K
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=device)

    BLOCK_K = 16

    # Per-token: produce BOTH rowwise (consumed by the fwd CUTLASS GEMM) AND
    # columnwise (kept for downstream dgrad/wgrad). Matches prod fwd quant
    # workload -- NVFP4Quantizer(rowwise=True, columnwise=True) also produces
    # both in K2. with_swizzle=True only affects rowwise SF layout (it
    # collapses to kWithSwizzle=False for colwise SF inside the encode kernel),
    # so col SF stays in plain layout (will be swizzled separately in dgrad).
    a_qr = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    a_sr = torch.empty((M, K // BLOCK_K), dtype=torch.uint8, device=device)
    a_ra = torch.empty((M,), dtype=torch.float32, device=device)
    a_qc = torch.empty((K, M // 2), dtype=torch.uint8, device=device)
    a_sc = torch.empty((K, M // BLOCK_K), dtype=torch.uint8, device=device)
    a_ca = torch.empty((K,), dtype=torch.float32, device=device)
    b_qr = torch.empty((N, K // 2), dtype=torch.uint8, device=device)
    b_sr = torch.empty((N, K // BLOCK_K), dtype=torch.uint8, device=device)
    b_ra = torch.empty((N,), dtype=torch.float32, device=device)
    b_qc = torch.empty((K, N // 2), dtype=torch.uint8, device=device)
    b_sc = torch.empty((K, N // BLOCK_K), dtype=torch.uint8, device=device)
    b_ca = torch.empty((K,), dtype=torch.float32, device=device)
    d_cf = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    a_sr_flat = a_sr.reshape(-1)
    b_sr_flat = b_sr.reshape(-1)

    def _cf_e2e():
        tex.nvfp4_per_token_quantize(
            a,
            a_qr,
            a_sr,
            a_ra,
            a_qc,
            a_sc,
            a_ca,
            True,
            True,
            with_rht=False,
            random_sign_mask_t=0,
            with_swizzle=True,
        )
        tex.nvfp4_per_token_quantize(
            b,
            b_qr,
            b_sr,
            b_ra,
            b_qc,
            b_sc,
            b_ca,
            True,
            True,
            with_rht=False,
            random_sign_mask_t=0,
            with_swizzle=True,
        )
        tex.nvfp4_cutlass_per_token_gemm(
            a_qr,
            b_qr,
            a_sr_flat,
            b_sr_flat,
            a_ra,
            b_ra,
            d_cf,
            M,
            N,
            K,
            a_sf_swizzled=True,
            b_sf_swizzled=True,
        )

    # Prod per-tensor baseline (NVFP4BlockScaling defaults, prod fwd) via
    # general_gemm -- the real production GEMM dispatch used by nn.Linear fwd.
    #   A (input)  -- input quantizer  -- RHT, 1D, no SR
    #   B (weight) -- weight quantizer -- no RHT, 2D, no SR
    # Both write rowwise + columnwise (col side kept for dgrad/wgrad parity).
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    input_q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=False,
        with_random_sign_mask=True,
    )
    weight_q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
        stochastic_rounding=False,
        with_random_sign_mask=True,
    )
    dst_a = input_q.make_empty(a.shape, dtype=torch.bfloat16, device=device)
    dst_b = weight_q.make_empty(b.shape, dtype=torch.bfloat16, device=device)
    d_pten = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    def _pten_e2e():
        tex.quantize(a, input_q, dst_a, None)
        tex.quantize(b, weight_q, dst_b, None)
        # general_gemm(weight, input, ...) matches linear.py prod fwd.
        general_gemm(dst_b, dst_a, out=d_pten, out_dtype=torch.bfloat16)

    t_pten = cuda_time_ms(_pten_e2e)
    t_cf = cuda_time_ms(_cf_e2e)
    t_pten_g = cuda_graph_time_ms(_pten_e2e)
    t_cf_g = cuda_graph_time_ms(_cf_e2e)

    return E2EForwardShapeBench(
        M=M,
        K=K,
        N=N,
        t_pten=t_pten,
        t_cf=t_cf,
        t_pten_g=t_pten_g,
        t_cf_g=t_cf_g,
    )


def _bench_shape_e2e_bwd(
    M: int,
    K: int,
    *,
    device: torch.device,
) -> E2EBackwardShapeBench:
    """E2E backward, mirroring REAL prod nn.Linear.bwd lifecycle (N = K).

    In prod, fwd produces quantized X (saved_inputmat) and W (wt_save) with
    rowwise + columnwise data, both kept in ctx. bwd reads them back as
    QuantizedTensorStorage and only flips usage flags via update_usage --
    NO re-quantization of W or X. Only dY is freshly quantized in bwd.

    Therefore the timing loop only contains:
        1 x dY quant  +  1 x dgrad GEMM  +  1 x wgrad GEMM

    Both paths follow the same lifecycle:
      - pten: dY via grad quantizer (RHT cols, SR per NVFP4BlockScaling
              default for fp4_quant_bwd_grad). X and W are pre-quantized
              ONCE outside the loop using input/weight quantizers (real-ship
              recipe defaults). Then general_gemm NN dgrad + NT wgrad,
              byte-equivalent to linear.py prod bwd dispatch.
      - cf:   dY via nvfp4_per_token_quantize(rowwise+columnwise, no RHT,
              no swizzle). X and W also pre-quantized ONCE outside the loop
              with the same kernel. Then fused-EVT dgrad (M,K,N) + wgrad
              (N,K,M).

    cf/pten now reflects the actual per-step bwd cost in real training.

    Kernel pipeline per bwd step (X_q, W_q reused from fwd; NOT re-quantized):

      pten (real-ship NVFP4BlockScaling bwd defaults):
        dY bf16 ─→ tex.quantize(grad_q  RHT-cols + SR 1D)     ~4 launches
                   (K1 row amax | K2 row cast | K1 col amax+RHT |
                    K2 col cast + SR)
        dgrad: dX = dY @ W ─→ general_gemm(W_q, dY_q, layout="NN")
                              (swizzle W col SF | swizzle dY row SF |
                               compute_α 1×1 | cuBLASLt NVFP4 GEMM)
                                                              4 launches
        wgrad: dW = dY^T @ X ─→ general_gemm(X_q, dY_q, layout="NT")
                                (swizzle X col SF | swizzle dY col SF |
                                 compute_α 1×1 | cuBLASLt NVFP4 GEMM)
                                                              4 launches
                                                          ──────────────
                                                          ~12 launches
        α = per-tensor scalar -- FREE in cuBLASLt epilogue.

      cf (per-token):
        dY bf16 ─→ tex.nvfp4_per_token_quantize(no RHT, no swizzle)
                   (K1 amax row+col | K2 encode rowwise + colwise SF)
                                                              2 launches
        dgrad: dX = dY @ W ─→ nvfp4_cutlass_per_token_gemm
                              (dY_row · W_col, fused EVT)     1 launch
        wgrad: dW = dY^T @ X ─→ nvfp4_cutlass_per_token_gemm
                                (dY_col · X_col, fused EVT)   1 launch
                                                          ──────────────
                                                          4 launches
        α = per-row × per-col vector, fused into CUTLASS EVT epilogue.
        Per-step launch budget is ~3× lighter than prod, but each
        CUTLASS GEMM is ~26% slower than the cuBLASLt counterpart at
        large M*K -- net win is shape-dependent.
    """
    N = K
    dy = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    w = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)

    BLOCK_K = 16

    # --- per-token side: dual-direction quant buffers for dY, W, X.
    def _alloc_pt(R: int, C: int):
        # Allocate full row+col buffers. For input shape (R, C):
        #   rowwise: data (R, C/2), sf (R, C/16), amax (R,)
        #   columnwise: data (C, R/2), sf (C, R/16), amax (C,)
        return (
            torch.empty((R, C // 2), dtype=torch.uint8, device=device),
            torch.empty((R, C // BLOCK_K), dtype=torch.uint8, device=device),
            torch.empty((R,), dtype=torch.float32, device=device),
            torch.empty((C, R // 2), dtype=torch.uint8, device=device),
            torch.empty((C, R // BLOCK_K), dtype=torch.uint8, device=device),
            torch.empty((C,), dtype=torch.float32, device=device),
        )

    dy_qr, dy_sr, dy_ra, dy_qc, dy_sc, dy_ca = _alloc_pt(M, N)
    w_qr, w_sr, w_ra, w_qc, w_sc, w_ca = _alloc_pt(N, K)
    x_qr, x_sr, x_ra, x_qc, x_sc, x_ca = _alloc_pt(M, K)
    dy_sr_flat = dy_sr.reshape(-1)
    dy_sc_flat = dy_sc.reshape(-1)
    w_sc_flat = w_sc.reshape(-1)
    x_sc_flat = x_sc.reshape(-1)
    d_dgrad = torch.empty((M, K), dtype=torch.bfloat16, device=device)
    d_wgrad = torch.empty((N, K), dtype=torch.bfloat16, device=device)

    # Pre-quantize X and W ONCE (mirrors fwd-saved tensors that bwd reads
    # back from ctx). These calls are NOT in the timing loop.
    tex.nvfp4_per_token_quantize(
        w,
        w_qr,
        w_sr,
        w_ra,
        w_qc,
        w_sc,
        w_ca,
        True,
        True,
        with_rht=False,
        random_sign_mask_t=0,
        with_swizzle=False,
    )
    tex.nvfp4_per_token_quantize(
        x,
        x_qr,
        x_sr,
        x_ra,
        x_qc,
        x_sc,
        x_ca,
        True,
        True,
        with_rht=False,
        random_sign_mask_t=0,
        with_swizzle=False,
    )

    def _cf_e2e():
        # Real bwd: only dY is freshly quantized. W, X reused from fwd.
        tex.nvfp4_per_token_quantize(
            dy,
            dy_qr,
            dy_sr,
            dy_ra,
            dy_qc,
            dy_sc,
            dy_ca,
            True,
            True,
            with_rht=False,
            random_sign_mask_t=0,
            with_swizzle=False,
        )
        # dgrad: dX = dY @ W (M, N) @ (N, K) = (M, K).
        tex.nvfp4_cutlass_per_token_gemm(
            dy_qr,
            w_qc,
            dy_sr_flat,
            w_sc_flat,
            dy_ra,
            w_ca,
            d_dgrad,
            M,
            K,
            N,
            a_sf_swizzled=False,
            b_sf_swizzled=False,
        )
        # wgrad: dW = dY^T @ X (N, M) @ (M, K) = (N, K).
        tex.nvfp4_cutlass_per_token_gemm(
            dy_qc,
            x_qc,
            dy_sc_flat,
            x_sc_flat,
            dy_ca,
            x_ca,
            d_wgrad,
            N,
            K,
            M,
            a_sf_swizzled=False,
            b_sf_swizzled=False,
        )

    # --- prod per-tensor side: real-ship NVFP4BlockScaling defaults.
    #   X (input)  -- input quantizer  -- RHT(cols), no SR, 1D
    #   W (weight) -- weight quantizer -- no RHT, no SR, 2D
    #   dY (grad)  -- grad quantizer   -- RHT(cols), SR, 1D
    # Real prod path = the actual ship config; cf/pten reflects ship delta.
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    input_q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=False,
        with_random_sign_mask=True,
    )
    weight_q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
        stochastic_rounding=False,
        with_random_sign_mask=True,
    )
    grad_q = NVFP4Quantizer(
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
    dst_dy = grad_q.make_empty(dy.shape, dtype=torch.bfloat16, device=device)
    dst_w = weight_q.make_empty(w.shape, dtype=torch.bfloat16, device=device)
    dst_x = input_q.make_empty(x.shape, dtype=torch.bfloat16, device=device)
    d_pten_dgrad = torch.empty((M, K), dtype=torch.bfloat16, device=device)
    d_pten_wgrad = torch.empty((N, K), dtype=torch.bfloat16, device=device)

    # Pre-quantize X and W ONCE (mirrors fwd-saved NVFP4Tensors that
    # prod bwd reuses without re-quantization). NOT in timing loop.
    tex.quantize(w, weight_q, dst_w, None)
    tex.quantize(x, input_q, dst_x, None)

    def _pten_e2e():
        # Real bwd: only dY is freshly quantized. W, X reused from fwd.
        tex.quantize(dy, grad_q, dst_dy, None)
        # dgrad: general_gemm(W, dY, layout='NN') -> (M, K)
        general_gemm(
            dst_w,
            dst_dy,
            layout="NN",
            grad=True,
            out=d_pten_dgrad,
            out_dtype=torch.bfloat16,
        )
        # wgrad: general_gemm(X, dY, layout='NT') -> (N, K)
        general_gemm(
            dst_x,
            dst_dy,
            layout="NT",
            grad=True,
            out=d_pten_wgrad,
            out_dtype=torch.bfloat16,
        )

    t_pten = cuda_time_ms(_pten_e2e)
    t_cf = cuda_time_ms(_cf_e2e)
    t_pten_g = cuda_graph_time_ms(_pten_e2e)
    t_cf_g = cuda_graph_time_ms(_cf_e2e)

    return E2EBackwardShapeBench(
        M=M,
        K=K,
        N=N,
        t_pten=t_pten,
        t_cf=t_cf,
        t_pten_g=t_pten_g,
        t_cf_g=t_cf_g,
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


def _print_gemm_only_table(records: List[GemmOnlyShapeBench]) -> None:
    """GEMM-only (--gemm-only) timings (3-way per-token Route 1 vs Route 2 vs prod):
      pten_gemm = cuBLAS LT NVFP4 per-tensor + alpha-fold (PROD baseline, 1 launch).
      lt_post   = cuBLAS LT NVFP4 (amax=1) + bf16 per-row*per-col post-scale
                  (per-token Route 1, 2 launches, D round-trips HBM).
      ct_fused  = forked CUTLASS NVFP4 + per-row*per-col rescale fused EVT
                  (per-token Route 2, 1 launch, no D round-trip).

    Headline ratio for the dispatcher decision:
      lp/cf = lt_post / ct_fused
              < 1.0 ⇒ Route 1 wins this shape  (cuBLASLt + post_scale)
              > 1.0 ⇒ Route 2 wins this shape  (CUTLASS-fused EVT)
    """
    w_pten, w_lp, w_clf, w_ratio = 11, 11, 11, 8
    block_w = w_pten + 1 + w_lp + 1 + w_clf + 1 + w_ratio
    header1 = f"{'':>7} {'':>6} {'':>6} |{'Eager':^{block_w}} |{'Graph':^{block_w}}"
    body = f"{'pten_gemm':>{w_pten}} {'lt_post':>{w_lp}} {'ct_fused':>{w_clf}} {'lp/cf':>{w_ratio}}"
    header2 = f"{'M':>7} {'K':>6} {'N':>6} |{body}|{body}"
    print(header1)
    print(header2)
    print("-" * len(header2))

    def _fmt(r: float) -> str:
        return "nan" if math.isnan(r) else f"{r:.2f}x"

    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        r_lpcf = _ratio(rec.t_lp, rec.t_clf)
        r_lpcf_g = _ratio(rec.t_lp_g, rec.t_clf_g)
        print(
            f"{rec.M:>7} {rec.K:>6} {rec.N:>6}"
            " |"
            f"{rec.t_pten:>{w_pten}.4f} {rec.t_lp:>{w_lp}.4f}"
            f" {rec.t_clf:>{w_clf}.4f} {_fmt(r_lpcf):>{w_ratio}}"
            "|"
            f"{rec.t_pten_g:>{w_pten}.4f} {rec.t_lp_g:>{w_lp}.4f}"
            f" {rec.t_clf_g:>{w_clf}.4f} {_fmt(r_lpcf_g):>{w_ratio}}"
        )


def _print_gemm_only_legend() -> None:
    print()
    print("Legend (GEMM-only; inputs pre-quantized + pre-swizzled, N = K):")
    print("  pten_gemm (ms) = nvfp4_per_tensor_gemm(sf_swizzled=True)")
    print("                   -> cuBLAS LT NVFP4 + alpha-fold (current PROD per-tensor GEMM).")
    print("                      1 launch, no post-scale; per-tensor scalar amax folded into")
    print("                      cuBLAS-internal alpha (free).")
    print("  lt_post   (ms) = nvfp4_per_token_gemm(sf_swizzled=True, skip_post_scale=False)")
    print("                   -> cuBLAS LT NVFP4 (operand amaxes pinned to 1.0)")
    print("                      + standalone bf16 per-row*per-col post-scale kernel.")
    print("                      Per-token Route 1: 2 launches; D round-trips HBM once.")
    print("                      Inherits cuBLASLt's tuned NVFP4 GEMM kernel.")
    print("  ct_fused  (ms) = nvfp4_cutlass_per_token_gemm(sf_swizzled=True)")
    print("                   -> forked CUTLASS NVFP4 GEMM with per-row * per-col rescale")
    print("                      FUSED into the EVT epilogue (1 launch, no post-scale).")
    print("                      D = bf16(alpha_a[i] * alpha_b[j] * (A @ B^T)[i, j]).")
    print("                      Per-token Route 2: 1 launch; D never round-trips.")
    print("  lp/cf          = lt_post / ct_fused  (DISPATCHER DECISION)")
    print("                   ** < 1.0 = Route 1 (cuBLASLt + post_scale) wins this shape **")
    print("                   ** > 1.0 = Route 2 (CUTLASS fused EVT)    wins this shape **")
    print("                   crossover threshold: lp/cf = 1.0; pick the faster route at runtime.")
    print("  (Graph) suffix = same under CUDA Graphs replay (Python + alloc elided).")
    print()
    print("Reading the absolute gap to prod (for context):")
    print("  cf/pten = ct_fused / pten_gemm     (Route 2 vs prod)")
    print("  lp/pten = lt_post  / pten_gemm     (Route 1 vs prod)")
    print("  Per-token fundamentally pays a per-row*per-col scaling tax that prod does not")
    print("  (prod uses per-tensor scalar -> free in cuBLASLt epilogue). Both routes lose")
    print("  to prod at large M*K; the dispatcher just picks whichever route loses LESS.")


def _print_e2e_fwd_table(records: List[E2EForwardShapeBench]) -> None:
    """E2E forward (--e2e-fwd): quant + GEMM inside the timing loop.
    pten_e2e = NVFP4Quantizer (RHT+SR) + nvfp4_per_tensor_gemm (prod baseline).
    ct_fused = nvfp4_per_token_quantize(with_swizzle=True) + fused-EVT GEMM.
    """
    w_pten, w_cf, w_ratio = 11, 11, 8
    block_w = w_pten + 1 + w_cf + 1 + w_ratio
    header1 = f"{'':>7} {'':>6} {'':>6} |{'Eager':^{block_w}} |{'Graph':^{block_w}}"
    body = f"{'pten_e2e':>{w_pten}} {'ct_fused':>{w_cf}} {'cf/pten':>{w_ratio}}"
    header2 = f"{'M':>7} {'K':>6} {'N':>6} |{body}|{body}"
    print(header1)
    print(header2)
    print("-" * len(header2))

    def _fmt(r: float) -> str:
        return "nan" if math.isnan(r) else f"{r:.2f}x"

    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        r_cf = _ratio(rec.t_cf, rec.t_pten)
        r_cf_g = _ratio(rec.t_cf_g, rec.t_pten_g)
        print(
            f"{rec.M:>7} {rec.K:>6} {rec.N:>6}"
            " |"
            f"{rec.t_pten:>{w_pten}.4f} {rec.t_cf:>{w_cf}.4f}"
            f" {_fmt(r_cf):>{w_ratio}}"
            "|"
            f"{rec.t_pten_g:>{w_pten}.4f} {rec.t_cf_g:>{w_cf}.4f}"
            f" {_fmt(r_cf_g):>{w_ratio}}"
        )


def _print_e2e_fwd_legend() -> None:
    print()
    print("Legend (E2E forward; quant + GEMM inside the timing loop; N = K):")
    print("  pten_e2e (ms) = tex.quantize(NVFP4Quantizer; RHT+SR) +")
    print("                  nvfp4_per_tensor_gemm  (PROD per-tensor pipeline).")
    print("  ct_fused (ms) = nvfp4_per_token_quantize(with_rht=False, with_swizzle=True) +")
    print("                  nvfp4_cutlass_per_token_gemm(sf_swizzled=True).")
    print("                  K2 emits SF in swizzled layout -> 1 quant launch per operand.")
    print("  cf/pten       = ct_fused / pten_e2e")
    print("                  ** < 1.0 = per-token fused E2E beats prod per-tensor E2E **")
    print("  (Graph) suffix = same under CUDA Graphs replay (Python + alloc elided).")


def _print_e2e_bwd_table(records: List[E2EBackwardShapeBench]) -> None:
    """E2E backward (--e2e-bwd): real prod bwd lifecycle. Timing loop =
    1 x dY quant + dgrad GEMM + wgrad GEMM (X, W pre-quantized outside loop).
      pten_bwd  = REAL-SHIP grad quantizer (RHT cols + SR) + general_gemm
                  dgrad (NN) + wgrad (NT). Byte-equivalent to prod nn.Linear bwd.
      ct_fused  = nvfp4_per_token_quantize (dual, no RHT/SR) + fused-EVT dgrad + wgrad.
    """
    w_pten, w_cf, w_ratio = 11, 11, 8
    block_w = w_pten + 1 + w_cf + 1 + w_ratio
    header1 = f"{'':>7} {'':>6} {'':>6} |{'Eager':^{block_w}} |{'Graph':^{block_w}}"
    body = f"{'pten_bwd':>{w_pten}} {'ct_fused':>{w_cf}} {'cf/pten':>{w_ratio}}"
    header2 = f"{'M':>7} {'K':>6} {'N':>6} |{body}|{body}"
    print(header1)
    print(header2)
    print("-" * len(header2))

    def _fmt(r: float) -> str:
        return "nan" if math.isnan(r) else f"{r:.2f}x"

    prev_M = None
    for rec in records:
        if prev_M is not None and rec.M != prev_M:
            print()
        prev_M = rec.M
        r_cf = _ratio(rec.t_cf, rec.t_pten)
        r_cf_g = _ratio(rec.t_cf_g, rec.t_pten_g)
        print(
            f"{rec.M:>7} {rec.K:>6} {rec.N:>6}"
            " |"
            f"{rec.t_pten:>{w_pten}.4f} {rec.t_cf:>{w_cf}.4f}"
            f" {_fmt(r_cf):>{w_ratio}}"
            "|"
            f"{rec.t_pten_g:>{w_pten}.4f} {rec.t_cf_g:>{w_cf}.4f}"
            f" {_fmt(r_cf_g):>{w_ratio}}"
        )


def _print_e2e_bwd_legend() -> None:
    print()
    print("Legend (E2E backward; N = K; real prod nn.Linear.bwd lifecycle):")
    print("  Timing loop = 1 x dY quant + dgrad GEMM + wgrad GEMM.")
    print("  X and W are PRE-QUANTIZED ONCE outside the loop. This mirrors")
    print("  real prod: fwd's quantized X (saved_inputmat) and W (wt_save)")
    print("  are read back from ctx in bwd; bwd only flips usage flags via")
    print("  update_usage(), never re-quantizes them. Only dY is freshly")
    print("  quantized per backward step.")
    print()
    print("  pten_bwd (ms) = dY quant via grad quantizer (RHT cols, SR, 1D, real-ship")
    print("                  fp4_quant_bwd_grad default) + general_gemm dgrad")
    print("                  (layout='NN', dX = dY @ W) + general_gemm wgrad")
    print("                  (layout='NT', dW = dY^T @ X). X/W pre-quantized via")
    print("                  input/weight quantizers (RHT cols / 2D respectively).")
    print("  ct_fused (ms) = dY quant via nvfp4_per_token_quantize (dual, no RHT/SR,")
    print("                  no swizzle) + fused-EVT dgrad (M,K,N) + wgrad (N,K,M).")
    print("                  X/W pre-quantized via the same kernel.")
    print("                  Per-token currently has NO RHT/SR (kernel TODO).")
    print("  cf/pten       = ct_fused / pten_bwd")
    print("                  ** < 1.0 = per-token bwd beats real-ship prod bwd **")
    print("                  Reflects actual per-step bwd cost in real training.")
    print("  (Graph) suffix = same under CUDA Graphs replay (Python + alloc elided).")


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
        "--gemm-only",
        action="store_true",
        help=(
            "GEMM-only mode (square N=M): inputs are pre-quantized + pre-swizzled "
            "outside the timed window, so only the cuBLAS LT NVFP4 GEMM call is "
            "timed. 2-way table: pt_gemm (per-token GEMM + per-row post-scale) "
            "vs pten_gemm (per-tensor GEMM, alpha-folded). ratio = pt / pten "
            "exposes the per-call cost of the per-token post-scale kernel. "
            "--rht composes (RHT applied only to the per-token quant setup)."
        ),
    )
    parser.add_argument(
        "--e2e-fwd",
        action="store_true",
        help=(
            "E2E forward mode (N = K): per-token NVFP4 quant (with_swizzle=True "
            "fused in K2) + fused-EVT CUTLASS GEMM vs prod per-tensor cuBLASLt "
            "(NVFP4Quantizer RHT+SR + nvfp4_per_tensor_gemm). 2-way table; "
            "cf/pten < 1.0 = per-token E2E beats prod."
        ),
    )
    parser.add_argument(
        "--e2e-bwd",
        action="store_true",
        help=(
            "E2E backward mode (N = K), real prod nn.Linear.bwd lifecycle. "
            "Timing loop = 1 x dY quant + dgrad GEMM + wgrad GEMM (X, W "
            "pre-quantized outside loop, mirroring prod's reuse of fwd-saved "
            "QuantizedTensorStorage). Per-token (dY dual K1+K2 + fused-EVT "
            "dgrad/wgrad) vs REAL-SHIP grad quantizer (RHT cols + SR) + "
            "general_gemm NN/NT. 2-way table; cf/pten < 1.0 = per-token bwd "
            "beats real-ship prod."
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

    exclusive = sum(
        int(x)
        for x in (
            args.k1_only,
            args.swizzle,
            args.qs,
            args.gemm_only,
            args.e2e_fwd,
            args.e2e_bwd,
        )
    )
    if exclusive > 1:
        print(
            "ERROR: --k1-only, --swizzle, --qs, --gemm-only, --e2e-fwd, --e2e-bwd "
            "are mutually exclusive.",
            file=sys.stderr,
        )
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
    elif args.gemm_only:
        records_go: List[GemmOnlyShapeBench] = [
            _bench_shape_gemm_only(M, K, device=device, with_rht=args.rht, mask_t=mask)
            for (M, K) in shapes
        ]
        _print_gemm_only_table(records_go)
        _print_gemm_only_legend()
    elif args.e2e_fwd:
        records_e2ef: List[E2EForwardShapeBench] = [
            _bench_shape_e2e_fwd(M, K, device=device) for (M, K) in shapes
        ]
        _print_e2e_fwd_table(records_e2ef)
        _print_e2e_fwd_legend()
    elif args.e2e_bwd:
        records_e2eb: List[E2EBackwardShapeBench] = [
            _bench_shape_e2e_bwd(M, K, device=device) for (M, K) in shapes
        ]
        _print_e2e_bwd_table(records_e2eb)
        _print_e2e_bwd_legend()
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
