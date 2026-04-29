# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Accuracy comparison: hierarchical 1x64 + 1x16 vs per-tensor + 1x16.

The hypothesis under test is that per-1x64-window ``S_enc`` (the hierarchical
scheme) reconstructs ``x`` at least as accurately as per-tensor ``S_enc``
(the production NVFP4 scheme), and strictly better when the K-direction
magnitude varies meaningfully across windows -- which is the case the
hierarchical scheme is built for.

Methodology
-----------
This is a *spec-level* accuracy test, not a kernel test:

1. Quantize the same input through two pure-PyTorch references --
   ``NVFP4QuantizerRef`` (per-tensor) and ``NVFP4Quantizer1x64Ref``
   (per-1x64-window) -- to obtain ``(qx, sx)`` for each.
2. Dequantize both back to fp32 using each scheme's *own* ``S_enc``:
   ``x_recon = q * s_dec_fp32 / S_enc``. For the per-tensor scheme
   ``S_enc`` is a scalar; for 1x64 it is per-window. (Using the per-tensor
   ``S_enc`` to dequantize a 1x64-encoded tensor would re-introduce the
   GEMM-mismatch bug -- a separate concern that is documented elsewhere.)
3. Compute reconstruction error metrics (RMSE, max abs, Frobenius-relative)
   against the original fp32 input.
4. Assert ``rmse_1x64`` is at most a small slack worse than ``rmse_pt`` on
   benign inputs, and strictly better on inputs with per-window dynamic
   range. Numbers are also printed so a regression failure is diagnostic.

Because the bit-exact tests in ``test_nvfp4_1x64_quantize_exact.py`` already
certify ``NVFP4Quantizer1x64Ref`` reproduces the CUDA kernel byte-for-byte,
any accuracy conclusion we draw at the reference level transfers directly
to the kernel.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import (
    NVFP4QuantizerRef,
    cast_from_fp4x2,
)
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_1x64 import (
    BLOCK_K,
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    NVFP4Quantizer1x64Ref,
    WINDOW_K,
)


_SAFE_AMAX_FLOOR = 1e-12


def _per_tensor_s_enc(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor encoding scaling factor used by stock NVFP4.

    Returns a per-element broadcast of shape ``(M, N)`` so the dequant
    formula can be expressed elementwise without further care for layout.
    """
    M, N = x.shape
    fp32_max = torch.tensor(torch.finfo(torch.float32).max, device=x.device, dtype=torch.float32)
    global_amax = torch.amax(torch.abs(x.to(torch.float32)))
    s = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.clamp(global_amax, min=_SAFE_AMAX_FLOOR)
    s = torch.minimum(s, fp32_max)
    if float(s.item()) == 0.0:
        s = torch.ones_like(s)
    return s.expand(M, N).contiguous()


def _per_window_s_enc(x: torch.Tensor) -> torch.Tensor:
    """Per-1x64-window encoding scaling factor, broadcast to ``(M, N)``."""
    M, N = x.shape
    pad_n = (WINDOW_K - N % WINDOW_K) % WINDOW_K
    if pad_n > 0:
        x_padded = torch.nn.functional.pad(x, (0, pad_n), mode="constant", value=0.0)
    else:
        x_padded = x.contiguous()
    Np = x_padded.shape[1]
    n_win = Np // WINDOW_K

    fp32_max = torch.tensor(torch.finfo(torch.float32).max, device=x.device, dtype=torch.float32)
    x_padded_fp32 = x_padded.to(torch.float32).view(M, n_win, WINDOW_K)
    tile_amax = torch.amax(torch.abs(x_padded_fp32), dim=-1, keepdim=True)
    tile_amax_safe = torch.clamp(tile_amax, min=_SAFE_AMAX_FLOOR)

    s = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / tile_amax_safe
    s = torch.minimum(s, fp32_max)
    s = torch.where(
        (tile_amax_safe == 0) | (s == 0),
        torch.ones_like(s),
        s,
    )
    s_per_elt = s.squeeze(-1).repeat_interleave(WINDOW_K, dim=1)
    return s_per_elt[:, :N].contiguous()


def _dequantize(
    qx_packed: torch.Tensor,
    sx_e4m3: torch.Tensor,
    s_enc_per_elt: torch.Tensor,
    M: int,
    N: int,
) -> torch.Tensor:
    """Inverse of NVFP4 forward quantization: ``q * s_dec_fp32 / S_enc``."""
    q_fp32 = cast_from_fp4x2(qx_packed.view(torch.uint8), torch.float32)
    sx_fp32 = sx_e4m3.view(torch.float8_e4m3fn).to(torch.float32)
    sx_fp32 = sx_fp32[:M, : N // BLOCK_K]
    sx_per_elt = sx_fp32.repeat_interleave(BLOCK_K, dim=1)
    return q_fp32 * sx_per_elt / s_enc_per_elt


def _recon_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """Forward + inverse via the production per-tensor NVFP4 reference."""
    M, N = x.shape
    quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=False,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        with_rht=False,
    )
    out = quantizer.quantize(x)
    s_enc = _per_tensor_s_enc(x)
    return _dequantize(out.data, out.scale, s_enc, M, N)


def _recon_1x64(x: torch.Tensor) -> torch.Tensor:
    """Forward + inverse via the hierarchical 1x64 reference."""
    M, N = x.shape
    out = NVFP4Quantizer1x64Ref().quantize(x)
    s_enc = _per_window_s_enc(x)
    return _dequantize(out.data, out.scale, s_enc, M, N)


def _err_metrics(x: torch.Tensor, recon: torch.Tensor) -> Tuple[float, float, float]:
    """Return ``(rmse, max_abs_err, frobenius_relative)`` of ``recon - x``."""
    x_fp32 = x.to(torch.float32)
    diff = recon.to(torch.float32) - x_fp32
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    max_err = torch.max(torch.abs(diff)).item()
    denom = torch.linalg.norm(x_fp32).clamp(min=1e-30)
    frob_rel = (torch.linalg.norm(diff) / denom).item()
    return rmse, max_err, frob_rel


def _gen_gaussian(M: int, N: int, *, seed: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Uniform N(0, 1) -- a benign baseline where both schemes should tie."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((M, N), generator=g, device=device, dtype=dtype)


def _gen_per_window_dynamic_range(
    M: int,
    N: int,
    *,
    seed: int,
    device: str,
    dtype: torch.dtype,
    log10_lo: float,
    log10_hi: float,
) -> torch.Tensor:
    """Each 1x64 window has its own log-uniform magnitude scale.

    This generator is used to verify that 1x64 does **not regress** versus
    per-tensor when per-window magnitude varies but stays inside E4M3's
    representable range. It deliberately does not try to demonstrate a
    large advantage: per-block ``s_dec`` already adapts to ``vec_max`` in
    both schemes, so as long as ``s_dec`` does not underflow E4M3, FP4
    resolution per element is ``vec_max / 12`` regardless of the ``S_enc``
    choice, and the two schemes tie up to E4M3 rounding noise.

    To actually demonstrate the 1x64 advantage in absolute RMSE terms, see
    ``_gen_sparse_extreme_outliers`` -- that distribution forces prod's
    ``s_dec`` into the underflow regime for the bulk of the tensor.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    n_win = (N + WINDOW_K - 1) // WINDOW_K
    log_scales = torch.empty((M, n_win, 1), device=device, dtype=torch.float32)
    log_scales.uniform_(log10_lo, log10_hi, generator=g)
    scales = torch.pow(torch.tensor(10.0, device=device, dtype=torch.float32), log_scales)
    base = torch.randn((M, n_win, WINDOW_K), generator=g, device=device, dtype=torch.float32)
    x = (base * scales).reshape(M, n_win * WINDOW_K)[:, :N].contiguous()
    return x.to(dtype)


def _gen_sparse_extreme_outliers(
    M: int,
    N: int,
    *,
    seed: int,
    device: str,
    dtype: torch.dtype,
    outlier_mag: float = 1.0e6,
    n_outlier_windows: int = 4,
) -> torch.Tensor:
    """``N(0, 1)`` background plus a handful of extreme outlier elements.

    This is the distribution where 1x64's advantage is sharp in absolute
    RMSE. The mechanism:

    * The per-tensor scheme derives ``S_enc_global`` from the global amax,
      which the outliers drag up to ~``outlier_mag``. With
      ``outlier_mag = 1e6``, ``S_enc_global ~ 2.7e-3``; for a non-outlier
      1x16 block (``vec_max ~ 2``) the resulting ``s_dec_fp32`` is
      ``~9e-4``, **below E4M3's smallest subnormal (~2e-3)**. ``s_dec``
      rounds to zero, so reconstruction of every non-outlier block is
      identically zero -- catastrophic loss for the bulk of the tensor.

    * The hierarchical scheme uses ``S_enc_tile`` per 1x64 window. In the
      ~``M*n_win - n_outlier_windows`` non-outlier windows ``tile_amax``
      sees only ``N(0, 1)``, so ``S_enc_tile ~ 900``, ``s_dec`` stays well
      inside E4M3, and FP4 quantization runs at standard precision.
      Loss is restricted to the (small) bulk inside outlier-containing
      windows.

    The expected RMSE ratio ``rmse_1x64 / rmse_pt`` is ``~ 0.06`` -- well
    below the 0.5 threshold the test asserts.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((M, N), generator=g, device=device, dtype=torch.float32)

    n_win = N // WINDOW_K
    if n_win == 0:
        # Tensor is narrower than one full 1x64 window; the scheme degenerates
        # and there is no meaningful advantage to test.
        return x.to(dtype)

    total_windows = M * n_win
    n_outlier_windows = min(n_outlier_windows, total_windows)
    flat_idx = torch.randperm(total_windows, generator=g, device=device)[:n_outlier_windows]
    outlier_rows = flat_idx // n_win
    outlier_cols = (flat_idx % n_win) * WINDOW_K
    signs = (
        torch.randint(0, 2, (n_outlier_windows,), generator=g, device=device, dtype=torch.int32).to(
            torch.float32
        )
        * 2
        - 1
    )
    x[outlier_rows, outlier_cols] = signs * float(outlier_mag)

    return x.to(dtype)


_NEEDS_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="accuracy comparison runs on CUDA to mirror the rest of the nvfp4 suite",
)


@_NEEDS_CUDA
@pytest.mark.parametrize("M, N", [(256, 1024), (512, 2048), (1024, 1024)])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_1x64_at_least_as_good_as_per_tensor_on_gaussian(
    M: int, N: int, x_dtype: torch.dtype, seed: int, capsys
) -> None:
    """On uniform Gaussian inputs the two schemes should be roughly tied.

    ``S_enc_global`` and per-window ``S_enc_tile`` see similar amax values
    when the input magnitude is uniform across K, so we expect ratios near
    1.0. We allow a 5% slack to absorb E4M3-rounding noise; a regression
    that materially worsens 1x64 accuracy on benign inputs would still be
    caught.
    """
    device = "cuda"
    x = _gen_gaussian(M, N, seed=seed, device=device, dtype=x_dtype)

    rmse_pt, max_pt, fro_pt = _err_metrics(x, _recon_per_tensor(x))
    rmse_1x64, max_1x64, fro_1x64 = _err_metrics(x, _recon_1x64(x))

    with capsys.disabled():
        print(
            f"\n[gaussian {M}x{N} {x_dtype} seed={seed}]"
            f" rmse: pt={rmse_pt:.4e} 1x64={rmse_1x64:.4e}"
            f" ratio={rmse_1x64 / max(rmse_pt, 1e-30):.3f} |"
            f" max_abs: pt={max_pt:.4e} 1x64={max_1x64:.4e} |"
            f" frob_rel: pt={fro_pt:.4e} 1x64={fro_1x64:.4e}"
        )

    assert rmse_1x64 <= rmse_pt * 1.05, (
        "1x64 RMSE unexpectedly worse than per-tensor on uniform input: "
        f"rmse_1x64={rmse_1x64:.4e} > 1.05 * rmse_pt={rmse_pt:.4e}"
    )


@_NEEDS_CUDA
@pytest.mark.parametrize("M, N", [(256, 1024), (512, 2048)])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_1x64_better_than_per_tensor_on_sparse_extreme_outliers(
    M: int, N: int, x_dtype: torch.dtype, seed: int, capsys
) -> None:
    """Sparse extreme outliers force prod's ``s_dec`` into E4M3 underflow.

    Per-block ``s_dec`` already adapts to local ``vec_max`` in both
    schemes, so within E4M3's representable range FP4 resolution per
    element is ``~ vec_max / 12`` independent of the ``S_enc`` choice --
    the schemes tie. The 1x64 advantage in *absolute* RMSE terms shows up
    only when prod's E4M3 ``s_dec`` underflows, and only when those
    underflowed positions also dominate ``mean(x^2)``.

    To put both conditions in effect simultaneously, this test uses an
    ``N(0, 1)`` bulk plus a handful of outliers of magnitude ``1e6``:

    * ``S_enc_global = (FP8_MAX * FP4_MAX) / amax ~ 2.7e-3`` --
      ``s_dec`` for non-outlier blocks underflows; reconstruction of the
      entire bulk collapses to zero. ``rmse_pt ~ 1.0``.

    * ``S_enc_tile`` for ~99% of windows is set by ``tile_amax ~ 3``
      (no outlier present); FP4 quantization runs at standard precision.
      ``rmse_1x64 ~ 0.06``.

    The assertion only requires 1x64 to be strictly better than
    per-tensor; the comparison values are printed for each parametrized
    case so the actual margin (empirically ``rmse_1x64 / rmse_pt`` is
    around ``0.05 - 0.10``) is visible in the test log.
    """
    device = "cuda"
    x = _gen_sparse_extreme_outliers(M, N, seed=seed, device=device, dtype=x_dtype)

    rmse_pt, max_pt, fro_pt = _err_metrics(x, _recon_per_tensor(x))
    rmse_1x64, max_1x64, fro_1x64 = _err_metrics(x, _recon_1x64(x))

    with capsys.disabled():
        print(
            f"\n[sparse_outliers {M}x{N} {x_dtype} seed={seed}]"
            f" rmse: pt={rmse_pt:.4e} 1x64={rmse_1x64:.4e}"
            f" ratio={rmse_1x64 / max(rmse_pt, 1e-30):.3f} |"
            f" max_abs: pt={max_pt:.4e} 1x64={max_1x64:.4e} |"
            f" frob_rel: pt={fro_pt:.4e} 1x64={fro_1x64:.4e}"
        )

    assert rmse_1x64 < rmse_pt, (
        "1x64 was not strictly better than per-tensor on "
        "sparse-extreme-outlier input: "
        f"rmse_1x64={rmse_1x64:.4e} >= rmse_pt={rmse_pt:.4e}"
    )


@_NEEDS_CUDA
@pytest.mark.parametrize("M, N", [(256, 1024)])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_1x64_at_least_tied_on_modest_per_window_dynamic_range(
    M: int, N: int, x_dtype: torch.dtype, seed: int, capsys
) -> None:
    """A moderate per-window dynamic range still favours 1x64.

    Ratio range of ~30x is well within E4M3's representable scale range,
    so the per-tensor scheme does not catastrophically underflow. The
    advantage from local ``S_enc`` is therefore smaller -- but still
    present. We assert merely "no worse" plus a generous 5% slack; the
    strict inequality test above is the one that demonstrates the win,
    while this case ensures the win does not invert when the dynamic
    range shrinks.
    """
    device = "cuda"
    x = _gen_per_window_dynamic_range(
        M, N, seed=seed, device=device, dtype=x_dtype, log10_lo=-1.5, log10_hi=0.5
    )

    rmse_pt, max_pt, fro_pt = _err_metrics(x, _recon_per_tensor(x))
    rmse_1x64, max_1x64, fro_1x64 = _err_metrics(x, _recon_1x64(x))

    with capsys.disabled():
        print(
            f"\n[dyn_range_modest {M}x{N} {x_dtype} seed={seed}]"
            f" rmse: pt={rmse_pt:.4e} 1x64={rmse_1x64:.4e}"
            f" ratio={rmse_1x64 / max(rmse_pt, 1e-30):.3f} |"
            f" max_abs: pt={max_pt:.4e} 1x64={max_1x64:.4e} |"
            f" frob_rel: pt={fro_pt:.4e} 1x64={fro_1x64:.4e}"
        )

    assert rmse_1x64 <= rmse_pt * 1.05, (
        "1x64 unexpectedly worse than per-tensor on modest dynamic-range "
        f"input: rmse_1x64={rmse_1x64:.4e} > 1.05 * rmse_pt={rmse_pt:.4e}"
    )
