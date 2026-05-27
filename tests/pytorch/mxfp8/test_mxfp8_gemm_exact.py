# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Layout-by-layout exactness test for MXFP8 single GEMM.

Each `te.Linear` forward + backward issues three cuBLAS GEMMs (see
`transformer_engine/pytorch/module/linear.py`):

  layout  role     A                     B                      Out
  ------  -------  --------------------  ---------------------  -------------------
  TN      fwd      (out_f, in_f)         (batch, in_f)          (batch, out_f)
  NN      dgrad    (out_f, in_f)         (batch, out_f)         (batch, in_f)
  NT      wgrad    (batch, in_f)         (batch, out_f)         (out_f, in_f)

This test drives the underlying ``general_gemm`` call directly for each layout
with MXFP8-quantized operands, and compares the cuBLAS output to a
dequantized-operand reference matmul.

Background: With cuBLAS 13.5.1.27 on sm_120, MXFP8 GEMM was only supported in
the TN layout; NN and NT returned ``CUBLAS_STATUS_NOT_SUPPORTED`` from
``cublasLtMatmulAlgoGetHeuristic``. cuBLAS 13.6.0.2 adds NN/NT support on
sm_120; both layouts then run end-to-end and match the dequantized reference
within MXFP8 tolerance. See ``Testing/cublas_logs/README.md`` and
``Testing/repro_mxfp8_layouts.cu`` in this repo for a layout-by-layout cuBLAS
reproducer. NN/NT are marked ``strict=True`` xfail only when the loaded
cuBLASLt is below the version that adds support, so the suite automatically
flags an XPASS once cuBLAS is upgraded in-place.
"""

from __future__ import annotations

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm


_MXFP8_AVAILABLE, _MXFP8_SKIP_REASON = te.is_mxfp8_available(return_reason=True)

CUBLASLT_MXFP8_FULL_LAYOUTS_SM120 = 130600
_CUBLASLT_VERSION = tex.get_cublasLt_version()


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (12, 0)


def _needs_sm120_non_tn_xfail() -> bool:
    """NN/NT MXFP8 is unsupported by cuBLAS on sm_120 below cuBLASLt 13.6.0.2."""
    return _is_sm120() and _CUBLASLT_VERSION < CUBLASLT_MXFP8_FULL_LAYOUTS_SM120


_SM120_NON_TN_XFAIL = pytest.mark.xfail(
    _needs_sm120_non_tn_xfail(),
    strict=True,
    reason=(
        f"MXFP8 NN/NT GEMM is not supported by cuBLAS on sm_120 below cuBLASLt"
        f" {CUBLASLT_MXFP8_FULL_LAYOUTS_SM120} (loaded={_CUBLASLT_VERSION});"
        " cublasLtMatmulAlgoGetHeuristic returns CUBLAS_STATUS_NOT_SUPPORTED."
        " Upgrade cuBLAS to ≥ 13.6.0.2 to unblock these layouts."
    ),
)


def _quantize_mxfp8(t: torch.Tensor):
    """MXFP8 quantize with both row-wise and column-wise data populated."""
    return MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )(t)


def _reference_for_layout(
    layout: str,
    w_q,
    x_q,
    dy_q,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference matmul computed in fp32 from dequantized operands.

    Mirrors the three GEMMs done by ``te.Linear`` (see module docstring).
    """
    if layout == "TN":  # fwd: x @ w.T -> (batch, out_f)
        a = w_q.dequantize(dtype=torch.float32)
        b = x_q.dequantize(dtype=torch.float32)
        return (b @ a.T).to(out_dtype)
    if layout == "NN":  # dgrad: dy @ w -> (batch, in_f)
        a = w_q.dequantize(dtype=torch.float32)
        b = dy_q.dequantize(dtype=torch.float32)
        return (b @ a).to(out_dtype)
    if layout == "NT":  # wgrad: dy.T @ x -> (out_f, in_f)
        a = x_q.dequantize(dtype=torch.float32)
        b = dy_q.dequantize(dtype=torch.float32)
        return (b.T @ a).to(out_dtype)
    raise ValueError(f"Unknown layout {layout!r}")


def _shapes_for_layout(layout: str, w_q, x_q, dy_q):
    if layout == "TN":
        return w_q, x_q
    if layout == "NN":
        return w_q, dy_q
    if layout == "NT":
        return x_q, dy_q
    raise ValueError(f"Unknown layout {layout!r}")


# Shape triples are (batch, in_features, out_features), all multiples of 32 as
# required by the MXFP8 quantizer (32-element scaling blocks).
_SHAPES = [
    (32, 32, 64),
    (128, 128, 128),
    (256, 1024, 512),
    (2048, 2048, 8192),
]


@pytest.mark.skipif(not _MXFP8_AVAILABLE, reason=_MXFP8_SKIP_REASON)
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param("TN", id="TN_fwd"),
        pytest.param("NN", id="NN_dgrad", marks=_SM120_NON_TN_XFAIL),
        pytest.param("NT", id="NT_wgrad", marks=_SM120_NON_TN_XFAIL),
    ],
)
@pytest.mark.parametrize(
    "batch, in_features, out_features",
    _SHAPES,
    ids=[f"b{b}_in{i}_out{o}" for (b, i, o) in _SHAPES],
)
@pytest.mark.parametrize(
    "in_dtype",
    [torch.bfloat16, torch.float16],
    ids=["bf16", "fp16"],
)
@pytest.mark.parametrize(
    "out_dtype",
    [torch.bfloat16, torch.float32],
    ids=["out_bf16", "out_fp32"],
)
def test_mxfp8_single_gemm_versus_reference(
    layout: str,
    batch: int,
    in_features: int,
    out_features: int,
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
):
    """One cuBLAS MXFP8 GEMM per layout, compared against a dequantized reference."""
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    w = torch.randn(out_features, in_features, dtype=in_dtype, device=device)
    x = torch.randn(batch, in_features, dtype=in_dtype, device=device)
    dy = torch.randn(batch, out_features, dtype=in_dtype, device=device)

    w_q = _quantize_mxfp8(w)
    x_q = _quantize_mxfp8(x)
    dy_q = _quantize_mxfp8(dy)

    A, B = _shapes_for_layout(layout, w_q, x_q, dy_q)
    out, *_ = general_gemm(A, B, out_dtype=out_dtype, layout=layout)

    ref = _reference_for_layout(layout, w_q, x_q, dy_q, out_dtype)
    assert tuple(out.shape) == tuple(
        ref.shape
    ), f"shape mismatch: cuBLAS {tuple(out.shape)} vs ref {tuple(ref.shape)}"

    # MXFP8 tolerance, aligned with tests/pytorch/utils.py::quantization_tols("mxfp8")
    # which returns dtype_tols(kFloat8E4M3) == dict(rtol=0.125, atol=0.0675).
    torch.testing.assert_close(
        out.to(torch.float32),
        ref.to(torch.float32),
        atol=0.0675,
        rtol=0.125,
    )
