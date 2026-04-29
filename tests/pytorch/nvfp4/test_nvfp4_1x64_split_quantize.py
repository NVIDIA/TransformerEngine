# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bit-exact tests for NVFP4 1x64 local-encode through ``tex.split_quantize``.

Together with ``test_nvfp4_1x64_quantize_exact.py`` (which covers single-tensor
quantize) these tests lock down the multi-chunk dispatch path that drives
``split_quantize_nvfp4_impl_helper`` (and its ``UNFUSED`` fallback) when the
1x64 local-encode flag is on. Two complementary axes are exercised:

* **Axis A -- algorithm oracle.**
  Compare the SUT (``tex.split_quantize``) against a pure-PyTorch oracle
  (``NVFP4Quantizer1x64Ref``) byte-for-byte, applied per chunk. The oracle is
  independent of any TransformerEngine CUDA kernel, so any deviation must
  come from the kernel itself (rounding, scale derivation, packed FP4
  layout). This catches algorithmic regressions even when both C++
  dispatch paths are wired identically.

* **Axis B -- wiring oracle.**
  Compare the SUT against ``quantizers[i](chunk)`` running per chunk via
  ``reference_group_quantize``. Both sides ultimately invoke the same
  ``quantize_1x64_local_encode`` kernel, so their outputs must be
  bit-identical. Any mismatch points at the split-quantize driver itself --
  buffer allocation (``BULK_NVFP4`` vs. ``UNFUSED``), per-chunk config
  propagation, ``nvte_quantize_v2`` arguments, or the inner-dim alignment
  fallback in the dispatcher.

Both axes use ``atol=rtol=0``: the 1x64 path is deterministic (no
stochastic rounding, no RHT random sign mask), so any non-zero diff is a
real regression.
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_1x64 import (
    NVFP4Quantizer1x64Ref,
)

from nvfp4_utils import (
    assert_same_shape_and_dtype,
    generate_split_sections,
    get_nvfp4_scale_shape_no_padding,
    reference_group_quantize,
)


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


# -----------------------------------------------------------------------------
# Shape matrix
#
# 1x64 hierarchy requires both M and N to be multiples of 64. The split-quantize
# dispatcher additionally has two routing branches that we want to exercise:
#
#   * N % 128 == 0 -> ``QuantizationMethod::FUSED_NVFP4`` keeps the call on the
#     fused ``split_quantize_nvfp4_impl_helper`` path.
#   * N % 128 != 0 -> dispatcher mirrors the ``BULK_NVFP4`` ``% 128`` fallback
#     and downgrades to ``QuantizationMethod::UNFUSED`` (per-tensor
#     ``NVFP4Quantizer::quantize_impl`` loop). The kernel itself is the same;
#     only the driver changes.
#
# Both branches are kept here so that a regression in either lane fails CI.
# -----------------------------------------------------------------------------
_SHAPES_FUSED_PATH = [
    (256, 1024),
    (1024, 256),
    (2048, 2048),
]
_SHAPES_FALLBACK_PATH = [
    (1024, 320),  # N % 128 != 0, hits the dispatcher's UNFUSED fallback.
]
_SHAPES = _SHAPES_FUSED_PATH + _SHAPES_FALLBACK_PATH


def _unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Unpack two FP4 values per byte into one ``uint8`` per element."""
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def _make_quantizers(num: int, *, rowwise: bool, columnwise: bool):
    """Build a list of ``NVFP4Quantizer`` configured for 1x64-compatible mode.

    1x64 is mutually exclusive with RHT and 2D quantization; the constructor
    flags here mirror that constraint so we never feed an invalid combination
    into either dispatch path.
    """
    return [
        NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=rowwise,
            columnwise=columnwise,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
        )
        for _ in range(num)
    ]


def _make_input(M: int, N: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    return torch.randn((M, N), dtype=dtype, device="cuda")


# -----------------------------------------------------------------------------
# Axis A: algorithm oracle. tex.split_quantize  vs  NVFP4Quantizer1x64Ref
# -----------------------------------------------------------------------------
def _check_axis_a_against_oracle(
    *,
    M: int,
    N: int,
    dtype: torch.dtype,
    split_sections: list[int],
    rowwise: bool,
    columnwise: bool,
) -> None:
    x = _make_input(M, N, dtype)
    chunks = torch.split(x, split_sections)
    quantizers = _make_quantizers(len(split_sections), rowwise=rowwise, columnwise=columnwise)

    sut_outputs = tex.split_quantize(x, split_sections, quantizers)

    for i, (sut, chunk) in enumerate(zip(sut_outputs, chunks)):
        if split_sections[i] == 0:
            # Empty chunk: SUT just allocates zero-row buffers. The oracle's
            # behaviour on an empty input is irrelevant -- we only need to
            # know the kernel did not write past the chunk boundary.
            if rowwise:
                assert sut._rowwise_data.shape[0] == 0, f"chunk {i} rowwise data not empty"
                assert sut._amax_rowwise.shape[0] == 0, f"chunk {i} rowwise amax not empty"
            if columnwise:
                # columnwise tensors are transposed: (N, M_chunk//2) etc.
                assert sut._columnwise_data.shape[1] == 0, f"chunk {i} columnwise data not empty"
                assert sut._amax_columnwise.shape[1] == 0, f"chunk {i} columnwise amax not empty"
            continue

        ref = NVFP4Quantizer1x64Ref(rowwise=rowwise, columnwise=columnwise).quantize(chunk)

        if rowwise:
            # Kernel pads qx/sx to alignment boundaries; only compare the
            # un-padded prefix, exactly as test_nvfp4_1x64_quantize_exact.py
            # does for the single-tensor path.
            qx_sut = _unpack_fp4(sut._rowwise_data.view(dtype=torch.uint8))
            qx_ref = _unpack_fp4(ref.data.view(dtype=torch.uint8))
            sx_sut = sut._rowwise_scale_inv.view(dtype=torch.uint8)
            sx_ref = ref.scale.view(dtype=torch.uint8)
            torch.testing.assert_close(
                qx_sut[: qx_ref.shape[0], : qx_ref.shape[1]], qx_ref, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                sx_sut[: sx_ref.shape[0], : sx_ref.shape[1]], sx_ref, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(sut._amax_rowwise, ref.window_amax_row, atol=0.0, rtol=0.0)

        if columnwise:
            qxt_sut = _unpack_fp4(sut._columnwise_data.view(dtype=torch.uint8))
            qxt_ref = _unpack_fp4(ref.columnwise_data.view(dtype=torch.uint8))
            sxt_sut = sut._columnwise_scale_inv.view(dtype=torch.uint8)
            sxt_ref = ref.columnwise_scale.view(dtype=torch.uint8)
            torch.testing.assert_close(
                qxt_sut[: qxt_ref.shape[0], : qxt_ref.shape[1]], qxt_ref, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                sxt_sut[: sxt_ref.shape[0], : sxt_ref.shape[1]], sxt_ref, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                sut._amax_columnwise, ref.window_amax_col, atol=0.0, rtol=0.0
            )


# -----------------------------------------------------------------------------
# Axis B: wiring oracle. tex.split_quantize  vs  per-chunk quantizer(chunk)
# -----------------------------------------------------------------------------
def _check_axis_b_against_per_tensor(
    *,
    M: int,
    N: int,
    dtype: torch.dtype,
    split_sections: list[int],
    rowwise: bool,
    columnwise: bool,
) -> None:
    x = _make_input(M, N, dtype)
    chunks = torch.split(x, split_sections)

    # Build TWO independent quantizer lists -- one drives the SUT, one drives
    # the per-tensor reference -- to keep the comparison strictly between the
    # two C++ dispatch paths and not accidentally reuse any quantizer state.
    sut_quantizers = _make_quantizers(len(split_sections), rowwise=rowwise, columnwise=columnwise)
    ref_quantizers = _make_quantizers(len(split_sections), rowwise=rowwise, columnwise=columnwise)

    sut_outputs = tex.split_quantize(x, split_sections, sut_quantizers)

    # ``reference_group_quantize`` calls ``ref_quantizers[i](chunk)`` per
    # chunk -- with the 1x64 env flag on, this still runs the same
    # ``quantize_1x64_local_encode`` kernel, just driven through the
    # single-tensor entry point. SUT and reference must agree bit-for-bit
    # on every byte the kernel actually writes.
    qx_ref, sx_ref, amax_row_ref, qxt_ref, sxt_ref, amax_col_ref = reference_group_quantize(
        x, ref_quantizers, split_sections, rowwise, columnwise
    )

    # NVFP4 scale buffers are over-allocated (rounded up to the cuBLAS
    # block-scaling-factor layout: 128 in the outer dim, 4 in the inner dim)
    # so that swizzle can run in place. The kernel only writes the un-padded
    # prefix; the padded tail is left in whatever state ``at::empty``
    # returned, which differs across allocations. Slice both sides down to
    # the valid prefix before doing a bit-exact compare. Data buffers
    # ``(M, N/2)`` and 1x64 amax buffers ``(M, N/64)`` / ``(N, M/64)`` are
    # already exact-sized, so they can be compared whole.
    if rowwise:
        sut_qx = [out._rowwise_data.view(dtype=torch.uint8) for out in sut_outputs]
        sut_sx = [out._rowwise_scale_inv for out in sut_outputs]
        sut_amax = [out._amax_rowwise for out in sut_outputs]
        for i in range(len(sut_outputs)):
            if split_sections[i] == 0:
                assert_same_shape_and_dtype(sut_qx[i], qx_ref[i])
                assert_same_shape_and_dtype(sut_sx[i], sx_ref[i])
                assert_same_shape_and_dtype(sut_amax[i], amax_row_ref[i])
                continue
            torch.testing.assert_close(sut_qx[i], qx_ref[i], atol=0.0, rtol=0.0)
            torch.testing.assert_close(sut_amax[i], amax_row_ref[i], atol=0.0, rtol=0.0)
            valid = get_nvfp4_scale_shape_no_padding(chunks[i].shape, columnwise=False)
            torch.testing.assert_close(
                sut_sx[i][: valid[0], : valid[1]],
                sx_ref[i][: valid[0], : valid[1]],
                atol=0.0,
                rtol=0.0,
            )

    if columnwise:
        sut_qxt = [out._columnwise_data.view(dtype=torch.uint8) for out in sut_outputs]
        sut_sxt = [out._columnwise_scale_inv for out in sut_outputs]
        sut_amax_t = [out._amax_columnwise for out in sut_outputs]
        for i in range(len(sut_outputs)):
            if split_sections[i] == 0:
                assert_same_shape_and_dtype(sut_qxt[i], qxt_ref[i])
                assert_same_shape_and_dtype(sut_sxt[i], sxt_ref[i])
                assert_same_shape_and_dtype(sut_amax_t[i], amax_col_ref[i])
                continue
            torch.testing.assert_close(sut_qxt[i], qxt_ref[i], atol=0.0, rtol=0.0)
            torch.testing.assert_close(sut_amax_t[i], amax_col_ref[i], atol=0.0, rtol=0.0)
            valid = get_nvfp4_scale_shape_no_padding(chunks[i].shape, columnwise=True)
            torch.testing.assert_close(
                sut_sxt[i][: valid[0], : valid[1]],
                sxt_ref[i][: valid[0], : valid[1]],
                atol=0.0,
                rtol=0.0,
            )


# -----------------------------------------------------------------------------
# Pytest entry points
# -----------------------------------------------------------------------------
@pytest.fixture
def _enable_1x64(monkeypatch):
    """Enable 1x64 local-encode for both the SUT and the reference paths.

    The flag is read at every quantize call (it gates ``local_encode_from_env``
    in ``nvfp4_1x64.h``), so it must be set before any tensor flows through
    either ``tex.split_quantize`` or ``quantizer(chunk)``. ``monkeypatch``
    scopes the change to a single test, so other tests in the same session
    are unaffected.
    """
    monkeypatch.setenv("NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT", "1")


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", _SHAPES)
@pytest.mark.parametrize(
    "edge_cases",
    [
        "regular",
        "zero_tokens_front",
        "zero_tokens_middle",
        "random_uneven_split",
    ],
)
@pytest.mark.parametrize(
    "rowwise, columnwise",
    [(True, False), (False, True), (True, True)],
    ids=["row", "col", "rowcol"],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
def test_split_quantize_1x64_axis_a_oracle(
    _enable_1x64,
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
    rowwise: bool,
    columnwise: bool,
) -> None:
    """Axis A: split-quantize output must equal the pure-PyTorch 1x64 oracle."""
    split_sections = generate_split_sections(M, N, edge_cases, least_multiple=64)
    _check_axis_a_against_oracle(
        M=M,
        N=N,
        dtype=x_dtype,
        split_sections=split_sections,
        rowwise=rowwise,
        columnwise=columnwise,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", _SHAPES)
@pytest.mark.parametrize(
    "edge_cases",
    [
        "regular",
        "zero_tokens_front",
        "zero_tokens_middle",
        "random_uneven_split",
    ],
)
@pytest.mark.parametrize(
    "rowwise, columnwise",
    [(True, False), (False, True), (True, True)],
    ids=["row", "col", "rowcol"],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
def test_split_quantize_1x64_axis_b_wiring(
    _enable_1x64,
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
    rowwise: bool,
    columnwise: bool,
) -> None:
    """Axis B: split-quantize and per-tensor driver must produce identical bytes."""
    split_sections = generate_split_sections(M, N, edge_cases, least_multiple=64)
    _check_axis_b_against_per_tensor(
        M=M,
        N=N,
        dtype=x_dtype,
        split_sections=split_sections,
        rowwise=rowwise,
        columnwise=columnwise,
    )
