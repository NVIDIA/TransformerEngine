# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer


def _valid_scales(output, direction, rows, cols):
    scales = getattr(output, f"_{direction}_scale_inv").view(torch.uint8)
    if output._with_gemm_swizzled_scales:
        padded_rows, padded_cols = scales.shape
        scales = (
            scales.reshape(padded_rows // 128, padded_cols // 4, 32, 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(padded_rows, padded_cols)
        )
    return scales[:rows, : cols // 16]


@pytest.mark.skipif(not te.is_nvfp4_available(), reason="NVFP4 requires Blackwell")
@pytest.mark.parametrize("splits", [(128, 128), (64, 64), (0, 64, 192), (128, 0, 128), (0, 0)])
@pytest.mark.parametrize("directions", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("optimize_for_gemm", [False, True])
@pytest.mark.parametrize("disable_fusion", [False, True])
def test_host_split_rht_fallback(
    splits, directions, optimize_for_gemm, disable_fusion, monkeypatch
):
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT_CAST_FUSION", str(int(disable_fusion)))
    torch.manual_seed(123)
    x = torch.randn(sum(splits), 128, dtype=torch.bfloat16, device="cuda")
    quantizers = [
        NVFP4Quantizer(
            fp4_dtype=te.DType.kFloat4E2M1,
            rowwise=directions[0],
            columnwise=directions[1],
            with_rht=True,
            with_post_rht_amax=True,
            with_random_sign_mask=False,
        )
        for _ in splits
    ]
    for quantizer in quantizers:
        quantizer.optimize_for_gemm = optimize_for_gemm
    expected = [q(part) for q, part in zip(quantizers, x.split(splits))]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        actual = tex.split_quantize(x, splits, quantizers)
    # Neither the full fusion nor its per-expert fallback may call retired kernels.
    names = [event.name for event in prof.events()]
    assert not any("group_quantize_transpose_nvfp4_kernel" in name for name in names)
    assert not any("group_rht_gemm_device" in name for name in names)
    for rows, result, reference in zip(splits, actual, expected):
        for enabled, direction, amax, shape in (
            (directions[0], "rowwise", "_amax_rowwise", (rows, 128)),
            (directions[1], "columnwise", "_amax_columnwise", (128, rows)),
        ):
            if not enabled:
                continue
            data = getattr(result, f"_{direction}_data")
            ref_data = getattr(reference, f"_{direction}_data")
            assert data.shape == ref_data.shape
            if rows == 0:
                continue  # Empty-expert storage is not initialized by either path.
            assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
            assert torch.equal(getattr(result, amax), getattr(reference, amax))
            assert torch.equal(
                _valid_scales(result, direction, *shape),
                _valid_scales(reference, direction, *shape),
            )
