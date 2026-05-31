# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer, NVFP4Tensor


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _valid_rowwise_scale(scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Trim rowwise scale padding."""
    return scale[: shape[0], : (shape[1] + 15) // 16]


def _valid_columnwise_scale(scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Trim columnwise scale padding."""
    return scale[: shape[1], : (shape[0] + 15) // 16]


def _assert_same_nvfp4_tensors(
    actual: NVFP4Tensor,
    expected: NVFP4Tensor,
    shape: torch.Size,
    columnwise: bool,
) -> None:
    """Check public NVFP4 tensor accessors against another NVFP4 tensor."""
    assert actual.rowwise_data is not None
    assert actual.rowwise_scale_inv is not None
    assert actual.amax_rowwise is not None
    torch.testing.assert_close(actual.rowwise_data, expected.rowwise_data, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        _valid_rowwise_scale(actual.rowwise_scale_inv, shape),
        _valid_rowwise_scale(expected.rowwise_scale_inv, shape),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(actual.amax_rowwise, expected.amax_rowwise, atol=0.0, rtol=0.0)

    if columnwise:
        assert actual.columnwise_data is not None
        assert actual.columnwise_scale_inv is not None
        assert actual.amax_columnwise is not None
        torch.testing.assert_close(
            actual.columnwise_data, expected.columnwise_data, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            _valid_columnwise_scale(actual.columnwise_scale_inv, shape),
            _valid_columnwise_scale(expected.columnwise_scale_inv, shape),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            actual.amax_columnwise,
            expected.amax_columnwise,
            atol=0.0,
            rtol=0.0,
        )
    else:
        assert actual.columnwise_data is None
        assert actual.columnwise_scale_inv is None
        assert actual.amax_columnwise is None


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", [(64, 64), (128, 64)], ids=["64x64", "128x64"])
@pytest.mark.parametrize(
    "quantizer_kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param(
            {
                "rowwise": True,
                "columnwise": True,
                "with_2d_quantization": True,
            },
            id="2d",
        ),
        pytest.param(
            {
                "rowwise": True,
                "columnwise": False,
                "row_scaled_nvfp4": True,
            },
            id="row_scaled",
        ),
        pytest.param(
            {
                "rowwise": True,
                "columnwise": False,
                "nvfp4_use_4over6": True,
                "nvfp4_e4m3_max": 256,
                "nvfp4_4over6_err_mode": "MSE",
            },
            id="4over6",
        ),
    ],
)
def test_quantize_nvfp4_public_api_matches_quantizer(shape, quantizer_kwargs) -> None:
    """Public NVFP4 API should match direct NVFP4Quantizer usage."""

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    public = te.quantize_nvfp4(x, **quantizer_kwargs)
    direct_quantizer_kwargs = {
        "columnwise": False,
        "with_random_sign_mask": False,
        **quantizer_kwargs,
    }
    direct = NVFP4Quantizer(**direct_quantizer_kwargs)(x)

    assert isinstance(public, NVFP4Tensor)
    _assert_same_nvfp4_tensors(public, direct, x.shape, direct_quantizer_kwargs["columnwise"])

    assert public.nvfp4_use_4over6 == direct_quantizer_kwargs.get("nvfp4_use_4over6", False)
    assert public.nvfp4_e4m3_max == direct_quantizer_kwargs.get("nvfp4_e4m3_max", 448)
