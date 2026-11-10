# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""SM12x scale-layout fidelity and allocation gates for fused NVFP4 quantization."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer

from nvfp4_utils import swizzle_nvfp4_scale


@pytest.fixture(autouse=True)
def require_sm12x():
    """These kernels require an SM120 or SM121 device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("SM120/SM121 is required")
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        pytest.skip(reason)


@pytest.mark.parametrize("shape", [(128, 128), (256, 512), (384, 256)])
@pytest.mark.parametrize("direction", ["row", "column", "both"])
@pytest.mark.parametrize("kind", ["1d", "2d", "rht"])
@pytest.mark.parametrize("stochastic_rounding", [False, True])
@pytest.mark.parametrize("scale_dtype", [te.DType.kFloat8E4M3, te.DType.kFloat8UE5M3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sm12x_swizzled_outputs_match_compact_reference(
    shape, direction, kind, stochastic_rounding, scale_dtype, dtype
):
    """Check FP4 bytes, both amaxes and an independent scale permutation."""
    if kind == "rht" and dtype != torch.bfloat16:
        pytest.skip("NVFP4 RHT only supports BF16 input")
    torch.manual_seed(1234)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    compact = NVFP4Quantizer(
        rowwise=direction != "column",
        columnwise=direction != "row",
        with_2d_quantization=kind == "2d",
        with_rht=kind == "rht",
        with_post_rht_amax=kind == "rht",
        with_random_sign_mask=False,
        stochastic_rounding=stochastic_rounding,
        scale_dtype=scale_dtype,
    )
    compact.optimize_for_gemm = False
    fused = compact.copy()
    fused.optimize_for_gemm = True
    # Test allocation separately: the fallback also sets this flag after quantization.
    allocated = fused.make_empty(shape, dtype=x.dtype)
    assert allocated._with_gemm_swizzled_scales

    torch.manual_seed(5678)
    expected = compact(x)
    torch.manual_seed(5678)
    actual = fused(x)
    for columnwise, prefix in [(False, "rowwise"), (True, "columnwise")]:
        ref_data = getattr(expected, f"_{prefix}_data")
        if ref_data is None:
            assert getattr(actual, f"_{prefix}_data") is None
            continue
        assert torch.equal(getattr(actual, f"_{prefix}_data"), ref_data)
        suffix = "columnwise" if columnwise else "rowwise"
        assert torch.equal(getattr(actual, f"_amax_{suffix}"), getattr(expected, f"_amax_{suffix}"))
        reference_scales = swizzle_nvfp4_scale(
            *shape, getattr(expected, f"_{prefix}_scale_inv"), columnwise=columnwise
        )
        assert torch.equal(
            getattr(actual, f"_{prefix}_scale_inv").view(torch.uint8),
            reference_scales.view(torch.uint8),
        )


@pytest.mark.parametrize("kind", ["1d", "2d", "rht"])
@pytest.mark.parametrize(
    "shape,eligible", [((128, 128), True), ((64, 128), False), ((128, 192), False)]
)
def test_sm12x_in_kernel_swizzle_shape_gate(kind, shape, eligible):
    """Partial scale tiles must retain compact allocation and the existing fallback."""
    q = NVFP4Quantizer(
        with_2d_quantization=kind == "2d",
        with_rht=kind == "rht",
        with_post_rht_amax=kind == "rht",
    )
    q.optimize_for_gemm = True
    out = q.make_empty(shape, dtype=torch.bfloat16)
    assert out._with_gemm_swizzled_scales is eligible


@pytest.mark.parametrize("with_rht", [False, True])
@pytest.mark.parametrize("special_recipe", ["row_scaled", "4over6"])
def test_sm12x_special_recipes_keep_compact_allocation(with_rht, special_recipe):
    """Unsupported recipes must be rejected before selecting the RHT scale layout."""
    q = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_rht=with_rht,
        with_post_rht_amax=with_rht,
        row_scaled_nvfp4=special_recipe == "row_scaled",
        nvfp4_use_4over6=special_recipe == "4over6",
    )
    q.optimize_for_gemm = True
    out = q.make_empty((128, 128), dtype=torch.bfloat16)
    assert not out._with_gemm_swizzled_scales
