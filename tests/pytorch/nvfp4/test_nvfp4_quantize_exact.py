# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.nvfp4_tensor import (
    NVFP4Quantizer,
)
from transformer_engine.pytorch.experimental.quantization_microblock_ref import NVFP4QuantizerRef
from transformer_engine.pytorch.experimental import utils
from transformer_engine.pytorch.fp8 import fp8_autocast, get_fp4_te_dtype


recipe_available, reason_for_no_recipe = FP8GlobalStateManager.is_nvfp4_available()


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    swizzled_scale: bool,
    use_cpp_allocator: bool,
    with_2d_quantization: bool,
) -> None:
    te_dtype = tex.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # Quantize
    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=with_2d_quantization,
    )
    if use_cpp_allocator:
        x_nvfp4_sut = nvfp4_quantizer(x)
    else:
        x_nvfp4_sut = nvfp4_quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    # Extract data from NVFP4Tensor
    assert x_nvfp4_sut._rowwise_data is not None
    qx: torch.Tensor = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise

    # Reference quantization
    quant_tile_shape = (1, 16) if not with_2d_quantization else (16, 16)
    ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=quant_tile_shape,
    )
    x_nvfp4_ref = ref_quantizer.quantize(x)

    # Extract data from RefNVFP4Tensor
    qx_ref = (
        unpack_fp4(x_nvfp4_ref.data.view(dtype=torch.uint8))
        if x_nvfp4_ref.data is not None
        else None
    )
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    qx_t_ref = (
        unpack_fp4(x_nvfp4_ref.data_t.view(dtype=torch.uint8))
        if x_nvfp4_ref.data_t is not None
        else None
    )
    sx_t_ref = (
        x_nvfp4_ref.scale_t.view(dtype=torch.uint8) if x_nvfp4_ref.scale_t is not None else None
    )
    ref_amax = x_nvfp4_ref.global_amax_row

    qx = unpack_fp4(qx)
    qx_t = unpack_fp4(qx_t) if qx_t is not None else None

    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    # Compare only the valid portion of scale tensors (reference may not have padding)
    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]

    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)

        # Compare only the valid portion of transpose scale tensors
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # full tile cases
        (128, 128),
        (256, 256),
        (256, 1024),
        (1024, 256),
        # Padding required cases
        (256, 272),
        (304, 304),
        (320, 256),
        # Some larger tiles
        (2048, 2048),
        (1024, 2048),
        (2048, 1024),
        # # largest tile
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "return_transpose", [True, False], ids=["quantize_transpose", "skip_transpose"]
)
@pytest.mark.parametrize("swizzled_scale", [False], ids=["linear_scale"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize(
    "with_2d_quantization", [True, False], ids=["2d_quantization", "1d_quantization"]
)
def test_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    swizzled_scale: bool,
    use_cpp_allocator: bool,
    with_2d_quantization: bool,
) -> None:
    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_transpose=return_transpose,
        swizzled_scale=swizzled_scale,
        use_cpp_allocator=use_cpp_allocator,
        with_2d_quantization=with_2d_quantization,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("extrema_high", [False, True], ids=["zeros", "maxes"])
@pytest.mark.parametrize(
    "return_transpose", [True, False], ids=["quantize_transpose", "skip_transpose"]
)
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
def test_nvfp4_quantization_extrema_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    extrema_high: bool,
    return_transpose: bool,
    use_cpp_allocator: bool,
):
    te_dtype = tex.DType.kFloat4E2M1

    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if extrema_high:
        x = torch.full((M, N), torch.finfo(x_dtype).max, dtype=x_dtype, device=device)
    else:
        x = torch.zeros((M, N), dtype=x_dtype, device=device)

    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    if use_cpp_allocator:
        x_nvfp4_sut = nvfp4_quantizer(x)
    else:
        x_nvfp4_sut = nvfp4_quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    assert x_nvfp4_sut._rowwise_data is not None
    qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise

    ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
    )
    x_nvfp4_ref = ref_quantizer.quantize(x)

    qx_ref = x_nvfp4_ref.data.view(dtype=torch.uint8) if x_nvfp4_ref.data is not None else None
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    qx_t_ref = (
        x_nvfp4_ref.data_t.view(dtype=torch.uint8) if x_nvfp4_ref.data_t is not None else None
    )
    sx_t_ref = (
        x_nvfp4_ref.scale_t.view(dtype=torch.uint8) if x_nvfp4_ref.scale_t is not None else None
    )
    ref_amax = x_nvfp4_ref.global_amax_row

    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (16, 128),
        (32, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "return_transpose", [True, False], ids=["quantize_transpose", "skip_transpose"]
)
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
def test_nvfp4_quantization_boundary_values(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
):
    """
    Stress rounding/threshold behavior by placing values just below/above
    many potential bin edges within each 16-element microblock.
    Validates native vs reference byte-for-byte and scale parity.
    """
    te_dtype = tex.DType.kFloat4E2M1

    device = "cuda"
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Construct a single row with paired boundary values: v-eps, v+eps
    # spanning a wide dynamic range to exercise clipping and multiple bins.
    # Ensure even N and N is multiple of 16 for microblocks, which holds for 128.
    base = torch.linspace(-12.0, 12.0, steps=N // 2, dtype=torch.float32, device=device)
    eps = torch.full_like(base, 1e-3)
    # Avoid zero eps for very small magnitudes
    eps = torch.maximum(eps, 1e-4 * torch.ones_like(base))
    lower = base - eps
    upper = base + eps
    row = torch.empty(N, dtype=torch.float32, device=device)
    row[0::2] = lower
    row[1::2] = upper
    x = row.unsqueeze(0).repeat(M, 1).to(dtype=x_dtype)

    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    if use_cpp_allocator:
        x_nvfp4_sut = nvfp4_quantizer(x)
    else:
        x_nvfp4_sut = nvfp4_quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    assert x_nvfp4_sut._rowwise_data is not None
    qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise

    ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
    )
    x_nvfp4_ref = ref_quantizer.quantize(x)

    qx_ref = x_nvfp4_ref.data.view(dtype=torch.uint8) if x_nvfp4_ref.data is not None else None
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    qx_t_ref = (
        x_nvfp4_ref.data_t.view(dtype=torch.uint8) if x_nvfp4_ref.data_t is not None else None
    )
    sx_t_ref = (
        x_nvfp4_ref.scale_t.view(dtype=torch.uint8) if x_nvfp4_ref.scale_t is not None else None
    )
    ref_amax = x_nvfp4_ref.global_amax_row

    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    # Compare only valid portion of scales (trim any padding)
    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (32, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "return_transpose", [True, False], ids=["quantize_transpose", "skip_transpose"]
)
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
def test_nvfp4_quantization_noncontiguous_inputs(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
):
    te_dtype = tex.DType.kFloat4E2M1

    device = "cuda"
    seed = 17
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Start from a contiguous tensor, then make a non-contiguous view by transpose
    x_base = torch.randn((M, N), dtype=x_dtype, device=device)
    x_nc = x_base.t()  # shape (N, M), non-contiguous
    assert not x_nc.is_contiguous()

    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    if use_cpp_allocator:
        x_nvfp4_sut = nvfp4_quantizer(x_nc)
    else:
        x_nvfp4_sut = nvfp4_quantizer.make_empty(
            x_nc.shape, dtype=x_dtype, device=device, requires_grad=False
        )
        x_nvfp4_sut = nvfp4_quantizer.update_quantized(x_nc, x_nvfp4_sut)

    assert x_nvfp4_sut._rowwise_data is not None
    qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise

    ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
    )
    x_nvfp4_ref = ref_quantizer.quantize(x_nc)

    qx_ref = x_nvfp4_ref.data.view(dtype=torch.uint8) if x_nvfp4_ref.data is not None else None
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    qx_t_ref = (
        x_nvfp4_ref.data_t.view(dtype=torch.uint8) if x_nvfp4_ref.data_t is not None else None
    )
    sx_t_ref = (
        x_nvfp4_ref.scale_t.view(dtype=torch.uint8) if x_nvfp4_ref.scale_t is not None else None
    )
    ref_amax = x_nvfp4_ref.global_amax_row

    # Quantized must match
    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    # Compare only valid portion of scales (trim padding)
    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)
