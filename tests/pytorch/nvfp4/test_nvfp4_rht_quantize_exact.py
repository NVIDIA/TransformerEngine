# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# NOTE: This file is dependent on the success of test_nvfp4_quantize_exact.py.
# Separate to make sure all the functionalities are working as expected.
# Otherwise reference implementation will get messy.

# Due to the structure of NVFP4Quantizer, we need to test the RHT functionality
# together with the quantization functionality.

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.common.recipe import NVFP4BlockScaling

import pytest
import torch

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    contiguous: bool,
    return_transpose: bool,
    use_cpp_allocator: bool,
    swizzled_scale: bool = False,
    hadamard_dimension: int = 16,
    with_rht: bool = True,
    with_post_rht_amax: bool = True,
    with_random_sign_mask: bool = True,
) -> None:
    assert with_rht and with_post_rht_amax, "RHT and post-RHT amax reduction must be enabled."

    te_dtype = tex.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    x = x.transpose(0, 1) if not contiguous else x

    # Quantize
    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=with_rht,
        with_post_rht_amax=with_post_rht_amax,
        with_random_sign_mask=with_random_sign_mask,
    )
    if use_cpp_allocator:
        x_nvfp4_sut = nvfp4_quantizer(x)
    else:
        x_nvfp4_sut = nvfp4_quantizer.make_empty(
            x.shape, dtype=x_dtype, device=device, requires_grad=False
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
    amax_rowwise = x_nvfp4_sut._amax_rowwise
    amax_colwise = x_nvfp4_sut._amax_columnwise

    qx = unpack_fp4(qx)
    qx_t = unpack_fp4(qx_t) if qx_t is not None else None

    # Reference quantization using NVFP4QuantizerRef with built-in RHT
    ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        with_rht=with_rht,
        with_random_sign_mask=with_random_sign_mask,
    )
    x_nvfp4_ref = ref_quantizer.quantize(x)
    # Extract data from RefNVFP4Tensor
    qx_ref = (
        unpack_fp4(x_nvfp4_ref.data.view(dtype=torch.uint8))
        if x_nvfp4_ref.data is not None
        else None
    )
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    ref_amax_rowwise = x_nvfp4_ref.global_amax_row

    if return_transpose:
        assert x_nvfp4_ref.data_t is not None
        assert x_nvfp4_ref.scale_t is not None
        qx_t_ref = unpack_fp4(x_nvfp4_ref.data_t.view(dtype=torch.uint8))
        sx_t_ref = x_nvfp4_ref.scale_t.view(dtype=torch.uint8)
        # Compute transpose amax using the same reference quantizer
        x_t_for_amax = (
            ref_quantizer._apply_rht(x.t().contiguous()) if with_rht else x.t().contiguous()
        )
        ref_amax_colwise_t = torch.max(torch.abs(x_t_for_amax)).to(torch.float32).view(1)
    else:
        qx_t_ref = None
        sx_t_ref = None
        ref_amax_colwise_t = None

    torch.testing.assert_close(amax_rowwise, ref_amax_rowwise, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)
    # Compare only the valid portion of scale tensors (reference may not have padding)
    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(amax_colwise, ref_amax_colwise_t, atol=0.0, rtol=0.0)

        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)

        # Compare only the valid portion of transpose scale tensors
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)


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
        # Real shapes,
        (8192, 5120),
        (8192, 10240),
        (8192, 2560),
        (8192, 11328),
        (8192, 512),
        (8192, 3584),
        (5120, 8192),
        (10240, 8192),
        (2560, 8192),
        (11328, 8192),
        (512, 8192),
        (3584, 8192),
        (4096, 16384),
        (14336, 16384),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "return_transpose", [True, False], ids=["quantize_transpose", "skip_transpose"]
)
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize(
    "with_random_sign_mask", [True, False], ids=["with_random_sign_mask", "no_random_sign_mask"]
)
def test_rht_with_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
    with_random_sign_mask: bool,
) -> None:
    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        contiguous=True,
        return_transpose=return_transpose,
        use_cpp_allocator=use_cpp_allocator,
        with_random_sign_mask=with_random_sign_mask,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (32, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "return_transpose", [True, False], ids=["quantize_transpose", "skip_transpose"]
)
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize(
    "with_random_sign_mask", [True, False], ids=["with_random_sign_mask", "no_random_sign_mask"]
)
def test_nvfp4_quantization_noncontiguous_inputs(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
    with_random_sign_mask: bool,
):
    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        contiguous=False,
        return_transpose=return_transpose,
        use_cpp_allocator=use_cpp_allocator,
        with_random_sign_mask=with_random_sign_mask,
    )
