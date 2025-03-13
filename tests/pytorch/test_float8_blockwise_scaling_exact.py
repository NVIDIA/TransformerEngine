# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Tuple
import math
import pytest
import torch
import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import get_device_compute_capability
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
)
from tests.pytorch.references.blockwise_quantizer_reference import (
    BlockwiseQuantizerReference,
    QuantizeResult,
)

# TODO replace with call to fp8.py when recipe added.
recipe_available = get_device_compute_capability() >= (9, 0) and float(torch.version.cuda) >= 12.8
reason_for_no_recipe = "Quantize kernels require TMA and are only relevant with GEMMS."


def initialize_for_many_scales(
    x_shape_2d: Tuple[int, int], tile_shape: Tuple[int, int], *, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """
    Put separate distributions into each quantization tile
    to avoid many tiles having similar scale values and
    causing false passes.
    """
    tile_grid_shape = (
        math.ceil(x_shape_2d[0] / tile_shape[0]),
        math.ceil(x_shape_2d[1] / tile_shape[1]),
    )
    # Arbitrary size
    max_val = 8192.0
    # Make a uniform distribution of [-max_val, max_val]
    tile_extrema = torch.rand(*tile_grid_shape, dtype=dtype) * max_val * 2 - max_val
    result = torch.empty(x_shape_2d, dtype=dtype, device=device)
    tile_elements = tile_shape[0] * tile_shape[1]
    for i in range(tile_grid_shape[0]):
        for j in range(tile_grid_shape[1]):
            target = tile_extrema[i, j].item()
            step = target / (tile_elements)
            if target == 0:
                tile = torch.zeros(tile_shape, dtype=dtype, device=device)
            else:
                tile = torch.arange(0.0, target, step=step, dtype=dtype, device=device)
            tile = tile.reshape(*tile_shape)
            min_dst_vals = (i * tile_shape[0], j * tile_shape[1])
            max_dst_vals = (
                min((i + 1) * tile_shape[0], x_shape_2d[0]),
                min((j + 1) * tile_shape[1], x_shape_2d[1]),
            )
            max_src_vals = (
                max_dst_vals[0] - min_dst_vals[0],
                max_dst_vals[1] - min_dst_vals[1],
            )
            result[min_dst_vals[0] : max_dst_vals[0], min_dst_vals[1] : max_dst_vals[1]] = tile[
                : max_src_vals[0], : max_src_vals[1]
            ]
    return result


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
        (303, 300),
        (305, 256),
        # Some larger tiles.
        (2000, 2000),
        (2048, 2000),
        (2000, 1024),
        (2048, 1024),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("eps", [0, 1e-12], ids=["eps_0", "eps_1e-12"])
@pytest.mark.parametrize(
    "return_transpose", [True, False], ids=["quantize_transpose", "quantize_only"]
)
@pytest.mark.parametrize("pow_2_scales", [True, False], ids=["pow2scales", "f32scales"])
@pytest.mark.parametrize("tile_size", [(1, 128), (128, 128)], ids=["1DTile", "2DTile"])
def test_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quant_dtype: torch.dtype,
    eps: float,
    return_transpose: bool,
    pow_2_scales: bool,
    tile_size: Tuple[int, int],
) -> None:
    te_dtype = TE_DType[quant_dtype]
    if tile_size == (1, 128):
        block_scaling_dim = 1
    elif tile_size == (128, 128):
        block_scaling_dim = 2
    else:
        raise ValueError("Non support tile size")
    # This test runs a comparison of the ref class versus the class using
    # CUDA kernels to quantize. They should quantize identically for pixels
    # that are not DC values in the scale factor shape.
    ref_quantizer = BlockwiseQuantizerReference()
    sut_quantizer = Float8BlockQuantizer(
        fp8_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        amax_epsilon=eps,
        force_pow_2_scales=pow_2_scales,
        block_scaling_dim=block_scaling_dim,
    )

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input
    x = initialize_for_many_scales((M, N), tile_size, dtype=x_dtype, device=device)

    x_fp8_sut = sut_quantizer.make_empty((M, N), dtype=x_dtype, device=device, requires_grad=False)
    x_fp8_sut = sut_quantizer.update_quantized(x, x_fp8_sut)

    assert x_fp8_sut._rowwise_data is not None
    qx: torch.Tensor = x_fp8_sut._rowwise_data.view(dtype=quant_dtype)
    assert x_fp8_sut._rowwise_scale_inv is not None
    sx: torch.Tensor = x_fp8_sut._rowwise_scale_inv
    qx_t = x_fp8_sut._columnwise_data
    sx_t = x_fp8_sut._columnwise_scale_inv

    qresult_ref = ref_quantizer.quantize(
        x,
        quant_dtype=quant_dtype,
        return_transpose=return_transpose,
        eps=eps,
        pow_2_scales=pow_2_scales,
        quant_tile_shape=tile_size,
    )
    qx_ref, sx_ref, qx_t_ref, sx_t_ref = (
        qresult_ref.data,
        qresult_ref.scale,
        qresult_ref.data_t,
        qresult_ref.scale_t,
    )

    # Check
    torch.testing.assert_close(qx.float(), qx_ref.float(), atol=0.0, rtol=0.0)
    # Zero out values that are don't care values
    # Scale format has padding.
    scale_mask = torch.ones(
        (math.ceil(M / tile_size[0]), math.ceil(N / tile_size[1])), device=sx.device
    )
    scale_mask = ref_quantizer.scale_munger.munge_scale_shapes_for_backend(
        QuantizeResult(qx, scale_mask, None, None), tile_size
    ).scale
    sx = sx * scale_mask
    torch.testing.assert_close(sx, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        assert qx_t is not None
        qx_t = qx_t.view(dtype=quant_dtype)
        assert qx_t_ref is not None
        assert sx_t is not None
        assert sx_t_ref is not None
        scale_mask = torch.ones(
            (math.ceil(N / tile_size[0]), math.ceil(M / tile_size[1])),
            device=sx_t.device,
        )
        scale_mask = ref_quantizer.scale_munger.munge_scale_shapes_for_backend(
            QuantizeResult(qx_t, scale_mask, None, None), tile_size
        ).scale
        sx_t = sx_t * scale_mask
        torch.testing.assert_close(qx_t.float(), qx_t_ref.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(sx_t, sx_t_ref, atol=0.0, rtol=0.0)
    else:
        # should be None
        assert qx_t is None and qx_t_ref is None
        assert sx_t is None and sx_t_ref is None


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # full tile cases
        (128, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("eps", [0, math.pow(2, -125)], ids=["eps_0", "eps_small"])
@pytest.mark.parametrize("pow_2_scales", [True, False], ids=["pow2scales", "f32scales"])
@pytest.mark.parametrize("tile_size", [(128, 128)])
@pytest.mark.parametrize("extrema_high", [False, True], ids=["zeros", "maxes"])
def test_quantization_block_tiling_extrema_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quant_dtype: torch.dtype,
    eps: float,
    pow_2_scales: bool,
    tile_size: Tuple[int, int],
    extrema_high: bool,
) -> None:
    # This test runs a single tile through a quantizer as a way to test
    # branch coverage of scale computation.
    te_dtype = TE_DType[quant_dtype]
    if tile_size == (1, 128):
        block_scaling_dim = 1
    elif tile_size == (128, 128):
        block_scaling_dim = 2
    else:
        raise ValueError("Non support tile size")
    ref_quantizer = BlockwiseQuantizerReference()
    sut_quantizer = Float8BlockQuantizer(
        fp8_dtype=te_dtype,
        rowwise=True,
        columnwise=False,
        amax_epsilon=eps,
        force_pow_2_scales=pow_2_scales,
        block_scaling_dim=block_scaling_dim,
    )
    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return_transpose = False
    # Input
    if extrema_high:
        x = torch.full((M, N), torch.finfo(x_dtype).max, dtype=x_dtype, device=device)
    else:
        x = torch.zeros((M, N), dtype=x_dtype, device=device)

    # Run cast and transpose kernel
    # Internal call ops.quantize_tensorwise
    x_fp8_sut = sut_quantizer.make_empty((M, N), dtype=x_dtype, device=device, requires_grad=False)
    x_fp8_sut = sut_quantizer.update_quantized(x, x_fp8_sut)
    qx = x_fp8_sut._rowwise_data.view(dtype=quant_dtype)
    sx = x_fp8_sut._rowwise_scale_inv

    qresult_ref = ref_quantizer.quantize(
        x,
        quant_dtype=quant_dtype,
        return_transpose=return_transpose,
        eps=eps,
        pow_2_scales=pow_2_scales,
        quant_tile_shape=tile_size,
    )
    qx_ref, sx_ref = (
        qresult_ref.data,
        qresult_ref.scale,
    )

    # Check
    torch.testing.assert_close(qx.float(), qx_ref.float(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(sx.flatten()[0], sx_ref.flatten()[0], atol=0.0, rtol=0.0)

    if extrema_high:
        expected_value = torch.finfo(quant_dtype).max / torch.finfo(x_dtype).max
        if pow_2_scales:
            expected_value = math.floor(math.log2(expected_value))
            expected_value = math.pow(2.0, expected_value)
        expected_value = 1 / expected_value
    elif not extrema_high and eps == 0:
        expected_value = 1.0
    else:
        assert not extrema_high
        # eps is small enough to trigger inf in quant_dtype_max / eps
        if pow_2_scales:
            expected_value = math.pow(2.0, -127)
        else:
            expected_value = 1 / torch.finfo(x_dtype).max
    torch.testing.assert_close(
        sx.flatten()[0],
        torch.tensor(expected_value, device=sx.device),
        atol=0.0,
        rtol=0.0,
    )
