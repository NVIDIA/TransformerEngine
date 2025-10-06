# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import transformer_engine as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.utils import get_device_compute_capability
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
)
from references.blockwise_quantizer_reference import CuBLASScaleMunger
from references.blockwise_fp8_gemm_reference import CuBLASRefBlockwiseGemm


def fp8_blockwise_gemm_supported() -> bool:
    supported, _ = FP8GlobalStateManager.is_fp8_block_scaling_available()
    emulated = get_device_compute_capability() >= (10, 0)
    return supported and not emulated


def cublas_gemm_fp8_blockwise_case(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    noise_type,
    x_magnitude,
    w_magnitude,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
    *,
    x_columnwise: bool = False,
    w_columnwise: bool = False,
    use_bias: bool = False,
    use_gelu: bool = False,
    use_grad: bool = False,
    atol: float = 0.0,
    rtol: float = 0.0
):
    if x_dtype == torch.float8_e5m2 and w_dtype == torch.float8_e5m2:
        pytest.skip("FP8 GEMM doesn't support both a and b types being torch.float8_e5m2")
    if not (is_x_1d_scaled or is_w_1d_scaled):
        pytest.skip("FP8 GEMM doesn't support 2dimensional qtile by 2dimensional qtile")
    if not fp8_blockwise_gemm_supported():
        pytest.skip("CUDA version does not support blockwise FP8 gemm.")
    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x_shape = (K, M) if x_columnwise else (M, K)
    w_shape = (K, N) if w_columnwise else (N, K)
    # generate random input and weight
    if noise_type == "uniform":
        x = torch.rand(x_shape, dtype=torch.float32, device=device) * x_magnitude * 2 - x_magnitude
        w = torch.rand(w_shape, dtype=torch.float32, device=device) * w_magnitude * 2 - w_magnitude
    elif noise_type == "normal":
        x = torch.randn(x_shape, dtype=torch.float32, device=device) * x_magnitude
        w = torch.randn(w_shape, dtype=torch.float32, device=device) * w_magnitude
    else:
        assert False

    # Setup out tensor if accumulate is True
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device) * x_magnitude
    else:
        out = None

    assert not (use_bias and use_grad), "Bias grad not supported by GEMM"
    # Set quantize_op and quantization parameters
    x_quant_tile_shape = (1, 128) if is_x_1d_scaled else (128, 128)
    w_quant_tile_shape = (1, 128) if is_w_1d_scaled else (128, 128)
    x_block_scaling_dim = 1 if is_x_1d_scaled else 2
    w_block_scaling_dim = 1 if is_w_1d_scaled else 2
    x_te_dtype = TE_DType[x_dtype]
    w_te_dtype = TE_DType[w_dtype]
    x_quantizer = Float8BlockQuantizer(
        fp8_dtype=x_te_dtype,
        rowwise=True,
        columnwise=True,
        amax_epsilon=0.0,
        force_pow_2_scales=True,
        block_scaling_dim=x_block_scaling_dim,
    )
    w_quantizer = Float8BlockQuantizer(
        fp8_dtype=w_te_dtype,
        rowwise=True,
        columnwise=True,
        amax_epsilon=0.0,
        force_pow_2_scales=True,
        block_scaling_dim=w_block_scaling_dim,
    )

    # Quantize x and w
    qx = x_quantizer.make_empty(x_shape, dtype=x_dtype, device=device, requires_grad=False)
    qx = x_quantizer.update_quantized(x, qx)
    qw = w_quantizer.make_empty(w_shape, dtype=w_dtype, device=device, requires_grad=False)
    qw = w_quantizer.update_quantized(w, qw)

    if not use_bias:
        bias = None
    else:
        bias = torch.randn((1, N), dtype=torch.bfloat16, device=device)

    # Reference GEMM
    ref_gemm = CuBLASRefBlockwiseGemm()
    scale_decoder = CuBLASScaleMunger()
    qx_data = (
        qx._columnwise_data.view(dtype=x_dtype)
        if x_columnwise
        else qx._rowwise_data.view(dtype=x_dtype)
    )
    qw_data = (
        qw._columnwise_data.view(dtype=w_dtype)
        if w_columnwise
        else qw._rowwise_data.view(dtype=w_dtype)
    )
    ref_scales_x = qx._columnwise_scale_inv if x_columnwise else qx._rowwise_scale_inv
    ref_scales_w = qw._columnwise_scale_inv if w_columnwise else qw._rowwise_scale_inv
    y_ref = ref_gemm.qgemm(
        qx=qx_data,
        qw=qw_data,
        out_dtype=out_dtype,
        demunged_sx=CuBLASScaleMunger.demunge_scale_shape_from_backend(
            qtensor_shape=(M, K), scales=ref_scales_x, tile_shape=x_quant_tile_shape
        ),
        demunged_sw=CuBLASScaleMunger.demunge_scale_shape_from_backend(
            qtensor_shape=(N, K), scales=ref_scales_w, tile_shape=w_quant_tile_shape
        ),
        quant_tile_shape_x=x_quant_tile_shape,
        quant_tile_shape_w=w_quant_tile_shape,
        bias=bias,
        out=out.clone() if accumulate else None,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    # Allocate cuBLAS workspace
    workspace_size = 0
    workspace = torch.empty(0, dtype=torch.uint8, device=device)

    transa = True if not w_columnwise else False
    transb = False if not x_columnwise else True
    out_quantizer = None
    assert not (use_gelu and use_bias), "Bias and GELU not supported by GEMM"
    aux_tensor = torch.randn((M, N), dtype=out_dtype, device=device) if use_gelu else None
    aux_tensor_ref = aux_tensor.clone() if use_gelu else None

    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]
    # cuBLAS GEMM
    # return type is out, bias_grad, gelu_input, extra_output
    # We are just capturing out.
    y = tex.generic_gemm(
        qw,
        transa,
        qx,
        transb,
        out.clone() if accumulate else None,
        out_quantizer,
        TE_DType[out_dtype],
        bias,
        bias_dtype,
        use_gelu,
        aux_tensor,
        use_grad,
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )[0]

    # just in case of accumulation, make sure y_ref and y are not the same tensor
    assert y_ref is not y, "y_ref and y should not be the same tensor"
    # Reset nans to zeros because torch.assert_close does not assume nans to be equal
    assert not torch.isnan(y_ref.float()).all(), "All elements are nan"
    y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
    y = torch.where(y.isnan(), torch.zeros_like(y), y)

    if use_gelu:
        # Check
        if use_grad:
            # With use_grad, GEMM should use aux tensor to calculate
            # gradient
            gelu_ref = tex.dgelu(y_ref, aux_tensor_ref, None)
            # TODO: How do we decide whether this is acceptably close?
            # Could also try to put the activation inside the reference
            # before the output cast to see different tolerances.
            torch.testing.assert_close(y, gelu_ref, atol=1e-3, rtol=1e-2)
        else:
            # aux tensor is pre-gelu aux output. Verify against y_ref.
            torch.testing.assert_close(aux_tensor, y_ref, atol=atol, rtol=rtol)
            act = torch.nn.GELU()
            gelu_ref = act(y_ref)
            # gelu_ref = tex.gelu(y_ref, None)
            torch.testing.assert_close(y, gelu_ref, atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


def cublas_gemm_test_constraint_enforced(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
    *,
    x_columnwise: bool = False,
    w_columnwise: bool = False,
    use_bias: bool = False,
    use_gelu: bool = False,
    use_grad: bool = False,
    expected_err_msg="CUBLAS_STATUS_NOT_SUPPORTED",
    expected_err_cls=RuntimeError
):
    if not fp8_blockwise_gemm_supported():
        pytest.skip("CUDA version does not support blockwise FP8 gemm.")
    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x_shape = (K, M) if x_columnwise else (M, K)
    w_shape = (K, N) if w_columnwise else (N, K)
    # generate random input and weight
    x = torch.rand(x_shape, dtype=torch.float32, device=device) * 2.0 - 1.0
    w = torch.rand(w_shape, dtype=torch.float32, device=device) * 2.0 - 1.0

    # Setup out tensor if accumulate is True
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = None

    # Set quantize_op and quantization parameters
    x_quant_tile_shape = (1, 128) if is_x_1d_scaled else (128, 128)
    w_quant_tile_shape = (1, 128) if is_w_1d_scaled else (128, 128)
    x_block_scaling_dim = 1 if is_x_1d_scaled else 2
    w_block_scaling_dim = 1 if is_w_1d_scaled else 2
    x_te_dtype = TE_DType[x_dtype]
    w_te_dtype = TE_DType[w_dtype]
    x_quantizer = Float8BlockQuantizer(
        fp8_dtype=x_te_dtype,
        rowwise=True,
        columnwise=True,
        amax_epsilon=0.0,
        force_pow_2_scales=True,
        block_scaling_dim=x_block_scaling_dim,
    )
    w_quantizer = Float8BlockQuantizer(
        fp8_dtype=w_te_dtype,
        rowwise=True,
        columnwise=True,
        amax_epsilon=0.0,
        force_pow_2_scales=True,
        block_scaling_dim=w_block_scaling_dim,
    )

    # Quantize x and w
    qx = x_quantizer.make_empty(x_shape, dtype=x_dtype, device=device, requires_grad=False)
    qx = x_quantizer.update_quantized(x, qx)
    qw = w_quantizer.make_empty(w_shape, dtype=w_dtype, device=device, requires_grad=False)
    qw = w_quantizer.update_quantized(w, qw)

    if not use_bias:
        bias = None
    else:
        bias = torch.randn((1, N), dtype=torch.bfloat16, device=device)

    # Allocate cuBLAS workspace
    workspace_size = 0
    workspace = torch.empty(0, dtype=torch.uint8, device=device)

    transa = True if not w_columnwise else False
    transb = False if not x_columnwise else True
    out_quantizer = None
    grad = use_grad
    gelu_in = None if not use_gelu else torch.randn((M, N), dtype=out_dtype, device=device)

    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]
    # cuBLAS GEMM
    # return type is out, bias_grad, gelu_input, extra_output
    # We are just capturing out.
    with pytest.raises(expected_err_cls, match=expected_err_msg):
        y = tex.generic_gemm(
            qw,
            transa,
            qx,
            transb,
            out.clone() if accumulate else None,
            out_quantizer,
            TE_DType[out_dtype],
            bias,
            bias_dtype,
            use_gelu,
            gelu_in,
            grad,
            workspace,
            workspace.shape[0],
            accumulate,
            use_split_accumulator,
        )


@pytest.mark.parametrize(
    "M, K, N",
    [
        # k = 128
        (128, 128, 128),
        (256, 128, 256),
        # non 128x128 divisible input shapes
        (320, 128, 336),
        (320, 64, 336),
        # k > 128
        (256, 256, 256),
        (320, 256, 336),
        (1024, 4096, 1024),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("noise_type", ["normal"], ids=str)
@pytest.mark.parametrize("x_magnitude", [1], ids=str)
@pytest.mark.parametrize("w_magnitude", [1], ids=str)
@pytest.mark.parametrize("accumulate", [False], ids=["no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
def test_cublas_gemm_fp8_blockwise_shape_varying(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    noise_type,
    x_magnitude,
    w_magnitude,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
):
    cublas_gemm_fp8_blockwise_case(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        noise_type,
        x_magnitude,
        w_magnitude,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        (256, 128, 256),
        (320, 256, 336),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("noise_type", ["normal", "uniform"], ids=str)
@pytest.mark.parametrize("x_magnitude", [1e-28, 1, 1e3], ids=str)
@pytest.mark.parametrize("w_magnitude", [1], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
def test_cublas_gemm_fp8_blockwise_accumulate_magnitude_varying(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    noise_type,
    x_magnitude,
    w_magnitude,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
):
    cublas_gemm_fp8_blockwise_case(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        noise_type,
        x_magnitude,
        w_magnitude,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        # k = 128
        (256, 128, 256),
        # non 128x128 divisible input shapes
        (320, 64, 336),
        # k > 128
        (256, 256, 256),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("noise_type", ["normal"], ids=str)
@pytest.mark.parametrize("x_magnitude", [1e-3], ids=str)
@pytest.mark.parametrize("w_magnitude", [1], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
def test_cublas_gemm_fp8_blockwise_bias(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    noise_type,
    x_magnitude,
    w_magnitude,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
):
    cublas_gemm_fp8_blockwise_case(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        noise_type,
        x_magnitude,
        w_magnitude,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
        use_bias=True,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        # k = 128
        (256, 128, 256),
        # non 128x128 divisible input shapes
        (16, 128, 128),
        (320, 64, 336),
        # k > 128
        (4096, 128, 4096),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("noise_type", ["normal"], ids=str)
@pytest.mark.parametrize("x_magnitude", [1], ids=str)
@pytest.mark.parametrize("w_magnitude", [1], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
@pytest.mark.parametrize(
    "is_x_columnwise, is_w_columnwise",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["colxrow", "colxcol", "rowxcol"],
)
def test_cublas_gemm_fp8_blockwise_columnwise(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    noise_type,
    x_magnitude,
    w_magnitude,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
    is_x_columnwise,
    is_w_columnwise,
):
    cublas_gemm_fp8_blockwise_case(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        noise_type,
        x_magnitude,
        w_magnitude,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
        x_columnwise=is_x_columnwise,
        w_columnwise=is_w_columnwise,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        # k = 128
        (256, 128, 256),
        # non 128x128 divisible input shapes
        (320, 64, 336),
        # k > 128
        (256, 256, 256),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("noise_type", ["normal"], ids=str)
@pytest.mark.parametrize("x_magnitude", [1], ids=str)
@pytest.mark.parametrize("w_magnitude", [1], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
@pytest.mark.parametrize(
    "use_grad",
    [
        True,
    ],
    ids=["grad"],
)
def test_cublas_gemm_fp8_gelu(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    noise_type,
    x_magnitude,
    w_magnitude,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
    use_grad,
):
    # NOTE: cuBLAS doesn't complain with not use_grad, but the tests don't succeed
    # so the epilogue is disabled on the transformer engine side.
    if not use_grad and not (is_x_1d_scaled and not is_w_1d_scaled):
        pytest.skip(
            "CUBLASLT_EPILOGUE_GELU_AUX epilogue is only supported for 1Dx2D (cuBLAS 2Dx1D)."
        )
    cublas_gemm_fp8_blockwise_case(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        noise_type,
        x_magnitude,
        w_magnitude,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
        use_gelu=True,
        use_grad=use_grad,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        # k = 128
        (256, 128, 256),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [False], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
def test_split_accumulator_enforced(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
) -> None:
    cublas_gemm_test_constraint_enforced(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        # k = 128
        (256, 128, 256),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
def test_bgrad_not_supported(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
) -> None:
    # NOTE: BGRAD epilogue is not supported for fp8.
    cublas_gemm_test_constraint_enforced(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
        use_grad=True,
        use_bias=True,
        expected_err_msg="Epilogue requested outside of the available",
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        # k = 128
        (256, 128, 256),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("use_grad", [True, False], ids=["grad", "no_grad"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
def test_gelu_unsupported_cases_error(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    accumulate,
    use_bias,
    use_grad,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
) -> None:
    if use_grad and not use_bias and out_dtype == torch.bfloat16:
        pytest.skip("DGELU epilogue is supported for bfloat16.")
    elif use_grad and not use_bias:
        expected_err = "an unsupported value or parameter was passed"
    else:
        expected_err = "Epilogue requested outside of the available"
    cublas_gemm_test_constraint_enforced(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
        use_grad=use_grad,
        use_bias=use_bias,
        use_gelu=True,
        expected_err_msg=expected_err,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        (256, 128, 256),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e5m2], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (True, True),
        (False, True),
    ],
    ids=["1Dx2D", "1Dx1D", "2Dx1D"],
)
def test_illegal_dtype_enforced(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
) -> None:
    # e5m2 by e5m2 not supported.
    cublas_gemm_test_constraint_enforced(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        (256, 128, 256),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (False, False),
    ],
    ids=["2Dx2D"],
)
def test_illegal_2D_by_2D_enforced(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
) -> None:
    # 2D block quantization by 2D block quantization is not supported.
    expected_err_msg = "Only 1D by 1D, 1D by 2D, and 2D by 1D block scaling supported"
    cublas_gemm_test_constraint_enforced(
        x_dtype,
        w_dtype,
        out_dtype,
        M,
        K,
        N,
        accumulate,
        use_split_accumulator,
        is_x_1d_scaled,
        is_w_1d_scaled,
        expected_err_msg=expected_err_msg,
    )


@pytest.mark.parametrize(
    "M, K, N, legalX1d, legalX2d",
    [
        # M dim unconstrained when X is 2D.
        (255, 128, 256, False, True),
        # K must be multiple of 16
        (256, 120, 256, False, False),
        # N must be a multiple of 8
        (256, 128, 252, False, False),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("accumulate", [False], ids=["no_accumulate"])
@pytest.mark.parametrize("use_split_accumulator", [True], ids=["split_acc"])
@pytest.mark.parametrize(
    "is_x_1d_scaled, is_w_1d_scaled",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
    ids=["1Dx2D", "2Dx1D", "1Dx1D"],
)
def test_unaligned_shapes(
    x_dtype,
    w_dtype,
    out_dtype,
    M,
    K,
    N,
    legalX1d,
    legalX2d,
    accumulate,
    use_split_accumulator,
    is_x_1d_scaled,
    is_w_1d_scaled,
) -> None:
    legal = legalX1d if is_x_1d_scaled else legalX2d
    if not legal:
        cublas_gemm_test_constraint_enforced(
            x_dtype,
            w_dtype,
            out_dtype,
            M,
            K,
            N,
            accumulate,
            use_split_accumulator,
            is_x_1d_scaled,
            is_w_1d_scaled,
            expected_err_msg="dimension requirement",
        )
    else:
        cublas_gemm_fp8_blockwise_case(
            x_dtype,
            w_dtype,
            out_dtype,
            M,
            K,
            N,
            "uniform",  # noise type
            1.0,  # x_magnitude
            1.0,  # w_magnitude
            accumulate,
            use_split_accumulator,
            is_x_1d_scaled,
            is_w_1d_scaled,
        )
