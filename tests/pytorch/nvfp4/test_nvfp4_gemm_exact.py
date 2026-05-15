# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def check_nvfp4_gemm_versus_reference(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    accumulate: bool,
    *,
    x_columnwise: bool = False,
    w_columnwise: bool = False,
    row_scaled_nvfp4: bool = False,
):
    te_dtype = tex.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input tensors
    x_shape = (K, M) if x_columnwise else (M, K)
    w_shape = (K, N) if w_columnwise else (N, K)
    x = torch.randn(x_shape, dtype=x_dtype, device=device)
    w = torch.randn(w_shape, dtype=w_dtype, device=device)

    # Setup out tensor if accumulate is True
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = None

    # Native TE NVFP4 quantization
    x_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=not row_scaled_nvfp4,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=row_scaled_nvfp4,
    )
    w_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    # Quantize x and w
    x_nvfp4_native = x_quantizer.make_empty(
        x_shape, dtype=x_dtype, device=device, requires_grad=False
    )
    x_nvfp4_native = x_quantizer.update_quantized(x, x_nvfp4_native)
    w_nvfp4_native = w_quantizer.make_empty(
        w_shape, dtype=w_dtype, device=device, requires_grad=False
    )
    w_nvfp4_native = w_quantizer.update_quantized(w, w_nvfp4_native)

    # Extract quantized data from native NVFP4Tensors
    qx_data = (
        x_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
        if x_columnwise
        else x_nvfp4_native._rowwise_data.view(dtype=torch.uint8)
    )
    qw_data = (
        w_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
        if w_columnwise
        else w_nvfp4_native._rowwise_data.view(dtype=torch.uint8)
    )
    sx_native = (
        x_nvfp4_native._columnwise_scale_inv if x_columnwise else x_nvfp4_native._rowwise_scale_inv
    )
    sw_native = (
        w_nvfp4_native._columnwise_scale_inv if w_columnwise else w_nvfp4_native._rowwise_scale_inv
    )

    # Trim quantized data to match the actual tensor dimensions (remove padding)
    qx_data = qx_data[:M, :]
    qw_data = qw_data[:N, :]

    # NVFP4 uses 16-element blocks, trim scales to remove padding
    block_length = 16  # NVFP4 uses 16-element blocks
    expected_sx_cols = expected_sw_cols = K // block_length
    # Trim the scales to remove padding
    sx_trimmed = sx_native[:M, :expected_sx_cols]
    sw_trimmed = sw_native[:N, :expected_sw_cols]

    # Native scales are stored as uint8 but need to be interpreted as float8_e4m3fn
    # for the reference GEMM to work correctly
    sx_trimmed = sx_trimmed.view(torch.float8_e4m3fn)
    sw_trimmed = sw_trimmed.view(torch.float8_e4m3fn)

    # Create reference quantizer for reference GEMM
    x_ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=not row_scaled_nvfp4,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        row_scaled_nvfp4=row_scaled_nvfp4,
    )
    w_ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=True,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
    )

    # Create reference quantized tensors needed by reference GEMM
    # Reference GEMM is only rowwise.
    if x_columnwise:
        x_nvfp4_ref = x_ref_quantizer.quantize(x.t().contiguous())
    else:
        x_nvfp4_ref = x_ref_quantizer.quantize(x)
    if w_columnwise:
        w_nvfp4_ref = w_ref_quantizer.quantize(w.t().contiguous())
    else:
        w_nvfp4_ref = w_ref_quantizer.quantize(w)

    # Reference GEMM using quantizer's qgemm method
    y_ref = x_ref_quantizer.qgemm(
        qx=qx_data,
        qw=qw_data,
        m_params=None,  # MMParams not used in reference
        out_dtype=out_dtype,
        sx=sx_trimmed,
        sw=sw_trimmed,
        bias=None,  # No bias for this test
        out=out.clone() if accumulate else None,
        accumulate=accumulate,
        gemm_type=None,  # GEMMType not used in reference
        qresult_x=x_nvfp4_ref,
        qresult_w=w_nvfp4_ref,
    )

    # Native TE GEMM using tex.generic_gemm (cuBLAS GEMM)
    # Allocate cuBLAS workspace
    workspace = torch.empty(4, dtype=torch.uint8, device=device)

    transa = True if not w_columnwise else False
    transb = False if not x_columnwise else True
    out_quantizer = None
    bias = None
    bias_dtype = TE_DType[torch.bfloat16]
    use_gelu = False
    gelu_input = None
    use_grad = False
    use_split_accumulator = False

    if x_columnwise:
        x_nvfp4_native.update_usage(rowwise_usage=False)
    if w_columnwise:
        w_nvfp4_native.update_usage(rowwise_usage=False)
    if row_scaled_nvfp4:
        layout = ("T" if transa else "N") + ("T" if transb else "N")
        y_native = general_gemm(
            w_nvfp4_native,
            x_nvfp4_native,
            out_dtype=out_dtype,
            accumulate=accumulate,
            layout=layout,
            out=out.clone() if accumulate else None,
        )[0]
    else:
        # Native cuBLAS GEMM
        # return type is out, bias_grad, gelu_input, extra_output
        # We are just capturing out.
        y_native = tex.generic_gemm(
            w_nvfp4_native,
            transa,
            x_nvfp4_native,
            transb,
            out.clone() if accumulate else None,
            out_quantizer,
            TE_DType[out_dtype],
            bias,
            bias_dtype,
            use_gelu,
            gelu_input,
            use_grad,
            workspace,
            workspace.shape[0],
            accumulate,
            use_split_accumulator,
        )[0]

    # just in case of accumulation, make sure y_ref and y_native are not the same tensor
    assert y_ref is not y_native, "y_ref and y_native should not be the same tensor"
    # Reset nans to zeros because torch.assert_close does not assume nans to be equal
    assert not torch.isnan(y_ref.float()).all(), "All elements are nan"
    y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
    y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)

    # Compare results with some tolerance
    torch.testing.assert_close(y_native, y_ref, atol=8e-3, rtol=8e-3)


def check_nvfp4_row_scaled_grouped_gemm_matches_per_gemm(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    m_splits: list[int],
    k: int,
    n: int,
    *,
    use_bias: bool,
    single_output: bool,
):
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"
    torch.manual_seed(23)
    torch.cuda.manual_seed(23)

    num_gemms = len(m_splits)

    x_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=True,
    )
    w_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    x_nvfp4 = []
    w_nvfp4 = []
    bias = []
    expected = []
    for m in m_splits:
        x = torch.randn((m, k), dtype=x_dtype, device=device)
        w = torch.randn((n, k), dtype=w_dtype, device=device)
        x_nvfp4.append(
            x_quantizer.update_quantized(
                x, x_quantizer.make_empty(x.shape, dtype=x_dtype, device=device)
            )
        )
        w_nvfp4.append(
            w_quantizer.update_quantized(
                w, w_quantizer.make_empty(w.shape, dtype=w_dtype, device=device)
            )
        )
        bias.append(torch.randn(n, dtype=torch.bfloat16, device=device) if use_bias else None)
        expected.append(
            general_gemm(
                w_nvfp4[-1],
                x_nvfp4[-1],
                out_dtype=out_dtype,
                layout="TN",
                bias=bias[-1],
            )[0]
        )

    if single_output:
        out = [torch.empty((sum(m_splits), n), dtype=out_dtype, device=device)]
    else:
        out = [torch.empty((m, n), dtype=out_dtype, device=device) for m in m_splits]

    grouped_out, _, _ = general_grouped_gemm(
        w_nvfp4,
        x_nvfp4,
        out,
        quantization_params=[None] * num_gemms,
        out_dtype=out_dtype,
        layout="TN",
        m_splits=m_splits,
        bias=bias,
        use_bias=use_bias,
        single_output=single_output,
    )

    if single_output:
        grouped_slices = torch.split(grouped_out, m_splits, dim=0)
    else:
        grouped_slices = grouped_out
    for grouped, ref in zip(grouped_slices, expected):
        torch.testing.assert_close(grouped, ref, atol=0.0, rtol=0.0)


def check_nvfp4_row_scaled_gemm_matches_emulated(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
):
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"
    torch.manual_seed(37)
    torch.cuda.manual_seed(37)

    x = torch.randn((M, K), dtype=x_dtype, device=device)
    w = torch.randn((N, K), dtype=w_dtype, device=device)

    x_row_scaled_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=True,
    )
    x_tensorwise_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    w_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    x_row_scaled = x_row_scaled_quantizer.update_quantized(
        x, x_row_scaled_quantizer.make_empty(x.shape, dtype=x_dtype, device=device)
    )
    w_nvfp4 = w_quantizer.update_quantized(
        w, w_quantizer.make_empty(w.shape, dtype=w_dtype, device=device)
    )
    y_row_scaled = general_gemm(w_nvfp4, x_row_scaled, out_dtype=out_dtype, layout="TN")[0]

    emulated_rows = []
    for i in range(M):
        x_padded = torch.zeros((16, K), dtype=x_dtype, device=device)
        x_padded[0].copy_(x[i])
        x_tensorwise = x_tensorwise_quantizer.update_quantized(
            x_padded,
            x_tensorwise_quantizer.make_empty(x_padded.shape, dtype=x_dtype, device=device),
        )
        emulated_rows.append(
            general_gemm(w_nvfp4, x_tensorwise, out_dtype=out_dtype, layout="TN")[0][:1]
        )

    y_emulated = torch.cat(emulated_rows, dim=0)
    if out_dtype == torch.bfloat16:
        torch.testing.assert_close(y_row_scaled, y_emulated, atol=0.0, rtol=7.8e-3)
    else:
        torch.testing.assert_close(y_row_scaled, y_emulated, atol=3.0517578125e-5, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, K, N",
    [
        (128, 128, 128),
        (256, 128, 256),
        (256, 256, 256),
        (256, 1024, 256),
        (1024, 1024, 1024),
        (4096, 512, 3072),
        (112, 128, 96),
        (304, 640, 304),
        (1008, 3072, 992),
        (256, 64, 256),
        (128, 128, 112),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize(
    "is_x_columnwise, is_w_columnwise",
    [
        (False, False),  # TN
        (True, False),  # NN
        (True, True),  # NT
    ],
    ids=["rowxrow", "colxrow", "colxcol"],
)
@pytest.mark.parametrize("row_scaled_nvfp4", [False, True], ids=["nvfp4", "nvfp4_row_scaled"])
def test_nvfp4_gemm_versus_reference(
    M: int,
    K: int,
    N: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    accumulate: bool,
    is_x_columnwise: bool,
    is_w_columnwise: bool,
    row_scaled_nvfp4: bool,
):
    if row_scaled_nvfp4:
        if accumulate:
            pytest.skip("Row-scaled NVFP4 GEMM output rescale does not support accumulation")
        if is_x_columnwise:
            pytest.skip("Row-scaled NVFP4 GEMM output rescale requires rowwise RHS usage")

    check_nvfp4_gemm_versus_reference(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        M=M,
        K=K,
        N=N,
        accumulate=accumulate,
        x_columnwise=is_x_columnwise,
        w_columnwise=is_w_columnwise,
        row_scaled_nvfp4=row_scaled_nvfp4,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "m_splits, k, n",
    [
        ([32, 48, 48], 128, 128),
        ([64, 80, 112], 128, 256),
        ([64, 80, 112], 256, 256),
        ([64, 80, 112], 1024, 256),
        ([256, 256, 512], 1024, 1024),
        ([1024, 1536, 1536], 512, 3072),
        ([16, 32, 64], 128, 96),
        ([80, 96, 128], 640, 304),
        ([320, 336, 352], 3072, 992),
        ([64, 80, 112], 64, 256),
        ([32, 48, 48], 128, 112),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("single_output", [False, True], ids=["list_output", "single_output"])
def test_nvfp4_row_scaled_grouped_gemm_matches_per_gemm(
    m_splits: list[int],
    k: int,
    n: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_bias: bool,
    single_output: bool,
):
    check_nvfp4_row_scaled_grouped_gemm_matches_per_gemm(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        m_splits=m_splits,
        k=k,
        n=n,
        use_bias=use_bias,
        single_output=single_output,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, K, N",
    [
        (128, 128, 128),
        (256, 128, 256),
        (256, 256, 256),
        (256, 1024, 256),
        (1024, 1024, 1024),
        (4096, 512, 3072),
        (112, 128, 96),
        (304, 640, 304),
        (1008, 3072, 992),
        (256, 64, 256),
        (128, 128, 112),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
def test_nvfp4_row_scaled_gemm_matches_emulated(
    M: int,
    K: int,
    N: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
):
    check_nvfp4_row_scaled_gemm_matches_emulated(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        M=M,
        K=K,
        N=N,
    )
