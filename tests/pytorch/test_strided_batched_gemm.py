# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for experimental strided batched GEMM."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    Float8Quantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
    get_device_compute_capability,
    is_bf16_available,
)
from transformer_engine.pytorch.cpp_extensions import general_gemm, strided_batched_gemm
from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import (
    Float8BlockwiseQTensorStorage,
)
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage
import transformer_engine_torch as tex

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

_UNSUPPORTED_INPUT_ERROR = (
    "strided_batched_gemm supports only high-precision or MXFP8 A and B tensor pairs"
)


def _skip_if_unavailable(recipe):
    if not is_bf16_available():
        pytest.skip("bfloat16 is not available.")
    if recipe == "fp8_block":
        if not te.is_fp8_block_scaling_available():
            pytest.skip("FP8 block scaling is not available.")
        if get_device_compute_capability() >= (10, 0):
            pytest.skip("FP8 block scaling GEMM is emulated on Blackwell; test on Hopper.")
    if recipe == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if recipe == "mxfp8" and tex.get_cublasLt_version() < 120800:
        pytest.skip("MXFP8 GEMM requires cuBLASLt 12.8+.")
    if recipe == "nvfp4" and not nvfp4_available:
        pytest.skip(reason_for_no_nvfp4)


def _group_quantize(tensors, quantizer):
    first_dims = torch.tensor([x.shape[0] for x in tensors], dtype=torch.int64, device="cuda")
    grouped = tex.group_quantize(torch.cat(tensors, dim=0), quantizer, len(tensors), first_dims)
    assert grouped._with_gemm_swizzled_scales
    return grouped.split_into_quantized_tensors()


def _quantize_operands(recipe, tensors):
    if recipe == "fp8":
        quantized = []
        for i, tensor in enumerate(tensors):
            quantizer = Float8Quantizer(
                scale=torch.tensor([1.0 + 0.25 * i], dtype=torch.float32, device="cuda"),
                amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
            )
            quantized.append(quantizer(tensor))
        return quantized

    if recipe == "fp8_block":
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            force_pow_2_scales=True,
            amax_epsilon=0.0,
            block_scaling_dim=1,
        )
        return [quantizer(tensor) for tensor in tensors]

    if recipe == "mxfp8":
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
        quantizer.optimize_for_gemm = True
        return _group_quantize(tensors, quantizer)

    if recipe == "nvfp4":
        quantizer = NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=True,
            columnwise=False,
            with_rht=True,
            with_post_rht_amax=True,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )
        quantizer.optimize_for_gemm = True
        return _group_quantize(tensors, quantizer)

    raise ValueError(f"Unknown quantized recipe: {recipe}")


def _rowwise_data(tensor, recipe):
    if recipe == "fp8":
        return tensor._data
    return tensor._rowwise_data


def _packed_storage(recipe, data, tensors, dtype):
    quantizer = tensors[0]._quantizer
    if recipe == "fp8":
        scales = torch.cat([tensor._scale_inv.reshape(-1) for tensor in tensors])
        return Float8TensorStorage(
            data=data,
            fp8_scale_inv=scales,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            fake_dtype=dtype,
        )

    scales = torch.cat([tensor._rowwise_scale_inv.reshape(-1) for tensor in tensors])
    if recipe == "fp8_block":
        return Float8BlockwiseQTensorStorage(
            rowwise_data=data,
            rowwise_scale_inv=scales,
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            is_2D_scaled=False,
            fake_dtype=dtype,
        )
    if recipe == "mxfp8":
        return MXFP8TensorStorage(
            rowwise_data=data,
            rowwise_scale_inv=scales,
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            with_gemm_swizzled_scales=True,
            fake_dtype=dtype,
        )
    if recipe == "nvfp4":
        amax = torch.cat([tensor._amax_rowwise.reshape(-1) for tensor in tensors])
        return NVFP4TensorStorage(
            rowwise_data=data,
            rowwise_scale_inv=scales,
            columnwise_data=None,
            columnwise_scale_inv=None,
            amax_rowwise=amax,
            amax_columnwise=None,
            fp4_dtype=tex.DType.kFloat4E2M1,
            quantizer=quantizer,
            with_gemm_swizzled_scales=True,
            fake_dtype=dtype,
        )
    raise ValueError(f"Unknown packed recipe: {recipe}")


def _make_operands(recipe, x, w, dtype):
    seq, micro_batch, groups, hidden = x.shape
    _, out_features, _ = w.shape
    rows = seq * micro_batch
    x_mats = [x[:, :, g, :].reshape(rows, hidden).contiguous() for g in range(groups)]
    w_mats = [w[g].contiguous() for g in range(groups)]

    if recipe == "bf16":
        return w, x, w_mats, x_mats

    w_quantized = _quantize_operands(recipe, w_mats)
    x_quantized = _quantize_operands(recipe, x_mats)
    packed_hidden = hidden // 2 if recipe == "nvfp4" else hidden
    w_data = torch.cat([_rowwise_data(tensor, recipe).reshape(-1) for tensor in w_quantized]).view(
        groups, out_features, packed_hidden
    )
    x_data = torch.empty(seq, micro_batch, groups, packed_hidden, dtype=torch.uint8, device="cuda")
    for g, tensor in enumerate(x_quantized):
        x_data[:, :, g, :].copy_(
            _rowwise_data(tensor, recipe).view(seq, micro_batch, packed_hidden)
        )
    return (
        _packed_storage(recipe, w_data, w_quantized, dtype),
        _packed_storage(recipe, x_data, x_quantized, dtype),
        w_quantized,
        x_quantized,
    )


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("recipe", ["bf16", "mxfp8"])
def test_strided_batched_gemm_interleaved_activation(recipe, accumulate):
    _skip_if_unavailable(recipe)

    torch.manual_seed(1234)
    dtype = torch.bfloat16
    seq, micro_batch, groups, hidden, out_features = 16, 8, 3, 128, 128
    rows = seq * micro_batch

    # Logical operation:
    #   X: [S, B, G, D], W: [G, R, D] -> Y: [S, B, G, R]
    # The GEMM view for each group is:
    #   X_g: [S*B, D], W_g: [R, D], Y_g: [S*B, R].
    x = torch.randn(seq, micro_batch, groups, hidden, dtype=dtype, device="cuda")
    w = torch.randn(groups, out_features, hidden, dtype=dtype, device="cuda")
    A, B, ref_A, ref_B = _make_operands(recipe, x, w, dtype)

    out = torch.empty(seq, micro_batch, groups, out_features, dtype=dtype, device="cuda")
    out_initial = None
    if accumulate:
        out_initial = torch.randn_like(out)
        out.copy_(out_initial)

    strided_batched_gemm(
        A,
        B,
        out,
        m=out_features,
        n=rows,
        k=hidden,
        batch_count=groups,
        lda=hidden,
        stridea=out_features * hidden,
        ldb=groups * hidden,
        strideb=hidden,
        ldd=groups * out_features,
        strided=out_features,
        layout="TN",
        accumulate=accumulate,
    )

    ref = out_initial.clone() if accumulate else torch.empty_like(out)
    for g in range(groups):
        if accumulate:
            out_g = ref[:, :, g, :].reshape(rows, out_features).contiguous()
        else:
            out_g = torch.empty(rows, out_features, dtype=dtype, device="cuda")
        general_gemm(
            ref_A[g],
            ref_B[g],
            out_dtype=dtype,
            out=out_g,
            layout="TN",
            accumulate=accumulate,
        )
        ref[:, :, g, :].copy_(out_g.view(seq, micro_batch, out_features))

    torch.testing.assert_close(out, ref, rtol=0.125, atol=0.0675)


@pytest.mark.parametrize("api", ["python", "cpp"])
@pytest.mark.parametrize("recipe", ["fp8", "fp8_block", "nvfp4"])
def test_strided_batched_gemm_rejects_unsupported_inputs(recipe, api):
    _skip_if_unavailable(recipe)

    torch.manual_seed(1234)
    dtype = torch.bfloat16
    seq, micro_batch, groups, hidden, out_features = 16, 8, 2, 128, 128
    rows = seq * micro_batch
    x = torch.randn(seq, micro_batch, groups, hidden, dtype=dtype, device="cuda")
    w = torch.randn(groups, out_features, hidden, dtype=dtype, device="cuda")
    A, B, _, _ = _make_operands(recipe, x, w, dtype)
    out = torch.empty(seq, micro_batch, groups, out_features, dtype=dtype, device="cuda")

    if api == "python":
        with pytest.raises(AssertionError, match=_UNSUPPORTED_INPUT_ERROR):
            strided_batched_gemm(
                A,
                B,
                out,
                m=out_features,
                n=rows,
                k=hidden,
                batch_count=groups,
                lda=hidden,
                stridea=out_features * hidden,
                ldb=groups * hidden,
                strideb=hidden,
                ldd=groups * out_features,
                strided=out_features,
                layout="TN",
            )
        return

    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match=_UNSUPPORTED_INPUT_ERROR):
        tex.strided_batched_gemm(
            A,
            True,
            B,
            False,
            out,
            workspace,
            workspace.numel(),
            out_features,
            rows,
            hidden,
            groups,
            hidden,
            out_features * hidden,
            groups * hidden,
            hidden,
            groups * out_features,
            out_features,
            False,
            False,
            0,
            1.0,
            0.0,
        )
