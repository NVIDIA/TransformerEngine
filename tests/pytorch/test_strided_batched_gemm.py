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
import transformer_engine_torch as tex

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

_UNSUPPORTED_INPUT_ERROR = (
    "strided_batched_gemm supports only high-precision or MXFP8 A and B tensor pairs"
)
_SWIZZLED_MXFP8_ERROR = "expects compact MXFP8 scales"


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


def _quantize_mxfp8_batched(tensor, batch_dim, *, rowwise=True, columnwise=False):
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    quantizer_input = tensor
    if columnwise and batch_dim == -2:
        groups = tensor.size(0 if batch_dim == 0 else tensor.ndim - 2)
        hidden = tensor.size(-1)
        rows = tensor.numel() // (groups * hidden)
        quantizer_input = tensor.view(rows, groups * hidden)
    return quantizer(quantizer_input)


def _quantize_mxfp8_reference(tensors, *, rowwise=True, columnwise=False):
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    quantizer.optimize_for_gemm = True
    return [quantizer(tensor) for tensor in tensors]


def _quantize_unsupported_operands(recipe, x, w):
    if recipe == "fp8":
        quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device="cuda"),
            amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
    elif recipe == "fp8_block":
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            force_pow_2_scales=True,
            amax_epsilon=0.0,
            block_scaling_dim=1,
        )
    elif recipe == "nvfp4":
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
    else:
        raise ValueError(f"Unknown quantized recipe: {recipe}")
    return quantizer(w), quantizer(x)


def _cpp_strided_batched_gemm(
    A,
    B,
    out,
    *,
    batch_dim_a=0,
    batch_dim_b=0,
    batch_dim_out=0,
    workspace=None,
):
    if workspace is None:
        workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    return tex.strided_batched_gemm(
        A,
        True,
        batch_dim_a,
        B,
        False,
        batch_dim_b,
        out,
        batch_dim_out,
        workspace,
        False,
        False,
        0,
        1.0,
        0.0,
    )


def _interleaved_operands(seq, micro_batch, groups, hidden, out_features, dtype=torch.bfloat16):
    """Logical operation: X: [S, B, G, D], W: [G, R, D] -> Y: [S, B, G, R]"""
    x = torch.randn(seq, micro_batch, groups, hidden, dtype=dtype, device="cuda")
    w = torch.randn(groups, out_features, hidden, dtype=dtype, device="cuda")
    out = torch.empty(seq, micro_batch, groups, out_features, dtype=dtype, device="cuda")
    return x, w, out


_INTERLEAVED_BATCH_DIMS = {"batch_dim_a": 0, "batch_dim_b": -2, "batch_dim_out": -2}


@pytest.mark.parametrize(
    "invalid_buffer",
    [
        "A",
        "B",
        "D_expanded",
        "workspace_stride",
        "A_device",
        "B_device",
        "workspace_device",
    ],
)
def test_strided_batched_gemm_invalid_buffers(invalid_buffer):
    """Reject invalid buffers in the binding before launching CUDA work."""
    a = torch.empty(2, 32, 32, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    out = torch.empty_like(a)
    workspace = torch.empty(32, device="cuda", dtype=torch.uint8)
    if invalid_buffer == "A":
        a = a.cpu()
        error = "same CUDA device as D"
    elif invalid_buffer == "B":
        b = b.cpu()
        error = "same CUDA device as D"
    elif invalid_buffer == "D_expanded":
        # Expanded tensor has fewer stored elements than logical elements
        out = torch.empty(1, 1, 1, device="cuda", dtype=torch.float32).expand(2, 32, 32)
        error = "D must be contiguous"
    elif invalid_buffer == "workspace_stride":
        workspace = workspace[::2]
        error = "workspace must be contiguous"
    else:
        if torch.cuda.device_count() < 2:
            pytest.skip("Mixed-device validation requires two CUDA devices.")
        other_device = (out.device.index + 1) % torch.cuda.device_count()
        error = "same CUDA device as D"
        if invalid_buffer == "A_device":
            a = a.to(device=other_device)
        elif invalid_buffer == "B_device":
            b = b.to(device=other_device)
        else:
            workspace = workspace.to(device=other_device)
            error = "same device as D"
    with pytest.raises(RuntimeError, match=error):
        _cpp_strided_batched_gemm(a, b, out, workspace=workspace)


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("recipe", ["bf16", "mxfp8"])
def test_strided_batched_gemm_interleaved_activation(recipe, accumulate):
    _skip_if_unavailable(recipe)

    torch.manual_seed(1234)
    dtype = torch.bfloat16
    seq, micro_batch, groups, hidden, out_features = 12, 8, 3, 160, 96
    rows = seq * micro_batch

    # The GEMM view for each group is:
    #   X_g: [S*B, D], W_g: [R, D], Y_g: [S*B, R].
    x, w, out = _interleaved_operands(seq, micro_batch, groups, hidden, out_features)
    x_mats = [x[:, :, g, :].reshape(rows, hidden).contiguous() for g in range(groups)]
    w_mats = list(w.unbind(dim=0))
    if recipe == "bf16":
        A, B = w, x
        ref_A, ref_B = w_mats, x_mats
    else:
        A = _quantize_mxfp8_batched(w, batch_dim=0)
        B = _quantize_mxfp8_batched(x, batch_dim=-2)
        scale_ptrs = (A._rowwise_scale_inv.data_ptr(), B._rowwise_scale_inv.data_ptr())
        assert not A._with_gemm_swizzled_scales
        assert not B._with_gemm_swizzled_scales
        ref_A = _quantize_mxfp8_reference(w_mats)
        ref_B = _quantize_mxfp8_reference(x_mats)

    out_initial = None
    if accumulate:
        out_initial = torch.randn_like(out)
        out.copy_(out_initial)

    strided_batched_gemm(
        A,
        B,
        out,
        layout="TN",
        accumulate=accumulate,
        **_INTERLEAVED_BATCH_DIMS,
    )

    if recipe == "mxfp8":
        assert (A._rowwise_scale_inv.data_ptr(), B._rowwise_scale_inv.data_ptr()) == scale_ptrs
        assert not A._with_gemm_swizzled_scales
        assert not B._with_gemm_swizzled_scales

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
def test_strided_batched_gemm_rejects_swizzled_mxfp8_scales(api):
    _skip_if_unavailable("mxfp8")

    torch.manual_seed(1234)
    x, w, out = _interleaved_operands(12, 8, 3, 160, 96)
    A, B = _quantize_mxfp8_reference([w, x])
    assert A._with_gemm_swizzled_scales and B._with_gemm_swizzled_scales

    if api == "python":
        with pytest.raises(ValueError, match=_SWIZZLED_MXFP8_ERROR):
            strided_batched_gemm(A, B, out, layout="TN", **_INTERLEAVED_BATCH_DIMS)
    else:
        with pytest.raises(RuntimeError, match=_SWIZZLED_MXFP8_ERROR):
            _cpp_strided_batched_gemm(A, B, out, **_INTERLEAVED_BATCH_DIMS)


@pytest.mark.parametrize("recipe", ["fp8", "fp8_block", "nvfp4"])
def test_strided_batched_gemm_unsupported_inputs_fallback(recipe):
    """The Python API dequantizes unsupported formats."""
    _skip_if_unavailable(recipe)

    torch.manual_seed(1234)
    x, w, out = _interleaved_operands(16, 8, 2, 128, 128)
    A, B = _quantize_unsupported_operands(recipe, x, w)

    with pytest.warns(UserWarning, match="Falling back to high-precision GEMM"):
        strided_batched_gemm(A, B, out, layout="TN", **_INTERLEAVED_BATCH_DIMS)

    # The fallback dequantizes to the output dtype
    w_ref = A.dequantize(dtype=out.dtype).float()
    x_ref = B.dequantize(dtype=out.dtype).float()
    ref = torch.einsum("sbgd,grd->sbgr", x_ref, w_ref)
    torch.testing.assert_close(out.float(), ref, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("recipe", ["fp8", "fp8_block", "nvfp4"])
def test_strided_batched_gemm_rejects_unsupported_inputs(recipe):
    """The C++ binding only supports high-precision and MXFP8 data."""
    _skip_if_unavailable(recipe)

    torch.manual_seed(1234)
    x, w, out = _interleaved_operands(16, 8, 2, 128, 128)
    A, B = _quantize_unsupported_operands(recipe, x, w)
    with pytest.raises(RuntimeError, match=_UNSUPPORTED_INPUT_ERROR):
        _cpp_strided_batched_gemm(A, B, out, **_INTERLEAVED_BATCH_DIMS)


@pytest.mark.parametrize("api", ["python", "cpp"])
@pytest.mark.parametrize(
    "invalid_dim,error",
    [
        ("batch_size", "inconsistent batch sizes"),
        ("k", "incompatible matrix dims"),
        ("out", "incompatible matrix dims"),
        ("batch_dim", "supports batch_dim=0 or batch_dim=-2"),
    ],
)
def test_strided_batched_gemm_rejects_inconsistent_shapes(api, invalid_dim, error):
    _skip_if_unavailable("bf16")

    groups, m, n, k = 2, 32, 64, 96
    A_shape, B_shape, out_shape = [groups, m, k], [groups, n, k], [groups, n, m]
    batch_dims = {"batch_dim_a": 0, "batch_dim_b": 0, "batch_dim_out": 0}
    if invalid_dim == "batch_size":
        A_shape[0] += 1
    elif invalid_dim == "k":
        B_shape[-1] += 32
    elif invalid_dim == "out":
        out_shape[-1] += 32
    else:
        batch_dims["batch_dim_a"] = 1
    A = torch.randn(A_shape, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(B_shape, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(out_shape, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match=error):
        if api == "python":
            strided_batched_gemm(A, B, out, layout="TN", **batch_dims)
        else:
            _cpp_strided_batched_gemm(A, B, out, **batch_dims)


@pytest.mark.parametrize("api", ["python", "cpp"])
@pytest.mark.parametrize("buffer_name", ["A", "B"])
def test_strided_batched_gemm_rejects_invalid_compact_scale_shape(api, buffer_name):
    _skip_if_unavailable("mxfp8")

    groups, m, n, k = 2, 128, 128, 128
    A = _quantize_mxfp8_batched(
        torch.randn(groups, m, k, dtype=torch.bfloat16, device="cuda"),
        batch_dim=0,
    )
    B = _quantize_mxfp8_batched(
        torch.randn(groups, n, k, dtype=torch.bfloat16, device="cuda"),
        batch_dim=0,
    )
    operand = A if buffer_name == "A" else B
    operand._rowwise_scale_inv = operand._rowwise_scale_inv.reshape(-1)[:-1]
    out = torch.empty(groups, n, m, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match="expects 2D compact scaling factors"):
        if api == "python":
            strided_batched_gemm(A, B, out, layout="TN")
        else:
            _cpp_strided_batched_gemm(A, B, out)


def test_strided_batched_gemm_two_dim_mxfp8_operand():
    """A 2D [G, C] MXFP8 operand uses batch_dim=0.

    With batch_dim=-2, 2D MXFP8 data is interpreted as a [R, G*C] view
    of a [R, G, C] tensor.

    """
    _skip_if_unavailable("mxfp8")

    torch.manual_seed(1234)
    # Note: MXFP8 quantization requires 32 | groups. The error check
    # requires groups not to divide hidden.
    groups, out_features, hidden = 64, 64, 96
    x = torch.randn(groups, hidden, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(groups, out_features, hidden, dtype=torch.bfloat16, device="cuda")
    A = _quantize_mxfp8_batched(w, batch_dim=0)
    B = _quantize_mxfp8_batched(x, batch_dim=0)
    out = torch.empty(groups, out_features, dtype=torch.bfloat16, device="cuda")

    strided_batched_gemm(A, B, out, layout="TN")
    ref = torch.einsum("gd,grd->gr", B.dequantize().float(), A.dequantize().float())
    torch.testing.assert_close(out.float(), ref, rtol=1.6e-2, atol=1e-2)

    with pytest.raises(RuntimeError, match="use batch_dim=0"):
        strided_batched_gemm(A, B, out, layout="TN", batch_dim_b=-2, batch_dim_out=-2)
