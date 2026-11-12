# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Portable QDQ numerics and integration, including Ampere/Hopper CI devices.

The oracle uses PyTorch arithmetic and explicit FP4 rounding, not TE's native
packed quantization kernels, which require Blackwell. The existing NVFP4 L0
directory invocation discovers this file on every hardware generation.
"""

from functools import partial

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch.custom_recipes.qdq import (
    MXFP8QDQQuantizer,
    NVFP4QDQQuantizer,
)
from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
    nvfp4_qdq_weight_fwd_bf16_bwd_factory,
)
from transformer_engine.pytorch.custom_recipes.reference_nvfp4 import (
    cast_from_fp4x2,
    cast_to_fp4x2,
)
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.module import _split_quantization

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="Portable QDQ requires compute capability 8.0 or higher",
)

requires_fp4_build = pytest.mark.skipif(
    not tex.nvfp4_qdq_supported(),
    reason="NVFP4 QDQ requires a TE build with CUDA 12.8 or higher",
)
portable_formats = [
    pytest.param("nvfp4", marks=requires_fp4_build),
    pytest.param("nvfp4_row", marks=requires_fp4_build),
    "mxfp8",
]


@pytest.fixture(autouse=True)
def ordinary_math(monkeypatch):
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")


def quantizer(fmt, backend="auto"):
    if fmt == "mxfp8":
        return MXFP8QDQQuantizer(backend=backend)
    return NVFP4QDQQuantizer(NVFP4Quantizer(row_scaled_nvfp4=fmt == "nvfp4_row"), backend=backend)


def reference(x, fmt):
    """Independent oracle with native rounding, fmax semantics, and decode order."""
    values = x.float().reshape(-1, x.shape[-1])
    width = 32 if fmt == "mxfp8" else 16
    blocks = values.reshape(values.shape[0], -1, width)
    magnitudes = torch.fmax(blocks.abs(), torch.zeros_like(blocks))
    block_amax = magnitudes.amax(dim=-1, keepdim=True)
    if fmt == "mxfp8":
        mantissa, exponent = torch.frexp(block_amax * (1.0 / 448.0))
        exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
        exponent = torch.where(block_amax == 0, -127, exponent).clamp(-127, 127)
        exponent = torch.where(torch.isinf(block_amax), 127, exponent)
        decode = torch.exp2(exponent.float())
        inverse = torch.exp2(-exponent.float())
        encoded = (blocks * inverse).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    else:
        outer = (
            magnitudes.amax(dim=(-1, -2), keepdim=True) if fmt == "nvfp4_row" else magnitudes.amax()
        )
        # PyTorch FP32 division may use an approximate reciprocal. Round an
        # FP64 quotient to FP32 to model CUDA's ordinary div.rn operation.
        encode = ((6.0 * 448.0) / outer.double()).float()
        encode = encode.clamp(max=torch.finfo(torch.float32).max)
        encode = torch.where((outer == 0) | torch.isinf(outer), 1.0, encode)
        scale = (block_amax * (encode * (1.0 / 6.0))).clamp(max=448)
        scale = scale.to(torch.float8_e4m3fn).float()
        reciprocal_encode = (1.0 / encode.double()).float()
        inverse = (1.0 / (scale * reciprocal_encode).double()).float()
        inverse = inverse.clamp(max=torch.finfo(torch.float32).max)
        scaled = blocks * inverse
        # Reuse the independent reference's explicit E2M1 tie boundaries.
        matrix = scaled.reshape(values.shape)
        encoded = cast_from_fp4x2(cast_to_fp4x2(matrix), torch.float32)
        # That reference's packing omits the sign of -0; retain it explicitly.
        encoded = torch.copysign(encoded.abs(), matrix).reshape(blocks.shape)
        encoded = torch.where(torch.isnan(scaled), 6.0, encoded)
        decode = scale * outer * (1.0 / (6.0 * 448.0))
    return (encoded * decode).reshape(x.shape).to(x.dtype)


def inputs(shape, dtype, distribution):
    generator = torch.Generator(device="cuda").manual_seed(123)
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    if distribution == "zeros":
        x.zero_()
        x[..., ::2] = -0.0
    elif distribution == "outliers":
        x.reshape(-1, shape[-1])[0, 0] = 4096
        x.reshape(-1, shape[-1])[1::2] *= 1e-5
    elif distribution == "tiny":
        x.fill_(torch.finfo(dtype).tiny / 4)
        x[..., ::2].neg_()
        x.reshape(-1, shape[-1])[0].zero_()
    elif distribution == "boundaries":
        values = torch.tensor(
            [0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -6, 448, 1e-7, 1e-4, -5, -0.25],
            device="cuda",
            dtype=dtype,
        )
        x.copy_(values.repeat(x.numel() // 16).reshape(x.shape))
    return x


@pytest.mark.parametrize("fmt", portable_formats)
@pytest.mark.parametrize("backend", ["auto", "fused"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(16, 32), (48, 160), (2, 8, 4096)])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers", "boundaries", "tiny"])
def test_portable_numerics(fmt, backend, dtype, shape, distribution):
    x = inputs(shape, dtype, distribution)
    q = quantizer(fmt, backend)
    assert q.selected_backend(x) == "fused"
    actual = q(x).dequantize()
    expected = reference(x, fmt)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    # Validate the independent oracle against the native path where it exists.
    native_shape = fmt != "mxfp8" or (x.numel() // x.shape[-1]) % 32 == 0
    if torch.cuda.get_device_capability() >= (10, 0) and native_shape:
        native = quantizer(fmt, "reference")(x).dequantize()
        torch.testing.assert_close(actual, native, rtol=0, atol=0)


@pytest.mark.parametrize("fmt", portable_formats)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "all_nan"])
def test_portable_exceptional_values(fmt, dtype, value):
    x = inputs((32, 128), dtype, "normal")
    if value == "all_nan":
        x.fill_(float("nan"))
    else:
        x[0, 0] = value
    actual = quantizer(fmt)(x).dequantize()
    expected = reference(x, fmt)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    valid = ~torch.isnan(expected)
    assert torch.equal(actual[valid].view(torch.int16), expected[valid].view(torch.int16))
    if torch.cuda.get_device_capability() >= (10, 0):
        native = quantizer(fmt, "reference")(x).dequantize()
        assert torch.equal(torch.isnan(actual), torch.isnan(native))
        torch.testing.assert_close(actual[valid], native[valid], rtol=0, atol=0)


@pytest.mark.parametrize("fmt", portable_formats)
def test_portable_lifecycle_and_graph(fmt):
    x = inputs((32, 128), torch.bfloat16, "normal").requires_grad_()
    q = quantizer(fmt)
    y = q(x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    pointer = y._hp_data.data_ptr()
    noop = torch.ones(1, device=x.device)
    q.update_quantized(x * 2, y, noop_flag=noop)
    torch.testing.assert_close(y.dequantize(), reference(x, fmt), rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        q.update_quantized(x, y, noop_flag=noop)
    with torch.no_grad():
        x.mul_(2)
    noop.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert y._hp_data.data_ptr() == pointer
    torch.testing.assert_close(y.dequantize(), reference(x, fmt), rtol=0, atol=0)


@pytest.mark.parametrize("grouped", [False, True])
@requires_fp4_build
def test_portable_weight_recipe(grouped):
    factory = partial(nvfp4_qdq_weight_fwd_bf16_bwd_factory, backend="fused")
    recipe = CustomRecipe(qfactory=factory)
    options = dict(bias=False, params_dtype=torch.bfloat16, device="cuda")
    layer = (
        te.GroupedLinear(2, 128, 128, single_grouped_weight=False, **options)
        if grouped
        else te.Linear(128, 128, **options)
    )
    x = inputs((32, 128), torch.bfloat16, "normal").requires_grad_()
    dy = torch.ones_like(x)
    weights = [layer.weight0, layer.weight1] if grouped else [layer.weight]
    splits = [16, 16] if grouped else [32]
    with te.autocast(recipe=recipe), torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x, splits) if grouped else layer(x)
    y.backward(dy)
    expected_y, expected_dx = [], []
    for weight, source, grad in zip(weights, x.detach().split(splits), dy.split(splits)):
        decoded = reference(weight.detach(), "nvfp4")
        expected_y.append(source @ decoded.T)
        expected_dx.append(grad @ decoded)
        torch.testing.assert_close(weight.grad, grad.T @ source, rtol=0.016, atol=0.002)
    torch.testing.assert_close(y, torch.cat(expected_y), rtol=0.016, atol=0.002)
    torch.testing.assert_close(x.grad, torch.cat(expected_dx), rtol=0.016, atol=0.002)


@pytest.mark.parametrize("fmt", portable_formats)
@pytest.mark.parametrize("grouped", [False, True])
def test_portable_full_qdq_recipe(fmt, grouped, monkeypatch):
    """QDQ operands use BF16 GEMMs without requiring native FP8 hardware."""
    monkeypatch.setattr(
        FP8GlobalStateManager,
        "is_fp8_available",
        classmethod(lambda cls: (False, "FP8 hardware unavailable")),
    )

    def factory(role):
        if role is not None and role.tensor_type in ("input", "weight"):
            return quantizer(fmt)
        return te.IdentityQuantizer()

    recipe = CustomRecipe(qfactory=factory)
    options = dict(bias=False, params_dtype=torch.bfloat16, device="cuda")
    layer = (
        te.GroupedLinear(2, 128, 128, single_grouped_weight=False, **options)
        if grouped
        else te.Linear(128, 128, **options)
    )
    x = inputs((64, 128), torch.bfloat16, "normal").requires_grad_()
    dy = torch.ones_like(x)
    weights = [layer.weight0, layer.weight1] if grouped else [layer.weight]
    splits = [32, 32] if grouped else [64]
    with te.autocast(recipe=recipe), torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x, splits) if grouped else layer(x)
    y.backward(dy)
    expected_y, expected_dx = [], []
    for weight, source, grad in zip(weights, x.detach().split(splits), dy.split(splits)):
        decoded_x = reference(source, fmt)
        decoded_w = reference(weight.detach(), fmt)
        expected_y.append(decoded_x @ decoded_w.T)
        expected_dx.append(grad @ decoded_w)
        torch.testing.assert_close(weight.grad, grad.T @ decoded_x, rtol=0.016, atol=0.002)
    torch.testing.assert_close(y, torch.cat(expected_y), rtol=0.016, atol=0.002)
    torch.testing.assert_close(x.grad, torch.cat(expected_dx), rtol=0.016, atol=0.002)


@pytest.mark.parametrize("fmt", portable_formats)
def test_portable_device_and_stream(fmt):
    device = 1 if torch.cuda.device_count() > 1 else 0
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
        x = inputs((32, 128), torch.bfloat16, "normal")
        actual = quantizer(fmt)(x).dequantize()
        assert torch.equal(actual.view(torch.int16), reference(x, fmt).view(torch.int16))


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_older_device_reference_rejected(fmt, monkeypatch):
    x = inputs((32, 128), torch.bfloat16, "normal")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(RuntimeError, match="Native.*Q\\+DQ requires compute capability 10.0"):
        quantizer(fmt, "reference")(x)


@pytest.mark.parametrize("fp4_supported", [False, True])
@pytest.mark.parametrize("fmt", ["nvfp4", "nvfp4_row"])
def test_nvfp4_build_support_dispatch(fp4_supported, fmt, monkeypatch):
    from transformer_engine.pytorch.custom_recipes import qdq

    monkeypatch.setattr(qdq, "_NVFP4_QDQ_SUPPORTED", fp4_supported)
    x = inputs((32, 128), torch.bfloat16, "normal")
    expected = "fused" if fp4_supported else "reference"
    assert quantizer(fmt).selected_backend(x) == expected
    assert quantizer("mxfp8").selected_backend(x) == "fused"
    if not fp4_supported:
        with pytest.raises(ValueError, match="Forced fused NVFP4 QDQ"):
            quantizer(fmt, "fused").selected_backend(x)


@pytest.mark.parametrize("fmt", portable_formats)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("disable_bulk", [False, True])
def test_split_qdq_exact(fmt, dtype, internal, disable_bulk, monkeypatch):
    splits = [0, 16, 32, 0, 48]
    x = inputs((sum(splits), 128), dtype, "normal")
    # Different expert ranges must not share a tensor-wide NVFP4 amax.
    x[16:48].mul_(64)
    qs = [quantizer(fmt, "fused") for _ in splits]
    for q in qs:
        q.internal = internal

    def unexpected_quantize(*args, **kwargs):
        pytest.fail("Fused split QDQ reentered Python quantization")

    def unexpected_native_split(*args, **kwargs):
        pytest.fail("QDQ entered native split_quantize")

    monkeypatch.setattr(type(qs[0]), "quantize_impl", unexpected_quantize)
    monkeypatch.setattr(tex, "split_quantize", unexpected_native_split)
    result = _split_quantization._split_quantize_non_hybrid(
        x, splits, qs, dtype, disable_bulk_allocation=disable_bulk
    )
    for src, out, q in zip(x.split(splits), result, qs):
        assert out.shape == src.shape
        assert out._dtype == dtype
        assert out._hp_data.grad_fn is None
        assert type(out._get_quantizer()) is type(q)
        if src.numel():
            assert torch.equal(
                out.dequantize().view(torch.int16), reference(src, fmt).view(torch.int16)
            )
    if not disable_bulk:
        nonempty = [out._hp_data for out in result if out._hp_data.numel()]
        assert len({out.untyped_storage().data_ptr() for out in nonempty}) == 1


@pytest.mark.parametrize("fmt", portable_formats)
@pytest.mark.parametrize("fallback", ["reference", "dtype", "noncontiguous"])
def test_split_qdq_python_fallback(fmt, fallback, monkeypatch):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("Native packed Q+DQ fallback requires compute capability 10.0+")
    splits = [32, 0, 64]
    x = inputs((sum(splits), 128), torch.bfloat16, "normal")
    if fallback == "noncontiguous":
        x = x.T.contiguous().T
    qs = [quantizer(fmt, "reference" if fallback == "reference" else "auto") for _ in splits]
    if fallback == "dtype":
        # One ineligible expert sends the operand through the Python fallback.
        qs[-1].dtype = torch.float32
    expected = [q(src).dequantize() for src, q in zip(x.split(splits), qs)]
    calls = []
    original = type(qs[0]).quantize_impl

    def tracked(q, src, **kwargs):
        calls.append(src.shape)
        return original(q, src, **kwargs)

    monkeypatch.setattr(type(qs[0]), "quantize_impl", tracked)
    actual = tex.split_quantize_custom(x, splits, qs)
    assert calls == [torch.Size([32, 128]), torch.Size([64, 128])]
    for out, ref in zip(actual, expected):
        assert out.dtype == ref.dtype
        torch.testing.assert_close(out.dequantize(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("fmt", portable_formats)
def test_split_qdq_empty_and_forced_backend(fmt):
    qs = [quantizer(fmt, "fused") for _ in range(2)]
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    qs[1].dtype = torch.float32
    result = tex.split_quantize_custom(x, [0, 0], qs)
    assert [out.dtype for out in result] == [torch.bfloat16, torch.float32]
    assert all(out.numel() == 0 for out in result)
    x = inputs((32, 128), torch.bfloat16, "normal")
    with pytest.raises(ValueError, match="Forced fused"):
        tex.split_quantize_custom(x, [16, 16], qs)


@pytest.mark.parametrize("fmt", portable_formats)
def test_split_qdq_graph_and_device(fmt):
    device = 1 if torch.cuda.device_count() > 1 else 0
    with torch.cuda.device(device):
        x = inputs((48, 128), torch.bfloat16, "normal")
        qs = [quantizer(fmt, "fused") for _ in range(2)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            tex.split_quantize_custom(x, [16, 32], qs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = tex.split_quantize_custom(x, [16, 32], qs)
        x.mul_(2)
        graph.replay()
        torch.cuda.synchronize()
        for src, out in zip(x.split([16, 32]), outputs):
            assert torch.equal(
                out.dequantize().view(torch.int16), reference(src, fmt).view(torch.int16)
            )


@pytest.mark.parametrize("fmt", portable_formats)
@pytest.mark.parametrize("source", ["original", "rowwise_dequantized"])
def test_hybrid_split_qdq_dispatch(fmt, source, monkeypatch):
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
    from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer

    splits = [16, 32]
    x = inputs((48, 128), torch.bfloat16, "normal")
    qs = [
        HybridQuantizer(
            rowwise_quantizer=quantizer(fmt, "fused"),
            columnwise_quantizer=IdentityQuantizer(),
            columnwise_source=source,
        )
        for _ in splits
    ]
    calls = []
    original = tex.split_quantize_custom

    def tracked(tensor, sizes, quantizers, **kwargs):
        calls.append(type(quantizers[0]))
        return original(tensor, sizes, quantizers, **kwargs)

    monkeypatch.setattr(tex, "split_quantize_custom", tracked)
    actual, _ = _split_quantization._split_quantize(x, splits, qs, x.dtype)
    assert calls == [type(qs[0].rowwise_quantizer)]
    for src, out in zip(x.split(splits), actual):
        decoded = reference(src, fmt)
        torch.testing.assert_close(out.rowwise_sub_storage.dequantize(), decoded, rtol=0, atol=0)
        expected_col = src if source == "original" else decoded
        torch.testing.assert_close(
            out.columnwise_sub_storage.dequantize(), expected_col, rtol=0, atol=0
        )


@pytest.mark.parametrize("fmt", portable_formats)
@pytest.mark.parametrize("disable_bulk", [False, True])
def test_split_qdq_contiguous_slices_of_strided_input(fmt, disable_bulk):
    # The leading dimension has gaps, but each expert is a contiguous matrix.
    x = inputs((3, 32, 128), torch.bfloat16, "normal")[:, :16, :]
    splits = [1, 1, 1]
    assert not x.is_contiguous()
    assert all(part.is_contiguous() for part in x.split(splits))
    qs = [quantizer(fmt, "fused") for _ in splits]
    actual = tex.split_quantize_custom(x, splits, qs, disable_bulk_allocation=disable_bulk)
    for part, out in zip(x.split(splits), actual):
        assert torch.equal(
            out.dequantize().view(torch.int16), reference(part, fmt).view(torch.int16)
        )
