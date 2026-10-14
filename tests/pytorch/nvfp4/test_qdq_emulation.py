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


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="Portable QDQ requires compute capability 8.0 or higher",
)


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


@pytest.mark.parametrize("fmt", ["nvfp4", "nvfp4_row", "mxfp8"])
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


@pytest.mark.parametrize("fmt", ["nvfp4", "nvfp4_row", "mxfp8"])
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


@pytest.mark.parametrize("fmt", ["nvfp4", "nvfp4_row", "mxfp8"])
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


@pytest.mark.parametrize("fmt", ["nvfp4", "nvfp4_row", "mxfp8"])
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
