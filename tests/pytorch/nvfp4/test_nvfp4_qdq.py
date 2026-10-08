# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Native parity, lifecycle, and operand-selection tests for weight QDQ."""

from functools import partial

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.custom_recipes.qdq import NVFP4QDQQuantizer
from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
    mxfp8_nvfp4_qdq_fwd_mxfp8_bwd_factory,
    nvfp4_qdq_weight_fwd_bf16_bwd_factory,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="QDQ fused implementation targets SM100",
)


def native_qdq(x, dtype=None, **options):
    q = NVFP4Quantizer(rowwise=True, columnwise=False, **options)
    q.internal = True
    return q(x).dequantize(dtype=dtype or x.dtype)


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(16, 16), (48, 80), (128, 256), (512, 1024)])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers", "boundaries"])
def test_plain_exact(dtype, shape, distribution, monkeypatch):
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if distribution == "zeros":
        x.zero_()
        x[:, ::2] = -0.0
    elif distribution == "outliers":
        x[0, 0] = 4096
        x[1, :] *= 1e-5
    elif distribution == "boundaries":
        values = torch.tensor(
            [
                0,
                -0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                6.0,
                -6.0,
                -5.0,
                -2.5,
                1e-7,
                1e-4,
                448.0,
            ],
            device="cuda",
            dtype=dtype,
        )
        x.copy_(values.repeat(shape[0], shape[1] // 16))
    q = NVFP4QDQQuantizer(backend="fused")
    assert q.selected_backend(x) == "fused"
    assert_bits(q(x).dequantize(), native_qdq(x))


@pytest.mark.parametrize("mode", ["MAE", "MSE"])
@pytest.mark.parametrize("maximum", [256, 448])
@pytest.mark.parametrize("fast_error", ["0", "1"])
def test_4over6_fallback(mode, maximum, fast_error, monkeypatch):
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", fast_error)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    options = dict(nvfp4_use_4over6=True, nvfp4_e4m3_max=maximum, nvfp4_4over6_err_mode=mode)
    q = NVFP4QDQQuantizer(NVFP4Quantizer(**options))
    assert q.selected_backend(x) == "reference"
    assert_bits(q(x).dequantize(), native_qdq(x, **options))
    q.backend = "fused"
    with pytest.raises(ValueError, match="Forced fused"):
        q(x)


@pytest.mark.parametrize("options", [{"with_2d_quantization": True}, {}])
def test_dtype_fallback(options):
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    q = NVFP4QDQQuantizer(NVFP4Quantizer(**options))
    actual = q.quantize(x, dtype=torch.bfloat16)
    assert_bits(actual.dequantize(), native_qdq(x, torch.bfloat16, **options))
    assert q.dtype is None
    assert actual.dtype == torch.bfloat16
    actual.quantize_(x * 2)
    assert_bits(actual.dequantize(), native_qdq(x * 2, torch.bfloat16, **options))


@pytest.mark.parametrize("backend", ["reference", "fused"])
@pytest.mark.parametrize("internal", [False, True])
def test_lifecycle_and_graph(backend, internal):
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    q = NVFP4QDQQuantizer(backend=backend)
    q.internal = internal
    y = q(x)
    assert y._get_quantizer().backend == backend
    address = y._hp_data.data_ptr()
    noop = torch.ones(1, device="cuda")
    q.update_quantized(x * 3, y, noop_flag=noop)
    assert_bits(y.dequantize(), native_qdq(x))
    noop.zero_()
    q.update_quantized(x * 2, y, noop_flag=noop)
    assert y._hp_data.data_ptr() == address
    assert_bits(y.dequantize(), native_qdq(x * 2))
    # Warm up on a side stream, then replay with changed input and noop flag.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            q.update_quantized(x, y, noop_flag=noop)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q.update_quantized(x, y, noop_flag=noop)
    x.mul_(2)
    graph.replay()
    assert_bits(y.dequantize(), native_qdq(x))
    previous = y.dequantize().clone()
    noop.fill_(1)
    x.mul_(2)
    graph.replay()
    assert_bits(y.dequantize(), previous)


def test_ste_and_rng():
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q = NVFP4QDQQuantizer(backend="reference")
    q(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    q = NVFP4QDQQuantizer(NVFP4Quantizer(stochastic_rounding=True))
    assert not q.is_requantization_safe()
    state = torch.cuda.get_rng_state()
    actual = q(x).dequantize()
    after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    expected = native_qdq(x, stochastic_rounding=True)
    assert_bits(actual, expected)
    assert torch.equal(after, torch.cuda.get_rng_state())


def mx_value(x, *, columnwise=False):
    q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=not columnwise, columnwise=columnwise)
    return q(x).dequantize()


@pytest.mark.parametrize("backend", ["reference", "fused"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mixed", [False, True])
def test_module_operands(backend, grouped, params_dtype, mixed, nvfp4_options=None):
    if backend == "fused" and params_dtype == torch.float32:
        pytest.skip("Mixed FP32 source/BF16 decode intentionally uses native fallback")
    torch.manual_seed(12)
    factory = (
        mxfp8_nvfp4_qdq_fwd_mxfp8_bwd_factory if mixed else nvfp4_qdq_weight_fwd_bf16_bwd_factory
    )
    recipe = CustomRecipe(qfactory=partial(factory, backend=backend, nvfp4_options=nvfp4_options))
    kwargs = dict(bias=True, params_dtype=params_dtype, device="cuda")
    layer = (
        te.GroupedLinear(2, 128, 128, single_grouped_weight=False, **kwargs)
        if grouped
        else te.Linear(128, 128, **kwargs)
    )
    splits = [32, 64] if grouped else [96]
    x = torch.randn(sum(splits), 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    weights = [getattr(layer, f"weight{i}") for i in range(2)] if grouped else [layer.weight]
    biases = [getattr(layer, f"bias{i}") for i in range(2)] if grouped else [layer.bias]
    with te.autocast(recipe=recipe), torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x, splits) if grouped else layer(x)
    y.backward(dy)
    expected_y, expected_dx = [], []
    for w, b, xs, gs in zip(weights, biases, x.detach().split(splits), dy.split(splits)):
        wh = native_qdq(w.detach(), torch.bfloat16, **(nvfp4_options or {}))
        expected_y.append(
            torch.nn.functional.linear(
                mx_value(xs) if mixed else xs, wh, b.detach().to(torch.bfloat16)
            )
        )
        bw = mx_value(wh, columnwise=True) if mixed else wh
        bx = mx_value(xs, columnwise=True) if mixed else xs
        grow = mx_value(gs) if mixed else gs
        gcol = mx_value(gs, columnwise=True) if mixed else gs
        expected_dx.append(grow @ bw)
        dw = gcol.T @ bx
        torch.testing.assert_close(w.grad.float(), dw.float(), rtol=0.016, atol=0.002)
        torch.testing.assert_close(
            b.grad.float(),
            gs.float().sum(0).to(b.grad.dtype).float(),
            rtol=0.016,
            atol=0.002,
        )
    torch.testing.assert_close(y, torch.cat(expected_y), rtol=0.016, atol=0.002)
    torch.testing.assert_close(x.grad, torch.cat(expected_dx), rtol=0.016, atol=0.002)


def test_weight_workspace_dtype_and_reuse():
    from transformer_engine.pytorch.module.base import quantize_weight
    from transformer_engine.pytorch.quantization import QuantizerRole

    factory = mxfp8_nvfp4_qdq_fwd_mxfp8_bwd_factory
    q = factory(QuantizerRole(module_type="linear", tensor_type="weight"))
    w = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    out, cache = quantize_weight(tensor=w, quantizer=q, workspace_dtype=torch.bfloat16, cache=True)
    expected = native_qdq(w, torch.bfloat16)
    assert_bits(out._rowwise_storage.dequantize(), expected)
    pointer = out._rowwise_storage._hp_data.data_ptr()
    w.mul_(3)
    out2, new = quantize_weight(
        tensor=w,
        quantizer=q,
        workspace=cache,
        workspace_dtype=torch.bfloat16,
        cache=True,
    )
    assert new is None and out2 is cache
    assert out2._rowwise_storage._hp_data.data_ptr() == pointer
    expected = native_qdq(w, torch.bfloat16)
    assert_bits(out2._rowwise_storage.dequantize(), expected)
    # Hybrid retains native decode-to-source-dtype behavior. Widening the
    # BF16 QDQ value preserves its rounding while MXFP8 metadata stays FP32.
    expected_col = mx_value(expected.to(w.dtype), columnwise=True)
    assert_bits(out2._columnwise_storage.dequantize(), expected_col)
    before = out2._rowwise_storage.dequantize().clone()
    w.mul_(2)
    quantize_weight(
        tensor=w,
        quantizer=q,
        workspace=cache,
        update_workspace=False,
        workspace_dtype=torch.bfloat16,
    )
    assert_bits(out2._rowwise_storage.dequantize(), before)
    out3, cache3 = quantize_weight(
        tensor=w,
        quantizer=q,
        workspace=cache,
        workspace_dtype=torch.float32,
        cache=True,
    )
    assert cache3 is not cache and out3._rowwise_storage.dequantize().dtype == torch.float32
    torch.testing.assert_close(out3._rowwise_storage.dequantize(), native_qdq(w), rtol=0, atol=0)


def test_noncontiguous_and_fast_math_fallback(monkeypatch):
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16).T
    q = NVFP4QDQQuantizer()
    assert q.selected_backend(x) == "reference"
    assert_bits(q(x).dequantize(), native_qdq(x))
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1")
    x = x.contiguous()
    assert q.selected_backend(x) == "reference"
    assert_bits(q(x).dequantize(), native_qdq(x))


@pytest.mark.parametrize("shape", [(0, 128), (128, 0)])
def test_empty_reference(shape):
    x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    q = NVFP4QDQQuantizer()
    assert q.selected_backend(x) == "reference"
    assert_bits(q(x).dequantize(), native_qdq(x))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_exceptional_values(value):
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = value
    actual = NVFP4QDQQuantizer(backend="fused")(x).dequantize()
    expected = native_qdq(x)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    mask = ~torch.isnan(expected)
    assert_bits(actual[mask], expected[mask])


def test_current_stream_and_device():
    device = 1 if torch.cuda.device_count() > 1 else 0
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
        x = torch.randn(48, 80, device=f"cuda:{device}", dtype=torch.bfloat16)
        assert_bits(NVFP4QDQQuantizer(backend="fused")(x).dequantize(), native_qdq(x))


def test_copy_preserves_runtime_reduction_state():
    q = NVFP4QDQQuantizer(backend="reference")
    group = object()
    q.amax_reduction_group = group
    q.with_amax_reduction = True
    copied = q.copy()
    assert copied.amax_reduction_group is group and copied.with_amax_reduction
    copied.nvfp4_quantizer.stochastic_rounding = True
    copied.with_amax_reduction = False
    assert not q.nvfp4_quantizer.stochastic_rounding
    assert q.with_amax_reduction


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_4over6_modules(grouped, mixed):
    test_module_operands(
        "reference",
        grouped,
        torch.bfloat16,
        mixed=mixed,
        nvfp4_options={
            "nvfp4_use_4over6": True,
            "nvfp4_e4m3_max": 448,
            "nvfp4_4over6_err_mode": "MSE",
        },
    )


@pytest.mark.parametrize("backend", ["reference", "fused"])
def test_qdq_explicit_dtype_and_destination(backend):
    """Output overrides must not inherit the quantizer's conflicting default."""
    q = NVFP4QDQQuantizer(dtype=torch.float16, backend=backend)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    out = q.quantize(x, dtype=torch.bfloat16)
    assert q.dtype == torch.float16
    assert_bits(out.dequantize(), native_qdq(x))
    pointer = out._hp_data.data_ptr()
    q.quantize(x * 2, out=out, dtype=torch.bfloat16)
    assert out._hp_data.data_ptr() == pointer
    assert_bits(out.dequantize(), native_qdq(x * 2))
    with pytest.raises(ValueError, match="dtype does not match"):
        q.quantize(x, out=out, dtype=torch.float16)


@pytest.mark.parametrize("empty", [False, True])
def test_qdq_columnwise_only_workspace_dtype(empty):
    """A transient forward QDQ source must round to the requested compute dtype."""
    from transformer_engine.pytorch.quantization import QuantizerRole

    q = mxfp8_nvfp4_qdq_fwd_mxfp8_bwd_factory(
        QuantizerRole(module_type="linear", tensor_type="weight")
    )
    q.set_usage(rowwise=False, columnwise=True)
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    out = (
        q.make_empty(x.shape, dtype=torch.bfloat16, device=x.device)
        if empty
        else q.quantize(x, dtype=torch.bfloat16)
    )
    x.mul_(3)
    q.update_quantized(x, out)
    assert out._rowwise_storage is None
    expected = mx_value(native_qdq(x, torch.bfloat16).float(), columnwise=True)
    # Native make_empty honors its allocation dtype; native quantize retains
    # the FP32 source dtype. Preserve both behaviors and compare equal decodes.
    assert out._columnwise_storage._dtype == (torch.bfloat16 if empty else torch.float32)
    assert_bits(out._columnwise_storage.dequantize(dtype=torch.float32), expected)
