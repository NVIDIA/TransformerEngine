# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention-boundary payload, gradient scaling, and normalization contracts."""

import math
from types import SimpleNamespace

import pytest
import torch
import transformer_engine_torch as tex

from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import DType, LayerNormLinear, MXFP8Quantizer, autocast
from transformer_engine.pytorch.distributed_weight import (
    get_weight_gradient_scale,
    mark_weight_gradient_scale,
)


def assert_payload_equal(actual, expected):
    rows = math.prod(expected.shape[:-1])
    columns = expected.shape[-1]
    for name in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    ):
        before, after = getattr(expected, name), getattr(actual, name)
        assert (before is None) == (after is None)
        if before is None:
            continue
        assert before.shape == after.shape
        before, after = before.view(torch.uint8), after.view(torch.uint8)
        if "scale_inv" in name:
            columnwise = name.startswith("_columnwise")
            outer, inner = (columns, rows // 32) if columnwise else (rows, columns // 32)
            if expected._with_gemm_swizzled_scales:
                i = torch.arange(outer, device=before.device).view(-1, 1)
                j = torch.arange(inner, device=before.device).view(1, -1)
                index = ((i // 128) * math.ceil(inner / 4) + j // 4) * 512
                index = index + (i % 32) * 16 + ((i % 128) // 32) * 4 + j % 4
                before, after = before.flatten()[index], after.flatten()[index]
            elif columnwise:
                before, after = before[:inner, :outer], after[:inner, :outer]
            else:
                before, after = before[:outer, :inner], after[:outer, :inner]
        if not torch.equal(before, after):
            mismatch = (before != after).nonzero()[:8]
            indexes = tuple(mismatch[:, i] for i in range(mismatch.shape[1]))
            pytest.fail(
                f"{name}: positions={mismatch.tolist()}, expected={before[indexes].tolist()},"
                f" actual={after[indexes].tolist()}"
            )


@pytest.fixture
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 requires Blackwell")


@pytest.mark.parametrize("usage", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("swizzle", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "singleton", "fallback", "broadcast"])
@pytest.mark.parametrize(
    "pattern", ["random", "signed_zero", "extreme", "all_nan", "all_inf", "all_zero"]
)
@pytest.mark.usefixtures("require_cuda")
def test_segmented_payload(usage, swizzle, layout, pattern):
    torch.manual_seed(1234)
    rows = 128 if layout != "fallback" else 96
    widths = (256, 128, 128)
    source = torch.randn(rows, sum(widths), device="cuda", dtype=torch.bfloat16)
    if pattern == "signed_zero":
        source[:, ::2] = 0.0
        source[:, 1::2] = -0.0
    elif pattern == "extreme":
        values = torch.tensor(
            [0.0, -0.0, 2**-100, -(2**-100), 2**100, -(2**100), float("inf"), float("nan")],
            device="cuda",
            dtype=torch.bfloat16,
        )
        source.copy_(values.repeat(rows, sum(widths) // values.numel()))
    if pattern in ("all_nan", "all_inf", "all_zero"):
        source.fill_({"all_nan": float("nan"), "all_inf": float("inf"), "all_zero": 0.0}[pattern])
    parts = torch.split(source, widths, dim=-1)
    if layout == "contiguous":
        parts = tuple(p.contiguous() for p in parts)
    elif layout == "singleton":
        parts = tuple(p[:, None, None, :] for p in parts)
    elif layout == "broadcast":
        parts = tuple(p[:1].expand(rows, -1) for p in parts)
    quantizer = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=usage[0], columnwise=usage[1])
    quantizer.optimize_for_gemm = swizzle
    reference = quantizer(torch.cat(parts, dim=-1))
    actual = quantizer.quantize_segments(parts)
    assert_payload_equal(actual, reference)
    replay = quantizer.quantize_segments(parts)
    assert_payload_equal(replay, actual)


@pytest.mark.parametrize("segments", [1, 3])
@pytest.mark.usefixtures("require_cuda")
def test_segmented_stream_and_graph(segments):
    parts = tuple(
        torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) for _ in range(segments)
    )
    quantizer = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = True
    expected = quantizer(torch.cat(parts, dim=-1))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = quantizer.quantize_segments(parts)
    torch.cuda.current_stream().wait_stream(stream)
    assert_payload_equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = quantizer.quantize_segments(parts)
    graph.replay()
    torch.cuda.synchronize()
    assert_payload_equal(captured, expected)


@pytest.mark.usefixtures("require_cuda")
def test_segmented_rejects_invalid_shapes():
    quantizer = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=True, columnwise=True)
    part = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one or three"):
        quantizer.quantize_segments((part, part))
    with pytest.raises(ValueError, match="output shape"):
        quantizer.quantize_segments((part,), (128, 256))
    with pytest.raises(ValueError, match="row counts"):
        quantizer.quantize_segments((part, part[:96], part))


@pytest.mark.parametrize("usage", [(True, False), (False, True)])
@pytest.mark.usefixtures("require_cuda")
def test_native_segmented_binding_rejects_unqualified_usage(usage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("The direct segmented binding is qualified for SM103")
    import transformer_engine_torch as tex

    part = torch.ones(128, 128, device="cuda", dtype=torch.bfloat16)
    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=usage[0], columnwise=usage[1])
    quantizer.optimize_for_gemm = True
    with pytest.raises(RuntimeError, match="both orientations"):
        tex.quantize_mxfp8_segments((part,), quantizer, tuple(part.shape))


def scaling_inputs(monkeypatch, scale=1.0 / 64):
    """Use a tensor implementer and the real MXFP8 quantizer type."""
    weight = torch.empty(1)
    weight.is_distributed_weight = True
    weight.fuse_wgrad_mean_scaling = True
    weight.get_weight_gradient_scale = lambda: scale
    args = SimpleNamespace(
        fp8=True,
        debug=False,
        use_bias=False,
        backward_override=None,
        activation_dtype=torch.bfloat16,
        grad_weight_quantizer=None,
        cpu_offloading=False,
        ub_bulk_wgrad=False,
        wgrad_store=None,
        input_quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return weight, args


def test_plain_and_existing_weights_keep_unit_scale():
    weight = torch.empty(1)
    assert get_weight_gradient_scale(weight, None) == 1.0
    weight.is_distributed_weight = True
    assert get_weight_gradient_scale(weight, None) == 1.0
    weight.fuse_wgrad_mean_scaling = True
    assert get_weight_gradient_scale(weight, None) == 1.0


def test_opt_in_scale(monkeypatch):
    weight, args = scaling_inputs(monkeypatch)
    assert get_weight_gradient_scale(weight, args) == 1.0 / 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("fp8", False),
        ("debug", True),
        ("use_bias", True),
        ("backward_override", "unfused"),
        ("activation_dtype", torch.float16),
        ("grad_weight_quantizer", object()),
        ("cpu_offloading", True),
        ("ub_bulk_wgrad", True),
        ("input_quantizer", None),
        ("wgrad_store", SimpleNamespace(delay_wgrad_compute=lambda: True)),
    ],
)
def test_unsupported_routes_keep_unit_scale(monkeypatch, field, value):
    weight, args = scaling_inputs(monkeypatch)
    setattr(args, field, value)
    assert get_weight_gradient_scale(weight, args) == 1.0


def test_graph_capture_keeps_unit_scale(monkeypatch):
    weight, args = scaling_inputs(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert get_weight_gradient_scale(weight, args) == 1.0


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_provider_scale_rejected(monkeypatch, scale):
    weight, args = scaling_inputs(monkeypatch, scale)
    with pytest.raises(ValueError, match="finite and positive"):
        get_weight_gradient_scale(weight, args)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_marker_preserves_values_and_rejects_duplicate_scaling(dtype):
    gradient = torch.arange(8, dtype=dtype) / 64
    expected = gradient.clone()
    mark_weight_gradient_scale(gradient, 1.0 / 64)
    assert torch.equal(gradient, expected)
    assert gradient._te_wgrad_scale == 1.0 / 64
    with pytest.raises(ValueError, match="already carries"):
        mark_weight_gradient_scale(gradient, 1.0 / 64)


def test_unsupported_gradient_dtype_rejected():
    with pytest.raises(ValueError, match="FP32 or BF16"):
        mark_weight_gradient_scale(torch.empty(1, dtype=torch.float16), 1.0 / 64)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("consumer", ["linear", "norm", "both"])
@pytest.mark.parametrize("fp8", [False, True])
def test_nonmaterialized_norm_gradient(normalization, consumer, fp8):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if fp8 and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 requires Blackwell")
    torch.manual_seed(1234)
    options = dict(
        in_features=128,
        out_features=256,
        normalization=normalization,
        params_dtype=torch.bfloat16,
        return_layernorm_output=True,
        bias=False,
    )
    reference = LayerNormLinear(**options)
    fused = LayerNormLinear(**options, materialize_norm_output_grads=False)
    fused.load_state_dict(reference.state_dict())
    source = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    linear_upstream = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    norm_upstream = torch.randn_like(source)
    results = []
    for module in (reference, fused):
        inp = source.detach().clone().requires_grad_(True)
        with autocast(enabled=fp8, recipe=MXFP8BlockScaling()):
            linear, norm = module(inp)
        outputs, upstream = [], []
        if consumer in ("linear", "both"):
            outputs.append(linear)
            upstream.append(linear_upstream)
        if consumer in ("norm", "both"):
            outputs.append(norm)
            upstream.append(norm_upstream)
        gradients = torch.autograd.grad(
            outputs, (inp, *module.parameters()), upstream, allow_unused=True
        )
        results.append((linear.detach(), norm.detach(), gradients))
    for before, after in zip(results[0][:2], results[1][:2]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for before, after in zip(results[0][2], results[1][2]):
        assert (before is None) == (after is None)
        if before is not None:
            torch.testing.assert_close(before, after, rtol=0, atol=0)
