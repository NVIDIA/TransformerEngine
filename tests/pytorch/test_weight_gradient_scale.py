# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Opt-in distributed-weight gradient scaling contract."""

from types import SimpleNamespace

import pytest
import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.distributed_weight import (
    get_weight_gradient_scale,
    mark_weight_gradient_scale,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


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
