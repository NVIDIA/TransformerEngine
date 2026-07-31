# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from types import SimpleNamespace

import pytest
import torch

from transformer_engine.pytorch.module import _common
from transformer_engine.pytorch.module import grouped_linear


@pytest.mark.parametrize(
    ("recipe", "metadata_name", "expected_value"),
    (
        ("fp8_current_scaling", "scale_inv", 0.25),
        ("fp8_delayed_scaling", "amax", 448.0),
        ("nvfp4", "amax", 2688.0),
        ("nvfp4_rowwise", "amax_rowwise", 1344.0),
    ),
)
def test_scale_buffer_info_selects_recipe_metadata(
    monkeypatch, recipe, metadata_name, expected_value
):
    monkeypatch.setattr(_common, "get_quantization_recipe_name", lambda _: recipe)
    tensor = SimpleNamespace(
        _scale_inv=torch.tensor([0.25], dtype=torch.float32),
        _amax_rowwise=torch.tensor([2688.0 if recipe == "nvfp4" else 1344.0], dtype=torch.float32),
    )
    quantizer = SimpleNamespace(amax=torch.tensor([448.0], dtype=torch.float32))

    buffer_name, value = _common._get_scale_buffer_info("input", tensor, quantizer)

    assert buffer_name == f"input_tensor_{metadata_name}_{recipe}_te_ptq_calibrated"
    torch.testing.assert_close(value, torch.tensor([expected_value]))


@pytest.mark.parametrize("recipe", ("mxfp8", "fp8_block_scaling"))
def test_scale_buffer_info_skips_non_global_scaling_recipes(monkeypatch, recipe):
    monkeypatch.setattr(_common, "get_quantization_recipe_name", lambda _: recipe)
    tensor = SimpleNamespace(_rowwise_scale_inv=torch.ones(2, 2))

    assert _common._get_scale_buffer_info("input", tensor, object()) is None


def test_grouped_scale_buffers_are_per_gemm(monkeypatch):
    monkeypatch.setattr(_common, "get_quantization_recipe_name", lambda _: "fp8_current_scaling")
    inputs = [
        SimpleNamespace(_scale_inv=torch.tensor([0.25])),
        SimpleNamespace(_scale_inv=torch.tensor([0.5])),
    ]
    weights = [
        SimpleNamespace(_scale_inv=torch.tensor([0.75])),
        SimpleNamespace(_scale_inv=torch.tensor([1.0])),
    ]
    scale_buffers = {}

    grouped_linear._update_grouped_scale_buffers(
        scale_buffers,
        inputs,
        weights,
        object(),
        object(),
        activation_scale_decay=0.0,
    )

    assert set(scale_buffers) == {
        "input_gemm0_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated",
        "input_gemm1_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated",
        "weight_gemm0_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated",
        "weight_gemm1_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated",
    }
    torch.testing.assert_close(
        scale_buffers["input_gemm1_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated"],
        torch.tensor([0.5]),
    )


@pytest.mark.parametrize(
    ("observed_scale", "expected_scale"),
    (
        # Decayed max is greater than the observed.
        (1.0, 2.0),
        # Decayed max is less than the observed.
        (3.0, 3.0),
    ),
)
def test_activation_scale_buffer_uses_decaying_maximum(observed_scale, expected_scale):
    name = "fc1_input_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated"
    scale_buffers = {name: torch.tensor([4.0])}

    _common._update_scale_buffers(
        scale_buffers,
        {name: torch.tensor([observed_scale])},
        activation_scale_decay=0.5,
    )
    torch.testing.assert_close(scale_buffers[name], torch.tensor([expected_scale]))
