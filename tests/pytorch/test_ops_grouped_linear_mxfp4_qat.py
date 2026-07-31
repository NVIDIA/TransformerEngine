# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4-QAT coverage for the fusible ops.GroupedLinear."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops.basic.grouped_linear as grouped_linear_impl
from transformer_engine.common.recipe import (
    MXFP4QATFloat8BlockScaling,
    MXFP4QATMXFP8BlockScaling,
)
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.mxfp4_qat import mxfp4_fake_quantize


def _skip_if_recipe_unavailable(recipe) -> None:
    if recipe.mxfp8():
        available, reason = te.is_mxfp8_available(return_reason=True)
    else:
        available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        pytest.skip(reason)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.to(torch.float64)
    expected = expected.to(torch.float64)
    return ((actual - expected).norm() / expected.norm().clamp(min=1e-30)).item()


@pytest.mark.parametrize(
    "recipe_cls",
    (MXFP4QATMXFP8BlockScaling, MXFP4QATFloat8BlockScaling),
)
@pytest.mark.parametrize("path", ("graph_safe", "legacy"))
def test_ops_grouped_linear_qat_none_forward_saved_weight_and_ste(
    recipe_cls,
    path,
    monkeypatch,
):
    """Project each expert once, save host-FP8 for dgrad, and return wgrad to the master."""
    recipe = recipe_cls(backward_override=None)
    _skip_if_recipe_unavailable(recipe)
    if path == "legacy":
        monkeypatch.setattr(
            grouped_linear_impl.GroupedLinear,
            "_is_graph_safe_path_supported",
            staticmethod(lambda **kwargs: False),
        )
    else:
        original_path_check = grouped_linear_impl.GroupedLinear._is_graph_safe_path_supported

        def require_graph_safe_path(**kwargs):
            supported = original_path_check(**kwargs)
            if not supported:
                pytest.skip("graph-safe grouped-tensor path is unavailable")
            return True

        monkeypatch.setattr(
            grouped_linear_impl.GroupedLinear,
            "_is_graph_safe_path_supported",
            staticmethod(require_graph_safe_path),
        )

    projection_calls = 0

    def tracked_fake_quantize(weight):
        nonlocal projection_calls
        projection_calls += 1
        return mxfp4_fake_quantize(weight)

    monkeypatch.setattr(grouped_linear_impl, "mxfp4_fake_quantize", tracked_fake_quantize)

    torch.manual_seed(1234)
    num_groups, in_features, out_features = 2, 256, 128
    split_sizes = torch.tensor([128, 128], dtype=torch.int64, device="cuda")
    op = te.ops.GroupedLinear(
        num_groups,
        in_features,
        out_features,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        for idx in range(num_groups):
            weight = getattr(op, f"weight{idx}")
            weight.copy_((0.05 * torch.randn_like(weight, dtype=torch.float32)).to(weight.dtype))
    projected_weights = [
        mxfp4_fake_quantize(getattr(op, f"weight{idx}").detach()).to(torch.float32)
        for idx in range(num_groups)
    ]

    x = torch.randn(256, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn(256, out_features, device="cuda", dtype=torch.bfloat16)
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        output = op(x, split_sizes)
    output.backward(grad_output)

    # The forward host-QTensor is saved for mode None, so backward must not replay QAT.
    assert projection_calls == num_groups

    x_splits = torch.split(x.detach().to(torch.float32), split_sizes.tolist())
    dy_splits = torch.split(grad_output.to(torch.float32), split_sizes.tolist())
    expected_output = torch.cat(
        [inp @ weight.t() for inp, weight in zip(x_splits, projected_weights)]
    )
    expected_dgrad = torch.cat([dy @ weight for dy, weight in zip(dy_splits, projected_weights)])
    assert _relative_error(output, expected_output) < 0.05
    assert _relative_error(x.grad, expected_dgrad) < 0.05

    for idx, (inp, dy) in enumerate(zip(x_splits, dy_splits)):
        weight = getattr(op, f"weight{idx}")
        assert weight.grad is not None
        expected_wgrad = dy.t() @ inp
        assert _relative_error(weight.grad, expected_wgrad) < 0.07


@pytest.mark.parametrize(
    "recipe_cls",
    (MXFP4QATMXFP8BlockScaling, MXFP4QATFloat8BlockScaling),
)
@pytest.mark.parametrize("backward_override", ("dequantized", "high_precision"))
def test_ops_grouped_linear_qat_rejects_unsupported_backward_overrides(
    recipe_cls,
    backward_override,
):
    """QAT support must not exceed the pure GroupedLinear override range."""
    recipe = recipe_cls(backward_override=backward_override)
    _skip_if_recipe_unavailable(recipe)
    op = te.ops.GroupedLinear(
        2,
        256,
        128,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    split_sizes = torch.tensor([128, 128], dtype=torch.int64, device="cuda")
    with pytest.raises(NotImplementedError, match="only supports backward_override=None"):
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            op(x, split_sizes)


@pytest.mark.parametrize(
    "recipe_cls",
    (MXFP4QATMXFP8BlockScaling, MXFP4QATFloat8BlockScaling),
)
def test_ops_grouped_linear_qat_rejects_primary_quantized_weights(recipe_cls):
    """A packed primary weight cannot provide the master required by QAT."""
    recipe = recipe_cls(backward_override=None)
    _skip_if_recipe_unavailable(recipe)
    with pytest.raises(NotImplementedError, match="high-precision master is required"):
        with te.quantized_model_init(enabled=True, recipe=recipe):
            te.ops.GroupedLinear(
                2,
                256,
                128,
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
            )
