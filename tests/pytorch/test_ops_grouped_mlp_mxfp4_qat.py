# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4-QAT coverage for the fused ops grouped-MLP path."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops.fused.grouped_mlp as grouped_mlp_impl
from transformer_engine.common.recipe import (
    MXFP4QATMXFP8BlockScaling,
    MXFP8BlockScaling,
)
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.mxfp4_qat import mxfp4_fake_quantize


def _require_fused_mxfp8_grouped_mlp() -> None:
    available, reason = te.is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    if not te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported():
        pytest.skip("CuTe DSL fused MXFP8 grouped-MLP is not available")


def _make_grouped_mlp():
    fc1 = te.ops.GroupedLinear(
        2,
        128,
        256,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        single_grouped_weight=False,
    )
    activation = te.ops.ScaledSwiGLU(glu_interleave_size=32)
    fc2 = te.ops.GroupedLinear(
        2,
        128,
        128,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        single_grouped_weight=False,
    )
    return te.ops.Sequential(fc1, activation, fc2), fc1, fc2


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.to(torch.float64)
    expected = expected.to(torch.float64)
    return ((actual - expected).norm() / expected.norm().clamp(min=1e-30)).item()


def test_fused_grouped_mlp_qat_none_matches_projected_pure_mxfp8(monkeypatch):
    """QAT None projects once in forward, saves host FP8, and returns master wgrads."""
    _require_fused_mxfp8_grouped_mlp()

    torch.manual_seed(1234)
    qat_module, qat_fc1, qat_fc2 = _make_grouped_mlp()
    ref_module, ref_fc1, ref_fc2 = _make_grouped_mlp()

    qat_weights = []
    ref_weights = []
    with torch.no_grad():
        for qat_fc, ref_fc in ((qat_fc1, ref_fc1), (qat_fc2, ref_fc2)):
            for group_idx in range(2):
                qat_weight = getattr(qat_fc, f"weight{group_idx}")
                ref_weight = getattr(ref_fc, f"weight{group_idx}")
                qat_weight.copy_(
                    (0.05 * torch.randn_like(qat_weight, dtype=torch.float32)).to(qat_weight.dtype)
                )
                ref_weight.copy_(mxfp4_fake_quantize(qat_weight.detach()))
                qat_weights.append(qat_weight)
                ref_weights.append(ref_weight)

    projection_calls = 0
    real_fake_quantize = grouped_mlp_impl.mxfp4_fake_quantize

    def tracked_fake_quantize(weight):
        nonlocal projection_calls
        projection_calls += 1
        return real_fake_quantize(weight)

    monkeypatch.setattr(grouped_mlp_impl, "mxfp4_fake_quantize", tracked_fake_quantize)

    split_sizes = torch.tensor([128, 128], dtype=torch.int64, device="cuda")
    x_qat = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x_qat.detach().clone().requires_grad_(True)
    probs_qat = torch.rand(256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs_ref = probs_qat.detach().clone().requires_grad_(True)

    with fp8_autocast(
        enabled=True,
        fp8_recipe=MXFP4QATMXFP8BlockScaling(backward_override=None),
    ):
        out_qat = qat_module(x_qat, split_sizes, probs_qat, split_sizes)
    with fp8_autocast(
        enabled=True,
        fp8_recipe=MXFP8BlockScaling(backward_override=None),
    ):
        out_ref = ref_module(x_ref, split_sizes, probs_ref, split_sizes)

    qat_fused_op = qat_module._module_groups[0]._forward_ops[0][0]
    ref_fused_op = ref_module._module_groups[0]._forward_ops[0][0]
    assert isinstance(qat_fused_op, te.ops.fused.GroupedMLP_CuTeGEMMGLU)
    assert isinstance(ref_fused_op, te.ops.fused.GroupedMLP_CuTeGEMMGLU)
    assert projection_calls == 4
    assert _relative_error(out_qat, out_ref) < 1e-6

    grad_output = torch.randn_like(out_qat)
    out_qat.backward(grad_output)
    out_ref.backward(grad_output)

    # The fused None path must reuse the forward host-QTensor, not replay QAT.
    assert projection_calls == 4
    assert _relative_error(x_qat.grad, x_ref.grad) < 1e-6
    assert _relative_error(probs_qat.grad, probs_ref.grad) < 1e-6
    for qat_weight, ref_weight in zip(qat_weights, ref_weights):
        assert qat_weight.grad is not None
        assert ref_weight.grad is not None
        assert _relative_error(qat_weight.grad, ref_weight.grad) < 1e-6


@pytest.mark.parametrize("backward_override", ("dequantized", "high_precision"))
def test_fused_grouped_mlp_qat_rejects_unsupported_overrides(backward_override):
    """Do not extend QAT beyond the pure fused grouped-MLP override range."""
    _require_fused_mxfp8_grouped_mlp()
    module, _, _ = _make_grouped_mlp()
    split_sizes = torch.tensor([128, 128], dtype=torch.int64, device="cuda")
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(256, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="fused grouped-MLP ops support only"):
        with fp8_autocast(
            enabled=True,
            fp8_recipe=MXFP4QATMXFP8BlockScaling(
                backward_override=backward_override,
            ),
        ):
            module(x, split_sizes, probs, split_sizes)
