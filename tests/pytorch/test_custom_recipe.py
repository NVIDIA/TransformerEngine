# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import check_fp8_support, fp8_autocast
from transformer_engine.pytorch import Linear
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.module.layernorm_linear import LayerNormLinear
from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
)
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear


@pytest.mark.parametrize("module_type", ["Linear", "LayerNormLinear", "OpsLinear", "LayerNormMLP"])
def test_custom_recipe_sanity(module_type):
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(0)

    # Simple linear layer with dims divisible by 16
    in_features = 64
    out_features = 64
    batch = 32

    if module_type == "Linear":
        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    elif module_type == "LayerNormLinear":
        model = LayerNormLinear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    elif module_type == "LayerNormMLP":
        # hidden_size == in_features == out_features for simplicity
        model = LayerNormMLP(hidden_size=in_features, ffn_hidden_size=out_features, params_dtype=torch.bfloat16).cuda()
    else:
        # OpsLinear path
        model = te_ops.Linear(in_features, out_features, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Provide factories for input, weight, and grad_output quantizers
    def make_inp():
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    def make_w():
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    def make_gout():
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")

    qf = recipe.CustomQuantizerFactories(
        input_factory=make_inp,
        weight_factory=make_w,
        grad_output_factory=make_gout,
    )

    custom_recipe = recipe.CustomRecipe(qfactories=qf)

    # Execute with custom recipe
    with fp8_autocast(enabled=True, fp8_recipe=custom_recipe):
        out = model(inp)
    loss = out.float().sum()
    loss.backward()

    # Basic sanity: gradients exist
    assert inp.grad is not None


def test_custom_recipe_grouped_linear_sanity():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(0)

    num_gemms = 3
    in_features = 64
    out_features = 64
    batch = 32
    base = batch // num_gemms
    rem = batch % num_gemms
    m_splits = [base + (1 if i < rem else 0) for i in range(num_gemms)]

    model = GroupedLinear(num_gemms, in_features, out_features, params_dtype=torch.bfloat16).cuda()
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def make_inp():
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    def make_w():
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    def make_gout():
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")

    qf = recipe.CustomQuantizerFactories(
        input_factory=make_inp,
        weight_factory=make_w,
        grad_output_factory=make_gout,
    )
    custom_recipe = recipe.CustomRecipe(qfactories=qf)

    with fp8_autocast(enabled=True, fp8_recipe=custom_recipe):
        out = model(inp, m_splits)
    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None


def test_custom_recipe_matches_current_scaling():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(123)

    in_features = 64
    out_features = 64
    batch = 32

    # Create two identical models
    model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    model_custom = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    model_custom.load_state_dict(model_ref.state_dict())

    # Identical inputs for both paths
    base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
    inp_ref = base_inp.clone().detach().requires_grad_(True)
    inp_custom = base_inp.clone().detach().requires_grad_(True)

    # Reference: use Float8CurrentScaling recipe
    ref_recipe = recipe.Float8CurrentScaling()
    with fp8_autocast(enabled=True, fp8_recipe=ref_recipe):
        out_ref = model_ref(inp_ref)
    loss_ref = out_ref.float().sum()
    loss_ref.backward()

    # Custom: use factories for Float8CurrentScalingQuantizers (only input, weight, grad_output)
    qf = recipe.CustomQuantizerFactories(
        input_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        weight_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        grad_output_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda"),
    )
    custom_recipe = recipe.CustomRecipe(qfactories=qf)

    with fp8_autocast(enabled=True, fp8_recipe=custom_recipe):
        out_custom = model_custom(inp_custom)
    loss_custom = out_custom.float().sum()
    loss_custom.backward()

    # Compare forward outputs (exact match expected)
    assert torch.allclose(out_ref, out_custom, rtol=0.0, atol=0.0)

    # Compare input gradients
    assert inp_ref.grad is not None and inp_custom.grad is not None
    assert torch.allclose(inp_ref.grad, inp_custom.grad, rtol=0.0, atol=0.0)

    # Compare parameter gradients (weights and bias if present)
    ref_params = dict(model_ref.named_parameters())
    custom_params = dict(model_custom.named_parameters())
    for name, p_ref in ref_params.items():
        p_cus = custom_params[name]
        # Parameter tensors should be identical initially; now check grads
        if p_ref.grad is None and p_cus.grad is None:
            continue
        assert p_ref.grad is not None and p_cus.grad is not None
        assert torch.allclose(p_ref.grad, p_cus.grad, rtol=0.0, atol=0.0)


def test_custom_recipe_ops_linear_2_1_layout():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(7)

    in_features = 64
    out_features = 64
    batch = 16

    # Use ops.Linear which consumes 2 forward quantizers and 1 backward quantizer
    op = te_ops.Linear(in_features, out_features, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    qf = recipe.CustomQuantizerFactories(
        input_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        weight_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda"),
        grad_output_factory=lambda: Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda"),
    )
    custom = recipe.CustomRecipe(qfactories=qf)

    with fp8_autocast(enabled=True, fp8_recipe=custom):
        out = op(inp)
    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None


def test_custom_recipe_factory_invocation_counts_and_cycling():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(13)

    in_features = 64
    out_features = 64
    batch = 8

    op = Linear(in_features, out_features, params_dtype=torch.bfloat16)
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Counters
    counts = {"input": 0, "grad_output": 0}

    def make_inp():
        counts["input"] += 1
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=torch.device("cuda"))

    def make_gout():
        counts["grad_output"] += 1
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device=torch.device("cuda"))

    # Only input (forward) and grad_output (backward) factories provided; weight/output/grad_input are None
    qf = recipe.CustomQuantizerFactories(
        input_factory=make_inp,
        weight_factory=None,  # deliberately missing to exercise reuse
        output_factory=None,
        grad_output_factory=make_gout,
        grad_input_factory=None,
    )
    custom = recipe.CustomRecipe(qfactories=qf)

    # Run fwd+bwd once; for a single GEMM, expect forward to build 3 quantizers (cycled from 1 factory),
    # and backward to build 2 quantizers (cycled from 1 factory).
    with fp8_autocast(enabled=True, fp8_recipe=custom):
        out = op(inp)
    loss = out.float().sum()
    loss.backward()

    assert counts["input"] == 3  # input factory used for input, weight, output via cycling
    assert counts["grad_output"] == 2  # grad_output factory used for grad_output and grad_input via cycling


def test_factories_return_distinct_instances_and_buffers():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    # Two calls should produce distinct quantizer objects and distinct tensor buffers
    def factory():
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=torch.device("cuda"))

    q1 = factory()
    q2 = factory()

    assert q1 is not q2
    assert q1.scale.data_ptr() != q2.scale.data_ptr()
    assert q1.amax.data_ptr() != q2.amax.data_ptr()

    # Mutating one should not affect the other
    q1.scale.fill_(123.0)
    assert not torch.equal(q1.scale, q2.scale)
