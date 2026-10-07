# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Delayed weight gradients accumulate for tied parameters and microbatches."""

import pytest
import torch
from torch import nn

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch.module.grouped_linear import is_module_grouped_tensor_path_supported
from transformer_engine.pytorch.ops.basic.grouped_linear import (
    is_op_fuser_grouped_tensor_path_supported,
)
from transformer_engine.pytorch.quantization import is_fp8_block_scaling_available


def _build_two_linears(delay, *, bias=False, device="cuda"):
    """Build two BF16 Linear layers with independent parameters."""
    hidden_size = 16
    return nn.Sequential(
        te.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
            params_dtype=torch.bfloat16,
            device=device,
            delay_wgrad_compute=delay,
            fuse_wgrad_accumulation=False,
        ),
        te.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
            params_dtype=torch.bfloat16,
            device=device,
            delay_wgrad_compute=delay,
            fuse_wgrad_accumulation=False,
        ),
    )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("set_to_none", [False, True])
@pytest.mark.parametrize("microbatches", [1, 2])
@pytest.mark.parametrize("backward_dw_order", [(0, 1), (1, 0)])
def test_tied_linear_gradients(bias, set_to_none, microbatches, backward_dw_order):
    """Tied weight and bias gradients match regular backward after either reset."""
    torch.manual_seed(9)
    delayed = _build_two_linears(True, bias=bias)
    regular = _build_two_linears(False, bias=bias)
    regular.load_state_dict(delayed.state_dict())
    for model in (delayed, regular):
        model[1].weight = model[0].weight
        if bias:
            model[1].bias = model[0].bias
    hidden_size, batch_size = 16, 2
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.0) for model in (delayed, regular)]
    for _ in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
        x = torch.randn(batch_size * microbatches, hidden_size, device="cuda", dtype=torch.bfloat16)
        for model in (delayed, regular):
            for microbatch in x.chunk(microbatches):
                model(microbatch.detach().clone().requires_grad_(True)).float().sum().backward()
        for _ in range(microbatches):
            for index in backward_dw_order:
                delayed[index].backward_dw()
        # Microbatch contributions can accumulate in a different BF16 rounding order.
        tolerance = 1e-2 if microbatches > 1 else 0.0
        torch.testing.assert_close(
            delayed[0].weight.grad, regular[0].weight.grad, rtol=tolerance, atol=tolerance
        )
        assert torch.count_nonzero(delayed[0].weight.grad) > 0
        if bias:
            torch.testing.assert_close(
                delayed[0].bias.grad, regular[0].bias.grad, rtol=tolerance, atol=tolerance
            )
        for optimizer in optimizers:
            optimizer.step()


def test_non_tied_linear_gradients():
    """Independent delayed parameters match regular backward."""
    torch.manual_seed(9)
    delayed, regular = _build_two_linears(True), _build_two_linears(False)
    regular.load_state_dict(delayed.state_dict())
    x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
    for model in (delayed, regular):
        model(x.detach().clone().requires_grad_(True)).float().sum().backward()
    for index in (0, 1):
        delayed[index].backward_dw()
        torch.testing.assert_close(
            delayed[index].weight.grad, regular[index].weight.grad, rtol=0, atol=0
        )


def test_tied_linear_matches_pytorch():
    """A plain PyTorch Linear provides an independent tied-weight reference."""
    torch.manual_seed(9)
    model = _build_two_linears(True)
    model[1].weight = model[0].weight
    reference = nn.Linear(16, 16, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(model[0].weight)
    x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
    model(x.detach().clone().requires_grad_(True)).float().sum().backward()
    for layer in model:
        layer.backward_dw()
    reference_x = x.detach().clone().requires_grad_(True)
    reference(reference(reference_x)).float().sum().backward()
    torch.testing.assert_close(model[0].weight.grad, reference.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize("set_to_none", [False, True])
@pytest.mark.parametrize("microbatches", [1, 2])
@pytest.mark.parametrize("backward_dw_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("fp8_block_scaling", [False, True])
def test_tied_layernorm_mlp_gradients(
    set_to_none, microbatches, backward_dw_order, fp8_block_scaling
):
    """Both FCs accumulate delayed gradients without duplicating unfused bias grads."""
    if fp8_block_scaling:
        supported, reason = is_fp8_block_scaling_available(return_reason=True)
        if not supported:
            pytest.skip(reason)
    torch.manual_seed(9)
    hidden_size, ffn_hidden_size, batch_size = 128, 256, 128
    delayed = nn.Sequential(
        te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=True,
            fuse_wgrad_accumulation=False,
        ),
        te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=True,
            fuse_wgrad_accumulation=False,
        ),
    )
    regular = nn.Sequential(
        te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=False,
            fuse_wgrad_accumulation=False,
        ),
        te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=False,
            fuse_wgrad_accumulation=False,
        ),
    )
    regular.load_state_dict(delayed.state_dict())
    names = ("fc1_weight", "fc2_weight", "fc1_bias", "fc2_bias")
    for model in (delayed, regular):
        for name in names:
            setattr(model[1], name, getattr(model[0], name))
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.0) for model in (delayed, regular)]
    for _ in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
        x = torch.randn(batch_size * microbatches, hidden_size, device="cuda", dtype=torch.bfloat16)
        for model in (delayed, regular):
            for microbatch in x.chunk(microbatches):
                with te.autocast(enabled=fp8_block_scaling, recipe=Float8BlockScaling()):
                    output = model(microbatch.detach().clone().requires_grad_(True))
                output.float().sum().backward()
        for _ in range(microbatches):
            for index in backward_dw_order:
                delayed[index].backward_dw()
        for name in names:
            torch.testing.assert_close(
                getattr(delayed[0], name).grad,
                getattr(regular[0], name).grad,
                rtol=1e-2,
                atol=1e-2,
            )
        assert torch.count_nonzero(delayed[0].fc1_weight.grad) > 0
        assert torch.count_nonzero(delayed[0].fc2_weight.grad) > 0
        for optimizer in optimizers:
            optimizer.step()


@pytest.mark.parametrize("set_to_none", [False, True])
@pytest.mark.parametrize("backward_dw_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("packed", [False, True])
def test_tied_grouped_linear_gradients(set_to_none, backward_dw_order, packed, monkeypatch):
    """Module grouped weights and deferred discrete biases accumulate across two microbatches."""
    dtype = torch.bfloat16
    if packed and not is_module_grouped_tensor_path_supported(None, dtype):
        pytest.skip("packed grouped weights require a supported GPU and cuBLASLt")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1" if packed else "0")
    torch.manual_seed(9)
    num_groups, hidden_size, tokens_per_group, microbatches = 2, 16, 16, 2
    delayed = nn.ModuleList(
        [
            te.GroupedLinear(
                num_groups,
                hidden_size,
                hidden_size,
                bias=not packed,
                params_dtype=dtype,
                fuse_wgrad_accumulation=False,
                use_grouped_tensor=packed,
                delay_wgrad_compute=True,
                single_grouped_weight=packed,
            )
            for _ in range(2)
        ]
    )
    regular = nn.ModuleList(
        [
            te.GroupedLinear(
                num_groups,
                hidden_size,
                hidden_size,
                bias=not packed,
                params_dtype=dtype,
                fuse_wgrad_accumulation=False,
                use_grouped_tensor=packed,
                delay_wgrad_compute=False,
                single_grouped_weight=packed,
            )
            for _ in range(2)
        ]
    )
    regular.load_state_dict(delayed.state_dict())
    names = ("weight",) if packed else ("weight0", "weight1", "bias0", "bias1")
    for model in (delayed, regular):
        for name in names:
            setattr(model[1], name, getattr(model[0], name))
    splits = [tokens_per_group] * num_groups
    if packed:
        splits = torch.tensor(splits, device="cuda", dtype=torch.int64)
    for _ in range(2):
        for model in (delayed, regular):
            model.zero_grad(set_to_none=set_to_none)
        x = torch.randn(
            microbatches * tokens_per_group * num_groups, hidden_size, device="cuda", dtype=dtype
        )
        for model in (delayed, regular):
            for microbatch in x.chunk(microbatches):
                output = model[0](microbatch.detach().clone().requires_grad_(True), splits)
                model[1](output, splits).float().sum().backward()
        for _ in range(microbatches):
            for index in backward_dw_order:
                delayed[index].backward_dw()
        for name in names:
            torch.testing.assert_close(
                getattr(delayed[0], name).grad,
                getattr(regular[0], name).grad,
                rtol=1e-2,
                atol=1e-2,
            )
        assert torch.count_nonzero(getattr(delayed[0], names[0]).grad) > 0


@pytest.mark.parametrize("set_to_none", [False, True])
@pytest.mark.parametrize("backward_dw_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("packed", [False, True])
def test_tied_grouped_linear_ops_gradients(set_to_none, backward_dw_order, packed, monkeypatch):
    """Fusible grouped operations accumulate delayed packed and discrete weight grads."""
    dtype = torch.bfloat16
    if packed and not is_op_fuser_grouped_tensor_path_supported(None, dtype):
        pytest.skip("packed grouped weights require a supported GPU and cuBLASLt")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1" if packed else "0")
    torch.manual_seed(9)
    num_groups, hidden_size, tokens_per_group, microbatches = 2, 16, 16, 2
    delayed = nn.ModuleList(
        [
            te.ops.GroupedLinear(
                num_groups,
                hidden_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                accumulate_into_main_grad=False,
                delay_wgrad_compute=True,
                single_grouped_weight=packed,
            )
            for _ in range(2)
        ]
    )
    regular = nn.ModuleList(
        [
            te.ops.GroupedLinear(
                num_groups,
                hidden_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                accumulate_into_main_grad=False,
                delay_wgrad_compute=False,
                single_grouped_weight=packed,
            )
            for _ in range(2)
        ]
    )
    regular.load_state_dict(delayed.state_dict())
    names = ("weight",) if packed else ("weight0", "weight1")
    for model in (delayed, regular):
        for name in names:
            setattr(model[1], name, getattr(model[0], name))
    splits = torch.tensor([tokens_per_group] * num_groups, device="cuda", dtype=torch.int64)
    for _ in range(2):
        for model in (delayed, regular):
            model.zero_grad(set_to_none=set_to_none)
        x = torch.randn(
            microbatches * tokens_per_group * num_groups, hidden_size, device="cuda", dtype=dtype
        )
        for model in (delayed, regular):
            for microbatch in x.chunk(microbatches):
                output = model[0](microbatch.detach().clone().requires_grad_(True), splits)
                model[1](output, splits).float().sum().backward()
        for _ in range(microbatches):
            for index in backward_dw_order:
                delayed[index].backward_dw()
        for name in names:
            torch.testing.assert_close(
                getattr(delayed[0], name).grad,
                getattr(regular[0], name).grad,
                rtol=1e-2,
                atol=1e-2,
            )
        assert torch.count_nonzero(getattr(delayed[0], names[0]).grad) > 0
