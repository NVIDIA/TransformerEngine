# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression coverage for delayed-scaling cast/transpose ordering on SM120."""

import math

import pytest
import torch
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format


class _StressBlock(torch.nn.Module):
    def __init__(self, width: int, kv_width: int, mlp_width: int) -> None:
        super().__init__()

        def linear(inp: int, out: int) -> te.Linear:
            return te.Linear(
                inp,
                out,
                bias=False,
                params_dtype=torch.bfloat16,
            )

        self.norm = torch.nn.RMSNorm(width, dtype=torch.bfloat16, device="cuda")
        self.q_proj = linear(width, width)
        self.k_proj = linear(width, kv_width)
        self.v_proj = linear(width, kv_width)
        self.o_proj = linear(width, width)
        self.gate_proj = linear(width, mlp_width)
        self.up_proj = linear(width, mlp_width)
        self.down_proj = linear(mlp_width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(value)
        q_value = self.q_proj(hidden)
        kv_value = torch.cat((self.k_proj(hidden), self.v_proj(hidden)), dim=-1)
        attention = self.o_proj(torch.tanh(q_value + 0.125 * kv_value))
        gate = F.silu(self.gate_proj(hidden))
        mlp = self.down_proj(gate * self.up_proj(hidden))
        return value + 0.03125 * (attention + mlp)


class _StressNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_StressBlock(2048, 1024, 4096) for _ in range(14)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = te.checkpoint(layer, value, use_reentrant=True)
        return value


def _grad_metrics(model: torch.nn.Module) -> tuple[float, float, bool]:
    squared_norm = torch.zeros((), dtype=torch.float32, device="cuda")
    max_abs = torch.zeros((), dtype=torch.float32, device="cuda")
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        squared_norm += grad.float().square().sum()
        max_abs = torch.maximum(max_abs, grad.abs().max().float())
        finite = finite and bool(torch.isfinite(grad).all())
    return float(squared_norm.sqrt()), float(max_abs), finite


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12,
    reason="The regression requires an SM120 GPU",
)
@pytest.mark.parametrize("seed", [20260715, 20260716, 20260717])
def test_delayed_scaling_checkpointed_dgrad_ordering(seed: int) -> None:
    """Delayed-scaling dgrad must remain in the BF16-scale numerical range."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _StressNetwork().cuda()
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=1024)
    generator = torch.Generator(device="cuda")

    final_norm = 0.0
    for step in range(3):
        model.zero_grad(set_to_none=True)
        for microbatch in range(64):
            generator.manual_seed(seed + step * 100_003 + microbatch * 997)
            value = torch.randn(
                1,
                2048,
                2048,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
                requires_grad=True,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(
                enabled=True,
                recipe=recipe,
            ):
                loss = model(value).float().square().mean() / 64
            loss.backward()

            final_norm, parameter_max, finite = _grad_metrics(model)
            input_max = float(value.grad.detach().abs().max())
            assert math.isfinite(float(loss.detach()))
            assert finite
            assert math.isfinite(input_max)
            assert final_norm < 0.1, (
                f"gradient corruption at step={step}, microbatch={microbatch}: "
                f"grad_norm={final_norm}, parameter_max={parameter_max}, "
                f"input_max={input_max}"
            )
            assert input_max < 1e-4

    assert 0.001 < final_norm < 0.05
