# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gradient parity when the returned normalization output is unused."""

import pytest
import torch

from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import LayerNormLinear, autocast


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
