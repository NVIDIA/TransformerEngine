# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
from contextlib import nullcontext

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling


@pytest.mark.skipif(torch.__version__ < "2", reason="torch.compile not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TE Linear")
@pytest.mark.parametrize(
    "use_fp8,with_backward",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
    ids=["fp16_fwd", "fp16_fwd_bwd", "fp8_fwd", "fp8_fwd_bwd"],
)
def test_te_linear_fullgraph_compile(use_fp8, with_backward):
    if use_fp8:
        fp8_available, reason = te.is_fp8_available(return_reason=True)
        if not fp8_available:
            pytest.skip(reason)

    model = te.Linear(128, 64, device="cuda").to(dtype=torch.bfloat16)
    for param in model.parameters():
        param.requires_grad_(False)
    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=with_backward)

    fp8_recipe = DelayedScaling() if use_fp8 else None
    maybe_fp8 = te.autocast(enabled=True, recipe=fp8_recipe) if use_fp8 else nullcontext()

    with maybe_fp8:
        if use_fp8:
            model.init_fp8_metadata()
        compiled_model = torch.compile(model, fullgraph=True)
        out = compiled_model(x)
        assert out.shape == (16, 64)
        if with_backward:
            out.sum().backward()

    if with_backward:
        assert x.grad is not None
