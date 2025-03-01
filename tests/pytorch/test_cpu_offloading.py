# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
from contextlib import nullcontext

import transformer_engine.pytorch as te

SIZE = 4096

models = {
    "linear": te.Linear,
    "layernorm_mlp": te.LayerNormMLP,
    "layernorm_linear": te.LayerNormLinear,
}


def _get_input():
    return torch.empty((1, SIZE, SIZE)).cuda()  # input size - 1 * 2048 * 2048 * 4b = 16MB


def _measure_memory_between_forward_and_backward(model_cls, fp8, cpu_offload):
    torch.cuda.empty_cache()
    model = model_cls(SIZE, SIZE, 1)

    input = _get_input()
    if cpu_offload:
        offload_context, sync_function = te.get_cpu_offload_context(enabled=True)
    else:
        offload_context = nullcontext()
        sync_function = lambda x: x

    with te.fp8_autocast(enabled=fp8), offload_context:
        out = model(input)
    out = sync_function(out)
    input.data = torch.Tensor()  # delete data from input
    out.data = torch.Tensor()  # delete data from out
    del input
    del out
    torch.cuda.empty_cache()
    allocated_memory_mb = torch.cuda.memory_allocated() / 1024**2
    del model
    return allocated_memory_mb


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("model_key", models.keys())
def test_cpu_offload(fp8, model_key) -> None:
    model_cls = models[model_key]
    without_offloading = _measure_memory_between_forward_and_backward(model_cls, fp8, False)
    torch.cuda.empty_cache()
    with_offloading = _measure_memory_between_forward_and_backward(model_cls, fp8, True)

    assert without_offloading > 30
    assert with_offloading < 10
