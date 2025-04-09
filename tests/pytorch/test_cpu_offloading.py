# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
from contextlib import nullcontext

import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

# Check if FP8 supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

SIZE = 512

models = {
    "linear": te.Linear,
    "layernorm_mlp": te.LayerNormMLP,
    "layernorm_linear": te.LayerNormLinear,
}


def _get_input():
    return torch.empty((128, SIZE, SIZE)).cuda()


def _measure_memory_between_forward_and_backward(model_cls, fp8, cpu_offload, weight_offload=False, activation_offload=True):

    with te.cpu_model_init(enabled=weight_offload):
        input_layer = model_cls(SIZE, SIZE)
        hidden_layer = model_cls(SIZE, SIZE)
        output_layer = model_cls(SIZE, SIZE)

    input = _get_input()
    if cpu_offload:
        offload_context, sync_function = te.get_cpu_offload_context(
            enabled=True,
            num_layers=2,
            model_layers=3,
            offload_activations=activation_offload,
            offload_weights=weight_offload,
        )
    else:
        offload_context = nullcontext()
        sync_function = lambda x: x

    with te.fp8_autocast(enabled=fp8), offload_context:
        out = input_layer(input)
    out = sync_function(out)
    with te.fp8_autocast(enabled=fp8), offload_context:
        out = hidden_layer(out)
    out = sync_function(out)
    with te.fp8_autocast(enabled=fp8), offload_context:
        out = output_layer(out)
    out = sync_function(out)

    max_mem_used = torch.cuda.memory_allocated() / 1024**2

    out.sum().backward()

    del input_layer
    del hidden_layer
    del output_layer
    del input
    del out

    torch.cuda.synchronize()

    return max_mem_used


@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("model_key", models.keys())
def test_cpu_offload(fp8, model_key) -> None:

    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)

    model_cls = models[model_key]

    without_offloading = _measure_memory_between_forward_and_backward(model_cls, fp8, False)

    with_offloading = _measure_memory_between_forward_and_backward(model_cls, fp8, True)

    assert with_offloading < without_offloading


@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("model_key", models.keys())
def test_cpu_weight_offload(fp8, model_key) -> None:
    model_cls = models[model_key]

    without_offloading = _measure_memory_between_forward_and_backward(model_cls, fp8, False, weight_offload=False, activation_offload=False)

    with_offloading = _measure_memory_between_forward_and_backward(model_cls, fp8, True, weight_offload=True, activation_offload=False)

    print(f"without_offloading: {without_offloading}, with_offloading: {with_offloading}")

    assert with_offloading < without_offloading