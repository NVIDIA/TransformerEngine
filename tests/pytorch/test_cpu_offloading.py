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
NUM_HEADS = 8
NUM_LAYERS = 5

models = {
    "linear": lambda: te.Linear(SIZE, SIZE),
    "layernorm_mlp": lambda: te.LayerNormMLP(SIZE, SIZE),
    "layernorm_linear": lambda: te.LayerNormLinear(SIZE, SIZE)
}

def _get_input():
    return torch.empty((128, SIZE, SIZE)).cuda()


def _measure_memory_between_forward_and_backward(models, fp8, cpu_offload):
    tensor = _get_input()
    if cpu_offload:
        offload_context, sync_function = te.get_cpu_offload_context(
            enabled=True,
            num_layers=len(models) - 1,
            model_layers=len(models),
            offload_activations=True,
            offload_weights=False,
        )
    else:
        offload_context = nullcontext()
        sync_function = lambda x: x

    for model in models:
        with te.fp8_autocast(enabled=fp8), offload_context:
            tensor = model(tensor)
        tensor = sync_function(tensor)

    max_mem_used = torch.cuda.memory_allocated() / 1024**2
    torch.cuda.synchronize()

    return max_mem_used


@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("model_key", models.keys())
def test_cpu_offload(fp8, model_key) -> None:
    # We run three configurations:
    # - no offloading - all activations should be on GPU between forward and backward
    # - no offloading - one layer - only the first layer's activations should be on GPU between forward and backward
    # - with offloading - all layers - only the last layer's activations should be on GPU between forward and backward,
    #   all other layers should be offloaded to CPU.


    model_cls = models[model_key]
    models_list = [model_cls() for _ in range(NUM_LAYERS)]

    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)

    without_offloading = _measure_memory_between_forward_and_backward(models_list, fp8, False)
    without_offloading_one_layer = _measure_memory_between_forward_and_backward(models_list[:1], fp8, False)
    with_offloading = _measure_memory_between_forward_and_backward(models_list, fp8, True)

    assert with_offloading < without_offloading
    # 10 is upper bound of size of weight transposes which will be precomputed during forward pass,
    # all other tensors should be offloaded to CPU.
    assert abs(with_offloading - without_offloading_one_layer) < 10
