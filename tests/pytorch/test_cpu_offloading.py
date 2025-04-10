# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
from contextlib import nullcontext
import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

# Check if FP8 supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 5
EPSILON = 0.1

# Flash attention saves some internal tensor for the backward,
# that cannot be offloaded to CPU.
assert os.getenv("NVTE_FLASH_ATTN") == "0"

# Offloading is supported for attention only for fused and flash attention backends,
# so use of bfloat16 is required.
#
# For transformer layer activation offloading with dropout is not supported,
# so we set hidden_dropout to 0.0.
model_types = {
    "linear":
        lambda: te.Linear(SIZE, SIZE, params_dtype=torch.bfloat16),
    "layernorm_mlp":
        lambda: te.LayerNormMLP(SIZE, SIZE, params_dtype=torch.bfloat16),
    "layernorm_linear":
        lambda: te.LayerNormLinear(SIZE, SIZE, params_dtype=torch.bfloat16),
    "multihead_attention":
        lambda: te.MultiheadAttention(SIZE, NUM_HEADS, params_dtype=torch.bfloat16),
    "transformer_layer":
        lambda: te.TransformerLayer(
            SIZE, SIZE, NUM_HEADS, params_dtype=torch.bfloat16, hidden_dropout=0.0),
}


def _get_input():
    return torch.empty((128, SIZE, SIZE), dtype=torch.bfloat16).cuda()


def _get_fp8_weight_cache_size(models, fp8):
    if not fp8:
        return 0
    num_of_params = 0
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                num_of_params += param.numel()
    # one byte for columnwise and one byte for rowwise,
    # thus multiply by 2
    return 2 * num_of_params // 1024**2


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
            tensor = model(tensor, is_first_microbatch=True)
        tensor = sync_function(tensor)

    max_mem_used = torch.cuda.memory_allocated() / 1024**2
    torch.cuda.synchronize()

    return max_mem_used


@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("model_key", model_types.keys())
def test_cpu_offload(fp8, model_key) -> None:
    # We run three configurations:
    # - no offloading - all activations should be on GPU between forward and backward
    # - no offloading - one layer - only the first layer's activations should be on GPU between forward and backward
    # - with offloading - all layers - only the last layer's activations should be on GPU between forward and backward,
    #   all other layers should be offloaded to CPU.

    model_cls = model_types[model_key]
    models_list = [model_cls() for _ in range(NUM_LAYERS)]

    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)

    without_offloading = _measure_memory_between_forward_and_backward(models_list, fp8, False)
    without_offloading_one_layer = _measure_memory_between_forward_and_backward(models_list[:1], fp8, False)
    with_offloading = _measure_memory_between_forward_and_backward(models_list, fp8, True)

    assert with_offloading < without_offloading
    # The only difference between with_offloading memory consumption and
    # without_offloading_one_layer memory consumption should be
    # the size of weights after fp8 cast.

    memory_consumption_diff = abs(with_offloading - without_offloading_one_layer)
    assert memory_consumption_diff < _get_fp8_weight_cache_size(models_list, fp8) + EPSILON
