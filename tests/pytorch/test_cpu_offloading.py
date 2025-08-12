# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
from contextlib import nullcontext
import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from utils import ModelConfig, get_available_attention_backends

# Check if FP8 is supported
fp8_available, _ = FP8GlobalStateManager.is_fp8_available()

fp8_recipes = [None]
if fp8_available:
    fp8_recipes.append(recipe.Float8CurrentScaling())
    fp8_recipes.append(recipe.DelayedScaling())

model_config = {
    "small": ModelConfig(8, 512, 8, 64, num_layers=5, eps=0.1),
}
SIZE = model_config["small"].hidden_size
NUM_HEADS = model_config["small"].num_heads
NUM_LAYERS = model_config["small"].num_layers
EPSILON = model_config["small"].eps

# Flash attention saves some internal tensor for the backward pass
# that cannot be offloaded to CPU.
assert os.getenv("NVTE_FLASH_ATTN") == "0"

# Offloading is supported for attention only for fused and flash attention backends,
# so the use of bfloat16 is required.
#
# For the TransformerLayer, activation offloading with dropout is not supported,
# so we set hidden_dropout to 0.0.
model_types = {
    "linear": lambda: te.Linear(SIZE, SIZE, params_dtype=torch.bfloat16),
    "layernorm_mlp": lambda: te.LayerNormMLP(SIZE, SIZE, params_dtype=torch.bfloat16),
    "layernorm_linear": lambda: te.LayerNormLinear(SIZE, SIZE, params_dtype=torch.bfloat16),
    "multihead_attention": lambda: te.MultiheadAttention(
        SIZE, NUM_HEADS, params_dtype=torch.bfloat16
    ),
    "transformer_layer": lambda: te.TransformerLayer(
        SIZE, SIZE, NUM_HEADS, params_dtype=torch.bfloat16, hidden_dropout=0.0
    ),
}


def _get_input():
    return torch.empty((128, SIZE, SIZE), dtype=torch.bfloat16).cuda()


def _get_fp8_weight_cache_size(models, fp8_recipe):
    """
    Calculate the total FP8 weight cache size (in MB) for a list of models.
    """
    if fp8_recipe is None:
        return 0

    params_bytes = 0
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                params_bytes += param.numel()

    # One byte for columnwise and one byte for rowwise,
    # hence multiply by 2 and convert to MB
    # there is 1 byte of scale per 32 elements in mxFP8
    factor_for_scale_inv_tensor = (1 + 1 / 32) if fp8_recipe.mxfp8() else 1
    return (2 * params_bytes * factor_for_scale_inv_tensor) / (1024**2)


def _measure_memory_between_forward_and_backward(models, fp8_recipe, cpu_offload):
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
        with te.fp8_autocast(
            enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe
        ), offload_context:
            tensor = model(tensor)
        tensor = sync_function(tensor)

    max_mem_used = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.synchronize()

    tensor.sum().backward()

    return max_mem_used


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model_key", model_types.keys())
def test_cpu_offload(fp8_recipe, model_key) -> None:
    """
    We run three configurations:
    (1) No offloading: All activations remain on the GPU between forward and backward passes.
    (2) No offloading (one layer): Only the first layer's activations remain on the GPU between
        forward and backward passes.
    (3) With offloading (all layers): Only the last layer's activations remain on the GPU
        between forward and backward passes, while all other layers are offloaded to the CPU.

    We expect the memory consumption of configurations (2) and (3) to be similar, with
    the difference being the size of the FP8 cache that is not offloaded to the CPU.
    We also expect this memory consumption to be smaller than in scenario (1).
    """
    import gc

    gc.collect()

    model_cls = model_types[model_key]
    models_list = [model_cls() for _ in range(NUM_LAYERS)]

    if model_key in ["multihead_attention", "transformer_layer"]:
        available_backends, *_ = get_available_attention_backends(
            model_config["small"],
            qkv_dtype=torch.bfloat16,
            qkv_layout="sbhd_sbhd_sbhd",
        )
        _, fused_attn_supported, _ = available_backends
        if not fused_attn_supported:
            pytest.skip("Fused attention backend not available.")
        os.environ["NVTE_FLASH_ATTN"] = "0"
        _attention_backends["backend_selection_requires_update"] = True

    without_offloading = _measure_memory_between_forward_and_backward(
        models_list, fp8_recipe, False
    )
    without_offloading_one_layer = _measure_memory_between_forward_and_backward(
        models_list[:1], fp8_recipe, False
    )
    with_offloading = _measure_memory_between_forward_and_backward(models_list, fp8_recipe, True)

    assert with_offloading < without_offloading

    # The only difference between the memory consumption of with_offloading
    # and without_offloading_one_layer should be the size of the FP8 weights cache,
    # which is not offloaded to the CPU.
    memory_consumption_diff = abs(with_offloading - without_offloading_one_layer)
    assert (
        memory_consumption_diff < _get_fp8_weight_cache_size(models_list[1:], fp8_recipe) + EPSILON
    )
