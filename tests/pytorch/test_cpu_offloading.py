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

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

fp8_recipes = [
    None,  # non-fp8
    # recipe.MXFP8BlockScaling(), - scale inverse tensors offloading doest not work yet
    recipe.Float8CurrentScaling(),
    recipe.DelayedScaling(),
]

SIZE = 512
NUM_HEADS = 2
NUM_LAYERS = 5
EPSILON = 0.1

# Flash attention saves some internal tensor for the backward pass
# that cannot be offloaded to CPU.
assert os.getenv("NVTE_FLASH_ATTN") == "0"

# Offloading is supported for attention only for fused and flash attention backends,
# so the use of bfloat16 is required.
#
# For the TransformerLayer, activation offloading with dropout is not supported,
# so we set hidden_dropout to 0.0.
model_types = {
    "linear": lambda fuse_wgrad_accumulation=False: te.Linear(
        SIZE, SIZE, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=fuse_wgrad_accumulation),
    "layernorm_mlp": lambda fuse_wgrad_accumulation=False: te.LayerNormMLP(
        SIZE, SIZE, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=fuse_wgrad_accumulation),
    "layernorm_linear": lambda fuse_wgrad_accumulation=False: te.LayerNormLinear(
        SIZE, SIZE, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=fuse_wgrad_accumulation),
    "multihead_attention": lambda fuse_wgrad_accumulation=False: te.MultiheadAttention(
        SIZE, NUM_HEADS, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=fuse_wgrad_accumulation),
    "transformer_layer": lambda fuse_wgrad_accumulation=False: te.TransformerLayer(
        SIZE, SIZE, NUM_HEADS, params_dtype=torch.bfloat16, 
        hidden_dropout=0.0, fuse_wgrad_accumulation=fuse_wgrad_accumulation, fuse_qkv_params=fuse_wgrad_accumulation),
}


def _get_input():
    return torch.randn((128, SIZE, SIZE), dtype=torch.bfloat16, requires_grad=True).cuda()


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

def _get_weights_size(models):
    params_bytes = 0
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                params_bytes += param.numel() * param.element_size()
    return params_bytes / (1024**2)

def _delayed_scaling_helper_tensor_size(models):
    return 0.05 * len(models)

def _alloc_main_grad(models, device):
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                param.main_grad = torch.randn_like(param).to(device).to(torch.float32)

def _measure_memory_between_forward_and_backward(models, fp8_recipe, cpu_offload, microbatching=False):
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
    
    def run_fwd(is_first_microbatch):
        tensor = _get_input()

        for model in models:
            with te.fp8_autocast(
                enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe
            ), offload_context:
                tensor = model(tensor, is_first_microbatch=is_first_microbatch)
            tensor = sync_function(tensor)

        torch.cuda.synchronize()
        max_mem_used = torch.cuda.memory_allocated() / (1024**2)

        #tensor.sum().backward()
        return max_mem_used
    
    if microbatching:
        mem1 = run_fwd(True)  
        mem2 = run_fwd(False)
        return max(mem1, mem2)
    else:
        return run_fwd(None)



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

    model_cls = model_types[model_key]
    models_list = [model_cls() for _ in range(NUM_LAYERS)]

    if fp8_recipe and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if fp8_recipe is not None:
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)

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

@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("microbatching", [True, False])
@pytest.mark.parametrize("model_key", model_types.keys())
def test_weight_offloading(fp8_recipe, model_key, microbatching):
    if model_key == "multihead_attention":
        pytest.skip("MultiheadAttention does not support weight offloading with interleaved layers")

    with te.cpu_model_init(enabled=True):
        models_cpu_init = [model_types[model_key](fuse_wgrad_accumulation=True) for _ in range(NUM_LAYERS)]
    
    _alloc_main_grad(models_cpu_init, device="cpu")

    with_cpu_init = _measure_memory_between_forward_and_backward(models_cpu_init, fp8_recipe, False, microbatching=microbatching)
    del models_cpu_init
    

    with te.cpu_model_init(enabled=False):
        models_no_cpu_init = [model_types[model_key](fuse_wgrad_accumulation=True) for _ in range(NUM_LAYERS)]
    
    _alloc_main_grad(models_no_cpu_init, device="cuda")
    
    without_cpu_init = _measure_memory_between_forward_and_backward(models_no_cpu_init, fp8_recipe, False, microbatching=microbatching)
    #assert with_cpu_init < without_cpu_init
    if model_key != "transformer_layer":
        weights_size = _get_weights_size(models_no_cpu_init)
        weights_fp8_cache_size = _get_fp8_weight_cache_size(models_no_cpu_init, fp8_recipe)
        main_grad_buffer_size = _get_weights_size(models_no_cpu_init) * 2 # dtype is float32, not bfloat16, so the size is double the weights size
        tolerance = EPSILON
        if fp8_recipe is not None:
            if fp8_recipe.delayed():
                tolerance += _delayed_scaling_helper_tensor_size(models_no_cpu_init)

        assert abs((without_cpu_init - with_cpu_init) - weights_size - weights_fp8_cache_size - main_grad_buffer_size) < tolerance


