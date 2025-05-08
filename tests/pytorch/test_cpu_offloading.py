# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.quantized_tensor import prepare_for_saving, restore_from_saved
from transformer_engine.pytorch.cpu_offload import offload, _manual_reload, CPUOffload

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_available, reason_for_no_fp8_block = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)

fp8_recipes = [
    None,  # non-fp8
    recipe.MXFP8BlockScaling(),
    recipe.Float8CurrentScaling(),
    recipe.DelayedScaling(),
    recipe.Float8BlockScaling(),
]

SIZE = 512
NUM_HEADS = 8
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
    return torch.empty((1, SIZE, SIZE), dtype=torch.bfloat16, requires_grad=True).cuda()

def test_auto_offload():
    def run(offload_enabled):
        inp = _get_input()

        def compute(input_tensor):
            x = _get_input()
            if offload_enabled:
                offload(x)
            y = input_tensor * x
            # x is necessary for backward pass, thus it will be saved.
            return y

        cpu_offload = CPUOffload()

        y = cpu_offload(compute, inp)
        cpu_offload.sync_before_bwd()

        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        y.sum().backward()  # for sanity check
        return memory_allocated

    # x will be offloaded to CPU when offload_enabled is True
    # which should result in SIZE * SIZE * 2 / (1024 ** 2) memory allocated
    assert run(True) < run(False)
    assert run(False) - run(True) > (SIZE * SIZE * 2 / (1024 ** 2)) - EPSILON

def _tensor_size(x):
    if type(x) == torch.Tensor:
        return x.numel() * x.element_size() / (1024 ** 2)
    elif type(x) == te.float8_tensor.Float8Tensor:
        return x._data.numel() * x._data.element_size() / (1024 ** 2)
    elif type(x) == te.tensor._internal.float8_tensor_base.Float8TensorBase:
        return x._data.numel() * x._data.element_size() / (1024 ** 2)
    else:
        raise ValueError(f"Unknown tensor type: {type(x)}")

tensor_empty_constructrs = {
    "tensor": lambda: torch.empty((SIZE, SIZE), dtype=torch.bfloat16).cuda(),
    "float8tensor": lambda: te.float8_tensor.Float8Tensor(
        data=torch.empty((SIZE, SIZE), dtype=torch.uint8, device="cuda"),
        fp8_scale_inv=torch.tensor(1.0).cuda(),
        fp8_dtype=tex.DType.kFloat8E4M3,
        shape=(SIZE, SIZE),
        dtype=torch.bfloat16,
        device="cuda",
    ),
    "float8tensorbase": lambda: te.tensor._internal.float8_tensor_base.Float8TensorBase(
        data=torch.empty((SIZE, SIZE), dtype=torch.uint8, device="cuda"),
        fp8_scale_inv=torch.tensor(1.0).cuda(),
        fp8_dtype=tex.DType.kFloat8E4M3,
        shape=(SIZE, SIZE),
        dtype=torch.bfloat16,
    ),
}

@pytest.mark.parametrize("x_tensor_type", tensor_empty_constructrs.keys())
def test_manual_offload(x_tensor_type):
    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, x, offload_enabled):
            if offload_enabled:
                offload(x, manual_reload=True)
            
            tensors, tensor_objects = prepare_for_saving(x, input_tensor)
            ctx.tensor_objects = tensor_objects
            ctx.save_for_backward(*tensors)
            ctx.offload_enabled = offload_enabled
            return input_tensor
        
        @staticmethod
        def backward(ctx, _):
            torch.cuda.synchronize()
            x, input_tensor = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
            if ctx.offload_enabled:
                if hasattr(x, "device"):
                    assert x.device.type == "cpu"
                x = _manual_reload(x)
            #if hasattr(x, "device"): 
            #    assert x.device.type == "cuda"
            return input_tensor, None, None

    def run(offload_enabled):
        inp = _get_input()
        def compute(input_tensor):
            x = tensor_empty_constructrs[x_tensor_type]()
            return Function.apply(input_tensor, x, offload_enabled)

        cpu_offload = CPUOffload()

        y = cpu_offload(compute, inp)
        cpu_offload.sync_before_bwd()

        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        y.sum().backward()  # for sanity check
        return memory_allocated

    # x will be offloaded to CPU when offload_enabled is True
    assert run(True) < run(False)
    diff = run(False) - run(True)
    assert abs(diff - _tensor_size(tensor_empty_constructrs[x_tensor_type]())) < EPSILON

def _get_fp8_weight_cache_size(model, fp8_recipe):
    """
    Calculate the total FP8 weight cache size (in MB) for a list of models.
    """
    if fp8_recipe is None:
        return 0

    params_bytes = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            params_bytes += param.numel()

    # One byte for columnwise and one byte for rowwise,
    # hence multiply by 2 and convert to MB
    # there is 1 byte of scale per 32 elements in mxFP8
    factor_for_scale_inv_tensor = (1 + 1 / 32) if fp8_recipe.mxfp8() else 1
    return (2 * params_bytes * factor_for_scale_inv_tensor) / (1024**2)

@pytest.mark.parametrize("layer_type", model_types.keys())
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
def test_cpu_offload_on_layers(layer_type, fp8_recipe):
    if not fp8_available and fp8_recipe is not None:
        pytest.skip(reason_for_no_fp8)
    if fp8_recipe is not None:
        if not mxfp8_available and fp8_recipe.mxfp8():
            pytest.skip(reason_for_no_mxfp8)
        if not fp8_block_available and fp8_recipe.float8_block_scaling():
            pytest.skip(reason_for_no_fp8_block)
    model = model_types[layer_type]()

    def _get_memory(offload_enabled):
        def comp():
            inp = _get_input()
            with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
                y = model(inp)
            return y
        cpu_offload = CPUOffload()
        if offload_enabled:
            y = cpu_offload(comp)
            cpu_offload.sync_before_bwd()
        else:
            y = comp()
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        #y.sum().backward()
        return memory_allocated
    
    # warm up
    _get_memory(True)
    _get_memory(False)
    initial_memory = torch.cuda.memory_allocated() / (1024**2)

    with_offload = _get_memory(True)
    without_offload = _get_memory(False)
    print(f"initial_memory: {initial_memory}, with_offload: {with_offload}, without_offload: {without_offload}")
    assert with_offload < without_offload
    diff = with_offload - initial_memory
    output_size = _tensor_size(_get_input())
    assert (
        diff < _get_fp8_weight_cache_size(model, fp8_recipe) + EPSILON + output_size
    )

# prepare more general tests.