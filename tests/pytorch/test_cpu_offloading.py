# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import contextlib
import gc
import os
from typing import Iterable, Optional

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported
from utils import ModelConfig, get_available_attention_backends

# Check supported quantization schemes
fp8_available = te.is_fp8_available()
mxfp8_available = te.is_mxfp8_available()

quantization_recipes: Optional[recipe.Recipe] = [None]
if fp8_available:
    quantization_recipes.extend((recipe.Float8CurrentScaling(), recipe.DelayedScaling()))

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
    "linear_op": lambda: te.ops.Linear(SIZE, SIZE, dtype=torch.bfloat16),
    "layernorm_mlp_ops": lambda: te.ops.Sequential(
        te.ops.LayerNorm(SIZE, dtype=torch.bfloat16),
        te.ops.Linear(SIZE, SIZE, dtype=torch.bfloat16),
        te.ops.GELU(),
        te.ops.Linear(SIZE, SIZE, dtype=torch.bfloat16),
    ),
}


def _make_input() -> torch.Tensor:
    """Generate random input tensor."""
    return torch.randn(
        (128, SIZE, SIZE),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )


def _warmup_model(
    modules: Iterable[torch.nn.Module],
    quantization_recipe: Optional[recipe.Recipe],
) -> None:
    """Perform forward and backward pass"""
    tensor = _make_input()
    for module in modules:
        with te.autocast(
            enabled=quantization_recipe is not None,
            recipe=quantization_recipe,
        ):
            tensor = module(tensor)
    tensor.sum().backward()


def _estimate_cached_weight_size(
    model_name: str,
    modules: Iterable[torch.nn.Module],
    quantization_recipe: Optional[recipe.Recipe],
) -> float:
    """Calculate the memory (in MiB) needed for weight caching."""

    # The weight params are cached directly for unquantized compute
    if quantization_recipe is None:
        return 0

    # Count number of weight param elements
    param_elements = 0
    for module in modules:
        for param in module.parameters():
            if param.dim() == 2:
                param_elements += param.numel()

    # FP8 tensor-scaling caches one byte per element
    if quantization_recipe.delayed() or quantization_recipe.float8_current_scaling():
        if not is_non_tn_fp8_gemm_supported() and model_name not in (
            "linear_op",
            "layernorm_mlp_ops",
        ):
            # Modules do not deallocate FP8 transpose for weights
            return 2 * param_elements / 1024**2
        return param_elements / 1024**2

    # MXFP8 caches one data byte per element and one scale byte per 32
    # elements
    if quantization_recipe.mxfp8():
        if model_name not in ("linear_op", "layernorm_mlp_ops"):
            # Modules do not deallocate column-wise MXFP8 data for weights
            return 2 * param_elements * (1 + 1 / 32) / 1024**2
        return param_elements * (1 + 1 / 32) / 1024**2

    raise NotImplementedError(f"Unrecognized recipe ({quantization_recipe})")


def _measure_cached_memory(
    modules: Iterable[torch.nn.Module],
    quantization_recipe: Optional[recipe.Recipe],
    cpu_offload: bool,
) -> float:
    """Measure the growth in allocated GPU memory in MiB after a model forward pass.

    Memory measurement excludes the input and output tensors.

    """

    # Reset memory
    gc.collect()
    torch.cuda.empty_cache()

    # Context and sync function for CPU offloading
    if cpu_offload:
        offload_context, sync_function = te.get_cpu_offload_context(
            enabled=True,
            num_layers=len(modules),
            model_layers=len(modules) + 1,
            offload_activations=True,
            offload_weights=False,
        )
    else:
        offload_context = contextlib.nullcontext()
        sync_function = lambda x: x

    # Forward pass, with dummy step to trigger offload for last module
    inp = _make_input()
    tensor = inp
    memory_before_forward = torch.cuda.memory_allocated() / (1024**2)
    for module in modules:
        with te.autocast(
            enabled=quantization_recipe is not None, recipe=quantization_recipe
        ), offload_context:
            tensor = module(tensor)
        tensor = sync_function(tensor)
    with offload_context:
        tensor = tensor.clone()
    tensor = sync_function(tensor)
    memory_after_forward = (torch.cuda.memory_allocated() - tensor.nbytes) / (1024**2)

    # Backward pass
    tensor.sum().backward()
    torch.cuda.synchronize()

    # Memory usage in MiB
    return memory_after_forward - memory_before_forward


@pytest.mark.parametrize("quantization_recipe", quantization_recipes)
@pytest.mark.parametrize("model_name", model_types.keys())
def test_cpu_offload(quantization_recipe: Optional[recipe.Recipe], model_name: str) -> None:
    """Check that CPU offloading runs and has expected memory usage."""

    # Construct model
    modules_list = [model_types[model_name]() for _ in range(NUM_LAYERS)]
    if model_name in ["multihead_attention", "transformer_layer"]:
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

    # Warmup
    _warmup_model(modules_list, quantization_recipe)

    # Measure cached memory after forward pass
    memory_without_offload = _measure_cached_memory(modules_list, quantization_recipe, False)
    memory_with_offload = _measure_cached_memory(modules_list, quantization_recipe, True)

    # Check for expected memory usage
    assert memory_with_offload < memory_without_offload
    memory_from_cached_weights = _estimate_cached_weight_size(
        model_name,
        modules_list,
        quantization_recipe,
    )
    assert abs(memory_with_offload - memory_from_cached_weights) < EPSILON
