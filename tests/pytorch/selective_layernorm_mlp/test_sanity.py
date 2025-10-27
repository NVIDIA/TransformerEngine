
import torch
import pytest
import os

import transformer_engine.pytorch
from transformer_engine.pytorch import SelectiveLayerNormMLP
from transformer_engine.pytorch.fp8 import (
    fp8_autocast,
    FP8GlobalStateManager,
)
from utils import ModelConfig
from transformer_engine.common import recipe
from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
    is_bf16_compatible,
)

# ----------------------------------------------------------------------------------------------------------------------

# Only run FP8 tests on supported devices.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
fp8_block_scaling_available, _ = FP8GlobalStateManager.is_fp8_block_scaling_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

# Record initial RNG state from script run.
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

model_configs = {
    "126m": ModelConfig(2, 2048, 12, 64, num_layers=12),
    "small": ModelConfig(2, 32, 2, 32, num_layers=2),
    "weird": ModelConfig(3, 37, 3, 23, num_layers=2),
    "large": ModelConfig(2, 128, 4, 128, num_layers=1),
}

fp8_recipes = []
if mxfp8_available:
    fp8_recipes.append(recipe.MXFP8BlockScaling())
if fp8_block_scaling_available:
    fp8_recipes.append(recipe.Float8BlockScaling())
if fp8_available:
    fp8_recipes.append(recipe.Float8CurrentScaling())
    fp8_recipes.append(recipe.DelayedScaling())
fp8_recipes.append(None)

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

all_boolean = [True, False]
batch_sizes_with_zero = [0, 1, 2]
batch_sizes = [1, 2]


all_activations = ["gelu", "relu", "reglu", "geglu", "swiglu", "srelu", "qgelu", "qgeglu"]
all_normalizations = ["LayerNorm", "RMSNorm"]

# ----------------------------------------------------------------------------------------------------------------------

# tests from test_sanity.py

def _disable_wgrads(block):
    for p in block.parameters():
        p.requires_grad = False
        
def _test_sanity_common(
    block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad, microbatching=True
):
    if skip_dgrad and skip_wgrad:
        pytest.skip("No gradient computation; Skipping to avoid PyTorch RuntimeError.")

    te_inp = torch.randn(
        (config.max_seqlen_q, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=not skip_dgrad,
    )

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        if not microbatching:
            te_out = block(te_inp)
        else:
            _ = block(te_inp, is_first_microbatch=True)
            te_out = block(te_inp, is_first_microbatch=False)
    if isinstance(te_out, tuple):
        te_out = te_out[0]
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

def is_fp8_supported(config: ModelConfig):
    if (
        config.max_seqlen_q * config.batch_size % 16
        or config.max_seqlen_kv * config.batch_size % 16
    ):
        return False
    if config.hidden_size % 16 or config.hidden_size_kv % 16:
        return False
    return True

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("skip_dgrad", all_boolean)
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
@pytest.mark.parametrize("microbatching", all_boolean)
def test_sanity_layernorm_mlp(
    dtype,
    fp8_recipe,
    model,
    skip_wgrad,
    zero_centered_gamma,
    skip_dgrad,
    activation,
    normalization,
    microbatching,
):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not is_fp8_supported(config):
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = SelectiveLayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        zero_centered_gamma=zero_centered_gamma,
        activation=activation,
        normalization=normalization,
        params_dtype=dtype,
        device="cuda",
    )
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad, microbatching)
