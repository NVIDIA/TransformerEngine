# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Iterable, List, Union
import pytest

import torch
from transformer_engine.pytorch import (
    SelectiveLayerNormMLP,
    autocast,
    quantized_model_init,
    make_graphed_callables,
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_mxfp8_available,
    is_bf16_available,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common import recipe
from utils import ModelConfig, reset_rng_states

# Check if FP8 is supported.
fp8_available = is_fp8_available()
fp8_block_scaling_available = is_fp8_block_scaling_available()
mxfp8_available = is_mxfp8_available()

# Reset RNG states.
reset_rng_states()

model_configs = {
    "small": ModelConfig(2, 32, 2, 32),
}


def nvfp4_vanilla():
    nvfp4_recipe = recipe.NVFP4BlockScaling()
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
    return nvfp4_recipe


def nvfp4_rht_and_2d_quantization():
    nvfp4_recipe = recipe.NVFP4BlockScaling()
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(
        random_hadamard_transform=True, fp4_2d_quantization=False
    )
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(
        random_hadamard_transform=False, fp4_2d_quantization=True
    )
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(
        random_hadamard_transform=True, fp4_2d_quantization=False
    )
    return nvfp4_recipe


def check_rht_usage(recipe: recipe.Recipe) -> bool:
    # if using RHT, we can only support bf16
    # check fp4_quant_fwd_inp, fp4_quant_fwd_weight, fp4_quant_bwd_grad
    if recipe.nvfp4():
        if (
            recipe.fp4_quant_fwd_inp.random_hadamard_transform
            or recipe.fp4_quant_fwd_weight.random_hadamard_transform
            or recipe.fp4_quant_bwd_grad.random_hadamard_transform
        ):
            return True
    return False


def get_nvfp4_inp_supported_dtypes(recipe: recipe.Recipe, dtype: torch.dtype) -> bool:
    supported_input_dtypes = []
    if recipe.nvfp4():
        supported_input_dtypes.append(torch.bfloat16)
        # if not using RHT, we can add fp32 as well
    if not check_rht_usage(recipe):
        supported_input_dtypes.append(torch.float32)
    return supported_input_dtypes


fp8_recipes = []
if mxfp8_available:
    fp8_recipes.append(recipe.MXFP8BlockScaling())
    fp8_recipes.append(nvfp4_rht_and_2d_quantization())
if fp8_block_scaling_available:
    fp8_recipes.append(recipe.Float8BlockScaling())
if fp8_available:
    fp8_recipes.append(recipe.Float8CurrentScaling())
    fp8_recipes.append(recipe.DelayedScaling())

# Supported data types
dtypes: List[torch.dtype] = [torch.float32, torch.float16]
if is_bf16_available():  # bf16 requires sm_80 or higher
    dtypes.append(torch.bfloat16)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def assert_all_equal(l1: List[torch.Tensor], l2: List[torch.Tensor], names=None) -> bool:
    """Check that two lists of tensors match exactly."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    failure_message = "Output mismatches in:"
    failed_tensors = []
    for i, (t1, t2) in enumerate(zip(l1, l2)):
        if not torch.equal(t1, t2):
            failure_message += "\n    "
            if names is None:
                failure_message += f"tensor at idx={i}"
            else:
                failure_message += names[i]
            failed_tensors.append((t1, t2))
    if failed_tensors:
        print(failure_message)
        t1, t2 = failed_tensors[0]
        torch.testing.assert_close(t1, t2, rtol=0, atol=0)


def generate_data(
    model_config: ModelConfig,
    dtype: torch.dtype,
    warmup: bool = False,
    requires_grad: bool = True,
) -> torch.Tensor:
    """Generate synthetic data."""
    gen_func = torch.ones if warmup else torch.randn
    return gen_func(
        model_config.max_seqlen_q,
        model_config.batch_size,
        model_config.hidden_size,
        device="cuda",
        requires_grad=requires_grad,
        dtype=dtype,
    )


def get_outputs(
    model: torch.nn.Module,
    output: Union[torch.Tensor, Iterable[torch.Tensor]],
) -> List[torch.Tensor]:
    """Return grads and params for comparsion."""
    values = []
    for param in model.parameters():
        values.append(param)
        if param.grad is not None:
            values.append(param.grad)
    if isinstance(output, torch.Tensor):
        values.append(output)
    else:
        values.extend(output)
    return values


class _Sequential(torch.nn.Sequential):
    """Sequential model that forwards keyword arguments to modules"""

    def forward(self, input_: torch.Tensor, **kwargs) -> torch.Tensor:
        x = input_
        for module in self:
            x = module(x, **kwargs)
        return x


# Supported modules
_test_cuda_graphs_modules: List[str] = ["selective_layernorm_mlp"]

def _test_cuda_graphs(
    *,
    graph_mode: str,
    module: str,
    model_config: ModelConfig,
    num_layers: int,
    dtype: torch.dtype,
    fp8: bool,
    fp8_params: bool,
    fp8_weight_caching: bool,
    fp8_recipe: recipe.Recipe,
) -> List[torch.Tensor]:
    """Helper function for CUDA graph test."""
    reset_rng_states()
    FP8GlobalStateManager.reset()

    # Operation-based API does not support FP8 weight caching.
    if module == "linear_op":
        fp8_weight_caching = False

    # Create modules.
    with quantized_model_init(enabled=fp8_params, recipe=fp8_recipe):

        if module == "selective_layernorm_mlp":
            modules = [
                SelectiveLayerNormMLP(
                    model_config.hidden_size,
                    model_config.hidden_size,
                    params_dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        
        else:
            raise ValueError(f"Unknown module type ({module})")

        # Initialize gradient buffers.
        for module in modules:
            for param in module.parameters():
                param.grad = torch.empty_like(param)

        # Generate model and wrap API to return graphed version.
        if graph_mode == "full":
            # Graph entire model at once.
            model = torch.nn.Sequential(*modules)
            model = make_graphed_callables(
                model,
                (generate_data(model_config, dtype, warmup=True),),
                num_warmup_iters=10,
                enabled=fp8,
                cache_quantized_params=fp8_weight_caching,
                recipe=fp8_recipe,
            )
        elif graph_mode == "individual":
            # Graph individual modules.
            modules = [
                make_graphed_callables(
                    module,
                    (generate_data(model_config, dtype, warmup=True),),
                    num_warmup_iters=10,
                    enabled=fp8,
                    cache_quantized_params=fp8_weight_caching,
                    recipe=fp8_recipe,
                )
                for module in modules
            ]
            model = _Sequential(*modules)
        else:
            model = _Sequential(*modules)

    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Training steps.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=False)
        for grad_accumulation_step in range(2):
            input_ = generate_data(model_config, dtype)
            grad_output = generate_data(model_config, dtype, requires_grad=False)
            with autocast(enabled=fp8, recipe=fp8_recipe):
                kwargs = {}
                if fp8_weight_caching:
                    kwargs["is_first_microbatch"] = grad_accumulation_step == 0
                output = model(input_, **kwargs)
            output.backward(grad_output)
        optimizer.step()

    return get_outputs(model, output)


@pytest.mark.parametrize("module", _test_cuda_graphs_modules)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("fp8_params", (False, True))
@pytest.mark.parametrize("fp8_recipe", fp8_recipes + [None], ids=lambda r: type(r).__name__)
def test_make_graphed_callables(
    *,
    module: str,
    model_config: str = "small",
    num_layers: int = 3,
    dtype: torch.dtype,
    fp8_params: bool,
    fp8_recipe: recipe.Recipe,
    fp8_weight_caching: bool = False,
) -> None:

    fp8 = fp8_recipe is not None
    if fp8_params and not fp8:
        pytest.skip("FP8 needed for FP8 parameters.")
    if fp8_weight_caching and not fp8:
        pytest.skip("FP8 needed for FP8 parameters.")
    if fp8 and (fp8_recipe.float8_block_scaling() or fp8_recipe.nvfp4()) and module == "linear_op":
        pytest.skip(
            f"Module not yet supported for {fp8_recipe.__class__.__name__} with CUDA graphs"
        )
    if fp8 and fp8_recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(fp8_recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe"
                f" {fp8_recipe.__class__.__name__}"
            )
        if fp8_params:
            pytest.skip("NVFP4 params not supported")

    # Run model with different CUDA graph settings.
    model_config = model_configs[model_config]
    kwargs = dict(
        module=module,
        model_config=model_config,
        num_layers=num_layers,
        dtype=dtype,
        fp8=fp8,
        fp8_params=fp8_params,
        fp8_weight_caching=fp8_weight_caching,
        fp8_recipe=fp8_recipe,
    )
    # Put graphed callables first to test the case where the cuda context might not be set in
    # creating TMA descriptor for MXFP8 quantization.
    graph_outputs_mode1 = _test_cuda_graphs(graph_mode="full", **kwargs)
    graph_outputs_mode2 = _test_cuda_graphs(graph_mode="individual", **kwargs)
    outputs = _test_cuda_graphs(graph_mode="none", **kwargs)

    # Check that results match.
    assert_all_equal(outputs, graph_outputs_mode1)
    assert_all_equal(outputs, graph_outputs_mode2)


_test_make_graphed_callables_with_fp8_weight_caching_modules = [
    "selective_layernorm_mlp",
]


@pytest.mark.parametrize(
    "module",
    _test_make_graphed_callables_with_fp8_weight_caching_modules,
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("fp8_params", (False, True))
@pytest.mark.parametrize("fp8_recipe", fp8_recipes, ids=lambda r: type(r).__name__)
def test_make_graphed_callables_with_fp8_weight_caching(
    *,
    module: str,
    dtype: torch.dtype,
    fp8_params: bool,
    fp8_recipe: recipe.Recipe,
) -> None:
    test_make_graphed_callables(
        module=module,
        dtype=dtype,
        fp8_params=fp8_params,
        fp8_recipe=fp8_recipe,
        fp8_weight_caching=True,
    )
