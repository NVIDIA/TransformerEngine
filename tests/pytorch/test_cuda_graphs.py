# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Iterable, List, Union
import pytest

import torch
from transformer_engine.pytorch import (
    DotProductAttention,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    MultiheadAttention,
    TransformerLayer,
    fp8_autocast,
    fp8_model_init,
    make_graphed_callables,
)
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.utils import is_bf16_compatible
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common import recipe
from utils import ModelConfig, reset_rng_states

# Check if FP8 is supported.
fp8_available, _ = FP8GlobalStateManager.is_fp8_available()
fp8_block_scaling_available, _ = FP8GlobalStateManager.is_fp8_block_scaling_available()
mxfp8_available, _ = FP8GlobalStateManager.is_mxfp8_available()

# Reset RNG states.
reset_rng_states()

model_configs = {
    "small": ModelConfig(32, 2, 2, 32),
}

fp8_recipes = []
if mxfp8_available:
    fp8_recipes.append(recipe.MXFP8BlockScaling())
if fp8_block_scaling_available:
    fp8_recipes.append(recipe.Float8BlockScaling())
if fp8_available:
    fp8_recipes.append(recipe.Float8CurrentScaling())
    fp8_recipes.append(recipe.DelayedScaling())

# Supported data types
dtypes: List[torch.dtype] = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
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
_test_cuda_graphs_modules: List[str] = [
    "transformer",
    "layernorm_mlp",
    "layernorm_linear",
    "linear",
    "mha",
    "linear_op",
]


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
    with fp8_model_init(enabled=fp8_params, recipe=fp8_recipe):
        if module == "transformer":
            modules = [
                TransformerLayer(
                    model_config.hidden_size,
                    model_config.hidden_size,
                    model_config.num_heads,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                    fuse_qkv_params=True,
                    params_dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        elif module == "layernorm_mlp":
            modules = [
                LayerNormMLP(
                    model_config.hidden_size,
                    model_config.hidden_size,
                    params_dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        elif module == "layernorm_linear":
            modules = [
                LayerNormLinear(
                    model_config.hidden_size,
                    model_config.hidden_size,
                    params_dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        elif module == "mha":
            modules = [
                MultiheadAttention(
                    model_config.hidden_size,
                    model_config.num_heads,
                    attention_dropout=0.0,
                    params_dtype=dtype,
                    fuse_qkv_params=True,
                )
                for _ in range(num_layers)
            ]
        elif module == "linear":
            modules = [
                Linear(
                    model_config.hidden_size,
                    model_config.hidden_size,
                    device="cuda",
                    params_dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        elif module == "linear_op":
            modules = [
                te_ops.Sequential(
                    te_ops.Linear(
                        model_config.hidden_size,
                        model_config.hidden_size,
                        dtype=dtype,
                    ),
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
                fp8_enabled=fp8,
                fp8_weight_caching=fp8_weight_caching,
                fp8_recipe=fp8_recipe,
            )
        elif graph_mode == "individual":
            # Graph individual modules.
            modules = [
                make_graphed_callables(
                    module,
                    (generate_data(model_config, dtype, warmup=True),),
                    num_warmup_iters=10,
                    fp8_enabled=fp8,
                    fp8_weight_caching=fp8_weight_caching,
                    fp8_recipe=fp8_recipe,
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
            with fp8_autocast(enabled=fp8, fp8_recipe=fp8_recipe):
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
@pytest.mark.parametrize("fp8_recipe", fp8_recipes + [None])
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
    if fp8 and fp8_recipe.float8_block_scaling() and module == "linear_op":
        pytest.skip("Module not yet supported for float8_block_scaling with CUDA graphs")

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
    outputs = _test_cuda_graphs(graph_mode="none", **kwargs)
    graph_outputs_mode1 = _test_cuda_graphs(graph_mode="full", **kwargs)
    graph_outputs_mode2 = _test_cuda_graphs(graph_mode="individual", **kwargs)

    # Check that results match.
    assert_all_equal(outputs, graph_outputs_mode1)
    assert_all_equal(outputs, graph_outputs_mode2)


_test_make_graphed_callables_with_fp8_weight_caching_modules = [
    "transformer",
    "layernorm_mlp",
    "layernorm_linear",
    "linear",
    "mha",
]


@pytest.mark.parametrize(
    "module",
    _test_make_graphed_callables_with_fp8_weight_caching_modules,
)
@pytest.mark.parametrize("fp8_params", (False, True))
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
def test_make_graphed_callables_with_fp8_weight_caching(
    *,
    module: str,
    fp8_params: bool,
    fp8_recipe: recipe.Recipe,
) -> None:
    test_make_graphed_callables(
        module=module,
        dtype=torch.float32,
        fp8_params=fp8_params,
        fp8_recipe=fp8_recipe,
        fp8_weight_caching=True,
    )


def generate_data_for_dot_product_attention(
    model_config: ModelConfig,
    dtype: torch.dtype,
    warmup: bool = False,
) -> List[torch.Tensor]:
    """Generate synthetic data for dot product attention."""
    gen_func = torch.ones if warmup else torch.randn
    return [
        gen_func(
            model_config.max_seqlen_q,
            model_config.batch_size,
            model_config.num_heads,
            model_config.kv_channels,
            device="cuda",
            requires_grad=True,
            dtype=dtype,
        )
        for _ in range(3)
    ]


def _test_cuda_graphs_with_dot_product_attention(
    *,
    with_graph: bool,
    model_config: ModelConfig,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Helper function for CUDA graph test."""
    reset_rng_states()
    FP8GlobalStateManager.reset()

    # Create dot product attention module.
    assert model_config.hidden_size % model_config.num_heads == 0
    model = DotProductAttention(
        model_config.num_heads,
        model_config.kv_channels,
        attention_dropout=0.0,
    )

    # Graph model if needed.
    if with_graph:
        model = make_graphed_callables(
            model,
            generate_data_for_dot_product_attention(model_config, dtype, warmup=True),
            num_warmup_iters=10,
            fp8_enabled=False,
        )

    # Forward and backward passes.
    for _ in range(3):
        inputs = generate_data_for_dot_product_attention(model_config, dtype)
        grad_output = generate_data(model_config, dtype, requires_grad=False)
        output = model(*inputs)
        output.backward(grad_output)

    return get_outputs(model, output)


@pytest.mark.parametrize("dtype", dtypes)
def test_make_graphed_callables_with_dot_product_attention(
    *,
    model_config: str = "small",
    dtype: torch.dtype,
) -> None:
    """Test CUDA graphs with dot product attention."""
    model_config = model_configs[model_config]
    kwargs = dict(model_config=model_config, dtype=dtype)
    outputs = _test_cuda_graphs_with_dot_product_attention(with_graph=False, **kwargs)
    graph_outputs = _test_cuda_graphs_with_dot_product_attention(with_graph=True, **kwargs)
    assert_all_equal(outputs, graph_outputs)


def _test_cuda_graphs_with_kwargs(
    *,
    with_graph: bool,
    model_config: ModelConfig,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Helper function for CUDA graph test with keyword arguments."""
    reset_rng_states()

    # Initialize model.
    model = TransformerLayer(
        model_config.hidden_size,
        model_config.hidden_size,
        model_config.num_heads,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        self_attn_mask_type="arbitrary",
        fuse_qkv_params=True,
        params_dtype=dtype,
    )

    # Initialize gradient buffers.
    for param in model.parameters():
        param.grad = torch.empty_like(param)

    # Make graphed version of model if needed.
    if with_graph:
        attn_mask = torch.zeros(
            (
                model_config.batch_size,
                1,
                model_config.max_seqlen_q,
                model_config.max_seqlen_kv,
            ),
            dtype=torch.bool,
            device="cuda",
        )
        model = make_graphed_callables(
            model,
            (generate_data(model_config, dtype, warmup=True),),
            sample_kwargs=dict(attention_mask=attn_mask),
            allow_unused_input=True,
        )

    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Training loop.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=False)
        for grad_accumulation_step in range(2):
            input_ = generate_data(model_config, dtype)
            grad_output = generate_data(model_config, dtype, requires_grad=False)
            attn_mask = torch.randint(
                2,
                (
                    model_config.batch_size,
                    1,
                    model_config.max_seqlen_q,
                    model_config.max_seqlen_kv,
                ),
                dtype=torch.bool,
                device="cuda",
            )
            output = model(input_, attention_mask=attn_mask)
            output.backward(grad_output)
        optimizer.step()

    return get_outputs(model, output)


def test_make_graphed_callables_with_kwargs(
    *,
    model_config: str = "small",
    dtype: torch.dtype = torch.float32,
) -> None:
    """Test CUDA graphs with keyword arguments."""
    model_config = model_configs[model_config]
    kwargs = dict(model_config=model_config, dtype=dtype)
    outputs = _test_cuda_graphs_with_kwargs(with_graph=False, **kwargs)
    graph_outputs = _test_cuda_graphs_with_kwargs(with_graph=True, **kwargs)
    assert_all_equal(outputs, graph_outputs)


def _test_cuda_graphs_with_interleaved_pipeline_parallelism(
    *,
    with_graph: bool,
    model_config: ModelConfig,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Simulate Megatron-LM interleaved pipeline parallelism."""
    reset_rng_states()

    # Pipeline parallel configuration.
    num_layers = 2
    num_microbatches = 3
    layer_order = [1, 2, 1, 2, -2, -1, 1, 2, -2, -1, -2, -1]

    # Initialize model.
    model = torch.nn.ModuleList(
        [
            Linear(
                model_config.hidden_size,
                model_config.hidden_size,
                params_dtype=dtype,
            )
            for _ in range(num_layers)
        ]
    )

    # Initialize gradient buffers.
    for param in model.parameters():
        param.grad = torch.empty_like(param)

    # Make graphed version of model if needed.
    layer_forwards = {
        (i % num_layers, i // num_layers): model[i % num_layers]
        for i in range(num_layers * num_microbatches)
    }
    if with_graph:
        sample_args = tuple(
            (generate_data(model_config, dtype, warmup=True),)
            for _ in range(num_layers * num_microbatches)
        )
        layer_forwards = make_graphed_callables(
            tuple(model),
            sample_args,
            allow_unused_input=True,
            _order=layer_order,
        )
        layer_forwards = {
            (i // num_microbatches, i % num_microbatches): forward
            for i, forward in enumerate(layer_forwards)
        }

    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Training loop.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=False)

        # Generate data.
        inputs = {}
        grad_outputs = {}
        for layer_idx in range(num_layers):
            for microbatch_idx in range(num_microbatches):
                x = generate_data(model_config, dtype)
                dy = generate_data(model_config, dtype, requires_grad=False)
                idxs = (layer_idx, microbatch_idx)
                inputs[idxs] = x
                grad_outputs[idxs] = dy

        # Cache for layer outputs.
        outputs = {}

        def forward(layer_idx: int, microbatch_idx: int):
            """Helper function for forward steps"""
            idxs = (layer_idx, microbatch_idx)
            outputs[idxs] = layer_forwards[idxs](inputs[idxs])

        def backward(layer_idx: int, microbatch_idx: int):
            """Helper function for backward steps"""
            outputs[layer_idx, microbatch_idx].backward(grad_outputs[layer_idx, microbatch_idx])

        # Forward and backward steps.
        forward(0, 0)
        forward(1, 0)
        forward(0, 1)
        forward(1, 1)
        backward(1, 0)
        backward(0, 0)
        forward(0, 2)
        forward(1, 2)
        backward(1, 1)
        backward(0, 1)
        backward(1, 2)
        backward(0, 2)

        # Optimizer step.
        optimizer.step()

    outputs = [y for _, y in sorted(outputs.items())]
    return get_outputs(model, outputs)


def test_make_graphed_callables_with_interleaved_pipeline_parallelism(
    *,
    model_config: str = "small",
    dtype: torch.dtype = torch.float16,
) -> None:
    """Test CUDA graphs with Megatron-LM interleaved pipeline parallelism."""
    model_config = model_configs[model_config]
    kwargs = dict(model_config=model_config, dtype=dtype)
    outputs = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=False,
        **kwargs,
    )
    graph_outputs = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=True,
        **kwargs,
    )
    assert_all_equal(outputs, graph_outputs)
