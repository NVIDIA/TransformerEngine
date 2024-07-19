# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
import itertools
from typing import Iterable, List, Tuple, Union
import pytest

import torch
from transformer_engine.pytorch import (
    DotProductAttention,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    make_graphed_callables,
    MultiheadAttention,
    TransformerLayer,
    fp8_autocast,
    fp8_model_init,
)
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.utils import is_bf16_compatible


# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# Record initial RNG state from script run.
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()


@dataclass
class ModelConfig:
    """Data tensor dimensions within Transformer model"""

    sequence_length: int
    batch_size: int
    hidden_size: int
    num_heads: int
    kv_channels: int


model_configs = {"small": ModelConfig(2, 32, 64, 2, 32)}

modules = ["transformer", "layernorm_mlp", "layernorm_linear", "linear", "mha", "dpa"]

all_boolean = [True, False]

dtypes = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    dtypes.append(torch.bfloat16)


def reset_rng_states() -> None:
    """revert back to initial RNG state."""
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state(_cuda_rng_state)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def assert_all_equal(l1: List[torch.Tensor], l2: List[torch.Tensor], names=None) -> bool:
    """Ensures two lists are equal."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    failed = False
    failed_tensors = ""
    for i, (t1, t2) in enumerate(zip(l1, l2)):
        if not torch.equal(t1, t2):
            failed = True
            failed_tensors += (
                f"    {names[i]}\n" if names is not None else f"    tensor at idx={i}\n"
            )
    assert not failed, "Output mismatches in:\n" + failed_tensors


def generate_data(
    config: ModelConfig,
    dtype: torch.dtype,
    dpa: bool = False,
    warmup: bool = False,
    return_grad_output: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Generate synthetic data."""
    gen_func = torch.ones if warmup else torch.randn
    if dpa:
        inputs = [
            gen_func(
                config.sequence_length,
                config.batch_size,
                config.num_heads,
                config.kv_channels,
                device="cuda",
                requires_grad=True,
                dtype=dtype,
            )
            for _ in range(3)
        ]
    else:
        inputs = [
            gen_func(
                config.sequence_length,
                config.batch_size,
                config.hidden_size,
                device="cuda",
                requires_grad=True,
                dtype=dtype,
            )
        ]

    if not return_grad_output:
        return inputs

    grad_output = torch.randn(
        config.sequence_length,
        config.batch_size,
        config.hidden_size,
        device="cuda",
        dtype=dtype,
    )
    return inputs, grad_output


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


def _test_cuda_graphs(
    *,
    config: ModelConfig,
    num_layers: int,
    dtype: torch.dtype,
    fp8: bool,
    fp8_params: bool,
    fp8_weight_caching: bool,
    module: str,
    graph_mode: str,
) -> List[torch.Tensor]:
    """Helper function for CUDA graph test."""
    reset_rng_states()
    FP8GlobalStateManager.reset()
    dpa = module == "dpa"

    with fp8_model_init(enabled=fp8_params):
        # Create modules.
        if module == "transformer":
            modules = [
                TransformerLayer(
                    config.hidden_size,
                    config.hidden_size,
                    config.num_heads,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                    fuse_qkv_params=True,
                    params_dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        elif module == "layernorm_mlp":
            modules = [
                LayerNormMLP(config.hidden_size, config.hidden_size, params_dtype=dtype)
                for _ in range(num_layers)
            ]
        elif module == "layernorm_linear":
            modules = [
                LayerNormLinear(config.hidden_size, config.hidden_size, params_dtype=dtype)
                for _ in range(num_layers)
            ]
        elif module == "mha":
            modules = [
                MultiheadAttention(
                    config.hidden_size,
                    config.num_heads,
                    attention_dropout=0.0,
                    params_dtype=dtype,
                    fuse_qkv_params=True,
                )
                for _ in range(num_layers)
            ]
        elif dpa:
            assert config.hidden_size % config.num_heads == 0, "Err."
            assert num_layers == 1, "Err."
            modules = [
                DotProductAttention(config.num_heads, config.kv_channels, attention_dropout=0.0)
                for _ in range(num_layers)
            ]
        else:
            modules = [
                Linear(config.hidden_size, config.hidden_size, device="cuda", params_dtype=dtype)
                for _ in range(num_layers)
            ]

        # Initialize gradient buffers.
        for module in modules:
            for param in module.parameters():
                param.grad = torch.empty_like(param)

        # Generate model and wrap API to return graphed version.
        if graph_mode == "full":
            # Graph entire model at once.
            model = modules[0] if dpa else torch.nn.Sequential(*modules)
            model = make_graphed_callables(
                model,
                generate_data(config, dtype, dpa=dpa, warmup=True),
                num_warmup_iters=10,
                fp8_enabled=fp8,
                fp8_weight_caching=fp8_weight_caching,
            )
        elif graph_mode == "individual":
            # Graph individual modules
            modules = [
                make_graphed_callables(
                    module,
                    generate_data(config, dtype, dpa=dpa, warmup=True),
                    num_warmup_iters=10,
                    fp8_enabled=fp8,
                    fp8_weight_caching=fp8_weight_caching,
                )
                for module in modules
            ]
            model = modules[0] if dpa else _Sequential(*modules)
        else:
            model = modules[0] if dpa else _Sequential(*modules)

    # Optimizer.
    if not dpa:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Launch.
    for _ in range(3):
        if not dpa:
            optimizer.zero_grad(set_to_none=False)
        for grad_accumulation_step in range(2):
            inputs, grad_output = generate_data(config, dtype, dpa=dpa, return_grad_output=True)
            with fp8_autocast(enabled=fp8):
                kwargs = {}
                if fp8_weight_caching:
                    kwargs["is_first_microbatch"] = grad_accumulation_step == 0
                output = model(*inputs, **kwargs)
            output.backward(grad_output)
        if not dpa:
            optimizer.step()

    return get_outputs(model, output)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("fp8_params", all_boolean)
@pytest.mark.parametrize("fp8_weight_caching", all_boolean)
@pytest.mark.parametrize("module", modules)
def test_gpt_make_graphed_callables(
    dtype: torch.dtype,
    model: str,
    num_layers: int,
    fp8: bool,
    fp8_params: bool,
    fp8_weight_caching: bool,
    module: str,
) -> None:
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if fp8_params and not fp8:
        pytest.skip("FP8 needed for FP8 parameters.")
    if fp8_weight_caching and not fp8:
        pytest.skip("FP8 needed for FP8 parameters.")
    if module == "dpa" and num_layers > 1:
        pytest.skip("Max 1 layer for DPA.")

    config = model_configs[model]

    kwargs = dict(
        config=config,
        num_layers=num_layers,
        dtype=dtype,
        fp8=fp8,
        fp8_params=fp8_params,
        fp8_weight_caching=fp8_weight_caching,
        module=module,
    )
    outputs = _test_cuda_graphs(graph_mode="none", **kwargs)
    graph_outputs_mode1 = _test_cuda_graphs(graph_mode="full", **kwargs)
    graph_outputs_mode2 = _test_cuda_graphs(graph_mode="individual", **kwargs)

    # Check that results match
    assert_all_equal(outputs, graph_outputs_mode1)
    assert_all_equal(outputs, graph_outputs_mode2)


def _test_cuda_graphs_with_kwargs(
    *,
    config: ModelConfig,
    dtype: torch.dtype,
    with_graph: bool,
) -> List[torch.Tensor]:
    """Simulate Megatron-LM interleaved pipeline parallelism."""
    reset_rng_states()

    # Initialize model.
    model = TransformerLayer(
        config.hidden_size,
        config.hidden_size,
        config.num_heads,
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
            (config.batch_size, 1, config.sequence_length, config.sequence_length),
            dtype=torch.bool,
            device="cuda",
        )
        model = make_graphed_callables(
            model,
            generate_data(config, dtype, warmup=True),
            sample_kwargs=dict(attention_mask=attn_mask),
            allow_unused_input=True,
        )

    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Training loop.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=False)
        for grad_accumulation_step in range(2):
            inputs, grad_output = generate_data(config, dtype, return_grad_output=True)
            attn_mask = torch.randint(
                2,
                (config.batch_size, 1, config.sequence_length, config.sequence_length),
                dtype=torch.bool,
                device="cuda",
            )
            output = model(*inputs, attention_mask=attn_mask)
            output.backward(grad_output)
        optimizer.step()

    return get_outputs(model, output)


def test_make_graphed_callables_with_kwargs(
    dtype: torch.dtype = torch.float32,
    model: str = "small",
) -> None:
    """Test CUDA graphs with keyword arguments."""
    config = model_configs[model]
    kwargs = dict(config=config, dtype=dtype)
    outputs = _test_cuda_graphs_with_kwargs(with_graph=False, **kwargs)
    graph_outputs = _test_cuda_graphs_with_kwargs(with_graph=True, **kwargs)
    assert_all_equal(outputs, graph_outputs)


def _test_cuda_graphs_with_interleaved_pipeline_parallelism(
    *,
    config: ModelConfig,
    dtype: torch.dtype,
    with_graph: bool,
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
                config.hidden_size,
                config.hidden_size,
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
            generate_data(config, dtype, warmup=True) for _ in range(num_layers * num_microbatches)
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
                x, dy = generate_data(config, dtype, return_grad_output=True)
                idxs = (layer_idx, microbatch_idx)
                inputs[idxs] = x[0]
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
    dtype: torch.dtype = torch.float16,
    model: str = "small",
) -> None:
    """Test CUDA graphs with Megatron-LM interleaved pipeline parallelism."""
    config = model_configs[model]
    kwargs = dict(config=config, dtype=dtype)
    outputs = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=False,
        **kwargs,
    )
    graph_outputs = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=True,
        **kwargs,
    )
    assert_all_equal(outputs, graph_outputs)
