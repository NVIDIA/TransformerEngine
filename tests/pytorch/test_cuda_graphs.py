# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Callable, Dict, Iterable, List, Tuple, Union
import pytest
import contextlib
import copy
import gc
import weakref

import torch
from transformer_engine.pytorch import (
    DotProductAttention,
    GroupedLinear,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    MultiheadAttention,
    TransformerLayer,
    autocast,
    quantized_model_init,
    make_graphed_callables,
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_mxfp8_available,
    is_nvfp4_available,
    is_bf16_available,
)
import transformer_engine_torch as tex
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine.pytorch.ops as te_ops
import transformer_engine.pytorch.graph as te_graph
from transformer_engine.common import recipe
from utils import ModelConfig, recipe_id, reset_rng_states, skip_unsupported_backward_override

# Check if FP8 is supported.
fp8_available = is_fp8_available()
fp8_block_scaling_available = is_fp8_block_scaling_available()
mxfp8_available = is_mxfp8_available()
nvfp4_available = is_nvfp4_available()

# Reset RNG states.
reset_rng_states()

model_configs = {
    "small": ModelConfig(2, 32, 2, 32),
}


def test_slot_memory_arena_view_uses_typed_storage_offset():
    backing = torch.empty(16, dtype=torch.float32, device="cuda")
    arena = backing[4:]
    spec = ("output", None, (1,), (1,), torch.float32, backing.device, False, 4)

    view = te_graph._arena_view(arena, 4, spec)

    assert view.data_ptr() == backing.data_ptr() + 5 * backing.element_size()


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


def nvfp4_row_scaled():
    nvfp4_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="dequantized",
    )
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
    return nvfp4_recipe


def nvfp4_4over6():
    nvfp4_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        nvfp4_4over6="all",
    )
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(fp4_2d_quantization=True)
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
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
if nvfp4_available:
    fp8_recipes.append(nvfp4_rht_and_2d_quantization())
    fp8_recipes.append(nvfp4_row_scaled())
    fp8_recipes.append(nvfp4_4over6())
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


def reset_graphs(
    graphed_callables: Union[Callable, Tuple[Callable, ...], Dict[Tuple[int, int], Callable]],
) -> None:
    """Reset CUDA graphs."""
    if isinstance(graphed_callables, tuple) or isinstance(graphed_callables, list):
        for callable in graphed_callables:
            callable.reset()
    elif isinstance(graphed_callables, dict):
        for callable in graphed_callables.values():
            callable.reset()
    else:
        graphed_callables.reset()


class _Sequential(torch.nn.Sequential):
    """Sequential model that forwards keyword arguments to modules"""

    def forward(self, input_: torch.Tensor, **kwargs) -> torch.Tensor:
        x = input_
        for module in self:
            x = module(x, **kwargs)
        return x


class _GroupedLinearWrapper(torch.nn.Module):
    """Adapt GroupedLinear to the [seqlen, batch, hidden] data used in this test.

    GroupedLinear expects a 2D `[total_tokens, hidden]` input plus an `m_splits`
    list, so this wrapper flattens the leading dims, splits the tokens evenly
    across the GEMMs, and restores the original shape. It also forwards
    `is_first_microbatch` so the FP8 weight-caching path is exercised under CUDA
    graphs.
    """

    def __init__(self, hidden_size: int, num_gemms: int, params_dtype: torch.dtype) -> None:
        super().__init__()
        self.num_gemms = num_gemms
        self.grouped_linear = GroupedLinear(
            num_gemms,
            hidden_size,
            hidden_size,
            device="cuda",
            params_dtype=params_dtype,
        )

    def forward(self, input_: torch.Tensor, **kwargs) -> torch.Tensor:
        seqlen, batch, hidden = input_.shape
        x = input_.reshape(seqlen * batch, hidden)
        total_tokens = x.shape[0]
        assert total_tokens % self.num_gemms == 0, "tokens must split evenly across GEMMs"
        split = total_tokens // self.num_gemms
        m_splits = [split] * self.num_gemms
        out = self.grouped_linear(x, m_splits, **kwargs)
        return out.reshape(seqlen, batch, hidden)


# Supported modules
_test_cuda_graphs_modules: List[str] = [
    # Put linear first to test the case where the cuda context might not be set in
    # creating TMA descriptor for MXFP8 quantization.
    "linear",
    "transformer",
    "layernorm_mlp_nocheckpoint",
    "layernorm_mlp_checkpoint",
    "layernorm_linear",
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
    with quantized_model_init(enabled=fp8_params, recipe=fp8_recipe):
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
        elif module == "layernorm_mlp_nocheckpoint":
            modules = [
                LayerNormMLP(
                    model_config.hidden_size,
                    model_config.hidden_size,
                    params_dtype=dtype,
                    checkpoint=False,
                )
                for _ in range(num_layers)
            ]
        elif module == "layernorm_mlp_checkpoint":
            modules = [
                LayerNormMLP(
                    model_config.hidden_size,
                    model_config.hidden_size,
                    params_dtype=dtype,
                    checkpoint=True,
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
        elif module == "grouped_linear":
            modules = [
                _GroupedLinearWrapper(
                    model_config.hidden_size,
                    num_gemms=2,
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

    outputs = get_outputs(model, output)
    if graph_mode == "full":
        reset_graphs(model)
    elif graph_mode == "individual":
        reset_graphs(modules)
    return outputs


@pytest.mark.parametrize("module", _test_cuda_graphs_modules)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("fp8_params", (False, True))
@pytest.mark.parametrize("fp8_recipe", fp8_recipes + [None], ids=recipe_id)
@pytest.mark.parametrize("backward_override", (None, "high_precision", "dequantized"))
def test_make_graphed_callables(
    *,
    module: str,
    model_config: str = "small",
    num_layers: int = 3,
    dtype: torch.dtype,
    fp8_params: bool,
    fp8_recipe: recipe.Recipe,
    backward_override: str,
    fp8_weight_caching: bool = False,
) -> None:

    fp8 = fp8_recipe is not None

    skip_unsupported_backward_override(module, fp8_recipe, backward_override)
    if fp8:
        fp8_recipe = copy.deepcopy(fp8_recipe)
        fp8_recipe.backward_override = backward_override

    if fp8_params and not fp8:
        pytest.skip("FP8 needed for FP8 parameters.")
    if fp8_weight_caching and not fp8:
        pytest.skip("FP8 needed for FP8 parameters.")
    if fp8 and (fp8_recipe.float8_block_scaling() or fp8_recipe.nvfp4()) and module == "linear_op":
        pytest.skip(
            f"Module not yet supported for {fp8_recipe.__class__.__name__} with CUDA graphs"
        )
    if fp8 and fp8_recipe.nvfp4():
        if getattr(fp8_recipe, "row_scaled_activation", False) and module == "mha":
            pytest.skip("Row-scaled NVFP4 CUDA graph coverage applies to GEMM modules.")
        if dtype not in get_nvfp4_inp_supported_dtypes(fp8_recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe"
                f" {fp8_recipe.__class__.__name__}"
            )
        if fp8_params:
            pytest.skip("NVFP4 params not supported")
    if (
        fp8
        and fp8_recipe.delayed()
        and torch.cuda.get_device_capability() >= (10, 0)
        and module == "layernorm_mlp_checkpoint"
    ):
        pytest.skip(
            "CUDA graphs not supported for LayerNormMLP "
            "with checkpoint=True, SM>=10, "
            "and DelayedScaling recipe"
        )

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
    "transformer",
    "layernorm_mlp_nocheckpoint",
    "layernorm_mlp_checkpoint",
    "layernorm_linear",
    "linear",
    "mha",
]


@pytest.mark.parametrize(
    "module",
    _test_make_graphed_callables_with_fp8_weight_caching_modules,
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("fp8_params", (False, True))
@pytest.mark.parametrize("fp8_recipe", fp8_recipes, ids=recipe_id)
@pytest.mark.parametrize("backward_override", (None, "high_precision", "dequantized"))
def test_make_graphed_callables_with_fp8_weight_caching(
    *,
    module: str,
    dtype: torch.dtype,
    fp8_params: bool,
    fp8_recipe: recipe.Recipe,
    backward_override: str,
) -> None:
    test_make_graphed_callables(
        module=module,
        dtype=dtype,
        fp8_params=fp8_params,
        fp8_recipe=fp8_recipe,
        backward_override=backward_override,
        fp8_weight_caching=True,
    )


# Per-tensor FP8 recipes that support GroupedLinear FP8 weight caching.
_grouped_linear_fp8_weight_caching_recipes = []
if fp8_available:
    _grouped_linear_fp8_weight_caching_recipes.append(recipe.DelayedScaling())
    _grouped_linear_fp8_weight_caching_recipes.append(recipe.Float8CurrentScaling())


@pytest.mark.skipif(not fp8_available, reason="FP8 is not supported")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("fp8_params", (False, True))
@pytest.mark.parametrize("fp8_recipe", _grouped_linear_fp8_weight_caching_recipes, ids=recipe_id)
def test_make_graphed_callables_grouped_linear_with_fp8_weight_caching(
    *,
    dtype: torch.dtype,
    fp8_params: bool,
    fp8_recipe: recipe.Recipe,
    model_config: str = "small",
    num_layers: int = 3,
) -> None:
    """GroupedLinear must thread `is_first_microbatch` into the FP8 weight-update
    skip tensor under CUDA graphs.

    With `fp8_weight_caching` enabled, the graphed and non-graphed runs only match
    when `skip_fp8_weight_update` is propagated for every microbatch. Before the
    fix, GroupedLinear hardcoded it to `None`, so the cached FP8 weights diverged
    from the eager reference. This regresses if that propagation is dropped again.
    """
    config = model_configs[model_config]
    kwargs = dict(
        module="grouped_linear",
        model_config=config,
        num_layers=num_layers,
        dtype=dtype,
        fp8=True,
        fp8_params=fp8_params,
        fp8_weight_caching=True,
        fp8_recipe=fp8_recipe,
    )
    graph_outputs_full = _test_cuda_graphs(graph_mode="full", **kwargs)
    graph_outputs_individual = _test_cuda_graphs(graph_mode="individual", **kwargs)
    outputs = _test_cuda_graphs(graph_mode="none", **kwargs)

    assert_all_equal(outputs, graph_outputs_full)
    assert_all_equal(outputs, graph_outputs_individual)


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
            enabled=False,
        )

    # Forward and backward passes.
    for _ in range(3):
        inputs = generate_data_for_dot_product_attention(model_config, dtype)
        grad_output = generate_data(model_config, dtype, requires_grad=False)
        output = model(*inputs)
        output.backward(grad_output)

    outputs = get_outputs(model, output)
    if with_graph:
        reset_graphs(model)
    return outputs


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

    outputs = get_outputs(model, output)
    if with_graph:
        reset_graphs(model)
    return outputs


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


def test_make_graphed_callables_returns_owned_parameter_grads() -> None:
    """Parameter grads returned from graph replay must not alias static graph buffers."""
    reset_rng_states()
    model_config = model_configs["small"]
    dtype = torch.float32
    model = torch.nn.Linear(
        model_config.hidden_size,
        model_config.hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
    )
    model = make_graphed_callables(
        model,
        (generate_data(model_config, dtype, warmup=True, requires_grad=False),),
    )

    seen_grads = []

    def save_grad(grad):
        seen_grads.append(grad)
        return grad

    hook = model.weight.register_hook(save_grad)
    try:
        output = model(generate_data(model_config, dtype, requires_grad=False))
        output.backward(generate_data(model_config, dtype, requires_grad=False))

        assert len(seen_grads) == 1
        first_grad = seen_grads[0]
        first_grad_ptr = first_grad.data_ptr()
        first_grad_snapshot = first_grad.clone()

        model.zero_grad(set_to_none=True)

        output = model(generate_data(model_config, dtype, requires_grad=False))
        output.backward(generate_data(model_config, dtype, requires_grad=False))

        assert len(seen_grads) == 2
        assert first_grad.data_ptr() == first_grad_ptr
        assert seen_grads[1].data_ptr() != first_grad_ptr
        torch.testing.assert_close(first_grad, first_grad_snapshot, rtol=0, atol=0)
    finally:
        hook.remove()
        reset_graphs(model)


def test_make_graphed_callables_accumulates_owned_parameter_grads() -> None:
    """Parameter grad accumulation must not reuse overwritten static graph buffers."""
    reset_rng_states()
    model_config = model_configs["small"]
    dtype = torch.float32
    model = torch.nn.Linear(
        model_config.hidden_size,
        model_config.hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
    )
    model = make_graphed_callables(
        model,
        (generate_data(model_config, dtype, warmup=True, requires_grad=False),),
    )

    input_1 = generate_data(model_config, dtype, requires_grad=False)
    grad_1 = generate_data(model_config, dtype, requires_grad=False)
    input_2 = generate_data(model_config, dtype, requires_grad=False)
    grad_2 = generate_data(model_config, dtype, requires_grad=False)
    expected_grad = torch.einsum("...o,...i->oi", grad_1, input_1) + torch.einsum(
        "...o,...i->oi", grad_2, input_2
    )

    try:
        model.zero_grad(set_to_none=True)
        model(input_1).backward(grad_1)
        model(input_2).backward(grad_2)
        torch.testing.assert_close(model.weight.grad, expected_grad, rtol=0, atol=0)
    finally:
        reset_graphs(model)


def test_make_graphed_callables_preserves_skipped_parameter_grad_alias() -> None:
    """Delayed-wgrad parameters are excluded from returned-grad clone handling."""
    reset_rng_states()
    model_config = model_configs["small"]
    dtype = torch.float32
    model = torch.nn.Linear(
        model_config.hidden_size,
        model_config.hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
    )
    model.weight.skip_backward_post_hook = True
    model = make_graphed_callables(
        model,
        (generate_data(model_config, dtype, warmup=True, requires_grad=False),),
    )

    seen_grads = []

    def save_grad(grad):
        seen_grads.append(grad)
        return grad

    hook = model.weight.register_hook(save_grad)
    try:
        output = model(generate_data(model_config, dtype, requires_grad=False))
        output.backward(generate_data(model_config, dtype, requires_grad=False))

        assert len(seen_grads) == 1
        first_grad_ptr = seen_grads[0].data_ptr()

        model.zero_grad(set_to_none=True)

        output = model(generate_data(model_config, dtype, requires_grad=False))
        output.backward(generate_data(model_config, dtype, requires_grad=False))

        assert len(seen_grads) == 2
        assert seen_grads[1].data_ptr() == first_grad_ptr
    finally:
        hook.remove()
        reset_graphs(model)


def test_make_graphed_callables_can_skip_returned_parameter_grad_clone() -> None:
    """Parameter grad clone handling can be disabled for callers that manage lifetimes."""
    reset_rng_states()
    model_config = model_configs["small"]
    dtype = torch.float32
    model = torch.nn.Linear(
        model_config.hidden_size,
        model_config.hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
    )
    model = make_graphed_callables(
        model,
        (generate_data(model_config, dtype, warmup=True, requires_grad=False),),
        clone_param_grads_on_return=False,
    )

    seen_grads = []

    def save_grad(grad):
        seen_grads.append(grad)
        return grad

    hook = model.weight.register_hook(save_grad)
    try:
        output = model(generate_data(model_config, dtype, requires_grad=False))
        output.backward(generate_data(model_config, dtype, requires_grad=False))

        assert len(seen_grads) == 1
        first_grad_ptr = seen_grads[0].data_ptr()

        model.zero_grad(set_to_none=True)

        output = model(generate_data(model_config, dtype, requires_grad=False))
        output.backward(generate_data(model_config, dtype, requires_grad=False))

        assert len(seen_grads) == 2
        assert seen_grads[1].data_ptr() == first_grad_ptr
    finally:
        hook.remove()
        reset_graphs(model)


def test_make_graphed_callables_snapshots_parameter_grad_clone_policy() -> None:
    """Parameter grad clone policy is fixed at capture time."""
    reset_rng_states()
    model_config = model_configs["small"]
    dtype = torch.float32
    model = torch.nn.Linear(
        model_config.hidden_size,
        model_config.hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
    )
    model = make_graphed_callables(
        model,
        (generate_data(model_config, dtype, warmup=True, requires_grad=False),),
    )
    model.weight.skip_backward_post_hook = True

    seen_grads = []

    def save_grad(grad):
        seen_grads.append(grad)
        return grad

    hook = model.weight.register_hook(save_grad)
    try:
        output = model(generate_data(model_config, dtype, requires_grad=False))
        output.backward(generate_data(model_config, dtype, requires_grad=False))

        assert len(seen_grads) == 1
        first_grad = seen_grads[0]
        first_grad_ptr = first_grad.data_ptr()
        first_grad_snapshot = first_grad.clone()

        model.zero_grad(set_to_none=True)

        output = model(generate_data(model_config, dtype, requires_grad=False))
        output.backward(generate_data(model_config, dtype, requires_grad=False))

        assert len(seen_grads) == 2
        assert seen_grads[1].data_ptr() != first_grad_ptr
        torch.testing.assert_close(first_grad, first_grad_snapshot, rtol=0, atol=0)
    finally:
        hook.remove()
        reset_graphs(model)


def _make_capture_time_hooks(
    modules: Tuple[torch.nn.Module, ...],
    records: List[Tuple[int, str]],
) -> List[Dict[str, Dict[int, Callable]]]:
    """Make capture-time hooks that record call order."""

    def make_hook(module_idx: int, hook_name: str) -> Callable:
        expected_module = modules[module_idx]

        def hook(module: torch.nn.Module) -> None:
            assert module is expected_module
            assert not torch.cuda.is_current_stream_capturing()
            records.append((module_idx, hook_name))

        return hook

    return [
        {
            "forward_pre_hooks": {0: make_hook(module_idx, "forward_pre_hooks")},
            "forward_hooks": {0: make_hook(module_idx, "forward_hooks")},
            "backward_pre_hooks": {0: make_hook(module_idx, "backward_pre_hooks")},
            "backward_hooks": {0: make_hook(module_idx, "backward_hooks")},
        }
        for module_idx in range(len(modules))
    ]


def test_ordered_warmup_releases_consumed_outputs() -> None:
    """Ordered warmup should only retain outputs until their corresponding backward."""

    class OutputLifetimeModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.previous_output = None

        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            is_warmup = not torch.cuda.is_current_stream_capturing()
            if is_warmup and self.previous_output is not None:
                assert self.previous_output() is None
            output = input_ * 2
            if is_warmup:
                self.previous_output = weakref.ref(output)
            return output

    module = OutputLifetimeModule()
    sample_args = tuple((torch.ones(4, 8, device="cuda", requires_grad=True),) for _ in range(2))
    graphed_callables = make_graphed_callables(
        (module,),
        sample_args,
        num_warmup_iters=2,
        _order=[1, -1, 1, -1],
        _num_layers_per_chunk=[1],
    )
    assert module.previous_output is not None
    assert module.previous_output() is None
    reset_graphs(graphed_callables)


def test_unordered_warmup_releases_consumed_outputs() -> None:
    """Unordered warmup should release each output after its corresponding backward."""

    class OutputLifetimeModule(torch.nn.Module):
        def __init__(self, output_refs: list, module_idx: int) -> None:
            super().__init__()
            self.output_refs = output_refs
            self.module_idx = module_idx
            self.capture_started = False

        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            output = input_ * 2
            if torch.cuda.is_current_stream_capturing():
                self.capture_started = True
            else:
                self.output_refs[self.module_idx] = weakref.ref(output)
            return output

    output_refs = [None, None]
    modules = tuple(OutputLifetimeModule(output_refs, module_idx) for module_idx in range(2))

    def first_module_backward_pre_hook(_module: torch.nn.Module) -> None:
        if not modules[0].capture_started:
            assert output_refs[1] is not None
            assert output_refs[1]() is None

    graphed_callables = make_graphed_callables(
        modules,
        tuple((torch.ones(4, 8, device="cuda", requires_grad=True),) for _ in modules),
        num_warmup_iters=2,
        capture_time_hooks=[
            {"backward_pre_hooks": {0: first_module_backward_pre_hook}},
            None,
        ],
    )
    assert all(output_ref is not None and output_ref() is None for output_ref in output_refs)
    reset_graphs(graphed_callables)


def test_inference_warmup_does_not_retain_outputs() -> None:
    """Inference warmup should release outputs as soon as each forward returns."""

    class OutputLifetimeModule(torch.nn.Module):
        def __init__(self, previous_output: list) -> None:
            super().__init__()
            self.previous_output = previous_output

        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            is_warmup = not torch.cuda.is_current_stream_capturing()
            if is_warmup and self.previous_output[0] is not None:
                assert self.previous_output[0]() is None
            output = input_ * 2
            if is_warmup:
                self.previous_output[0] = weakref.ref(output)
            return output

    previous_output = [None]
    modules = tuple(OutputLifetimeModule(previous_output).eval() for _ in range(2))
    graphed_callables = make_graphed_callables(
        modules,
        tuple((torch.ones(4, 8, device="cuda"),) for _ in modules),
        num_warmup_iters=2,
    )
    assert previous_output[0] is not None
    assert previous_output[0]() is None
    reset_graphs(graphed_callables)


def test_reused_capture_buffers_release_outputs_after_backward() -> None:
    """Capture locals must not keep weak-refed output buffers alive."""

    class OutputLifetimeModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.previous_capture_output = None

        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            if (
                torch.cuda.is_current_stream_capturing()
                and self.previous_capture_output is not None
            ):
                assert self.previous_capture_output() is None
            output = input_ * 2
            if torch.cuda.is_current_stream_capturing():
                self.previous_capture_output = weakref.ref(output)
            return output

    module = OutputLifetimeModule()
    sample_args = tuple((torch.ones(4, 8, device="cuda", requires_grad=True),) for _ in range(2))
    graphed_callables = make_graphed_callables(
        (module,),
        sample_args,
        _order=[1, -1, 1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
    )
    assert module.previous_capture_output is not None
    assert module.previous_capture_output() is None
    reset_graphs(graphed_callables)


def test_reset_releases_only_the_selected_callable() -> None:
    """Reset releases one callable's graph state without retaining its peers."""

    class CaptureOutputModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.capture_output = None

        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            output = input_ * 2
            if torch.cuda.is_current_stream_capturing():
                self.capture_output = weakref.ref(output)
            return output

    modules = tuple(CaptureOutputModule().cuda() for _ in range(2))
    graphed_callables = make_graphed_callables(
        modules,
        tuple((torch.ones(4, device="cuda", requires_grad=True),) for _ in modules),
    )
    capture_outputs = tuple(module.capture_output for module in modules)
    assert all(output is not None and output() is not None for output in capture_outputs)

    graphed_callables[0].reset()
    graphed_callables[0].reset()
    gc.collect()
    assert capture_outputs[0]() is None
    assert capture_outputs[1]() is not None

    output = graphed_callables[1](torch.randn(4, device="cuda", requires_grad=True))
    output.sum().backward()
    del output
    graphed_callables[1].reset()
    gc.collect()
    assert capture_outputs[1]() is None


@pytest.mark.parametrize("with_order", (False, True))
def test_reset_rejects_all_replay_entry_points(with_order: bool) -> None:
    """Reset is idempotent and terminal for forward and backward replay."""

    class TestModule(torch.nn.Module):
        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            return input_ * 2

    module = TestModule().cuda()
    sample_input = torch.ones(4, device="cuda", requires_grad=True)
    graph_options = {}
    if with_order:
        graph_options = {"_order": [1, -1], "_num_layers_per_chunk": [1]}
    graphed_callable = make_graphed_callables(module, (sample_input,), **graph_options)
    output = graphed_callable(torch.randn_like(sample_input, requires_grad=True))
    torch.cuda.synchronize()

    graphed_callable.reset()
    graphed_callable.reset()
    if not with_order:
        # The eager fallback for a different training state is invalid after reset too.
        graphed_callable.eval()

    error = "has been reset and can no longer be used"
    with pytest.raises(RuntimeError, match=error):
        graphed_callable(torch.randn_like(sample_input, requires_grad=True))
    with pytest.raises(RuntimeError, match=error):
        graphed_callable.backward_dw()
    with pytest.raises(RuntimeError, match=error):
        output.sum().backward()


@pytest.mark.parametrize("with_order", (False, True))
def test_make_graphed_callables_with_capture_time_hooks(with_order: bool) -> None:
    """Test capture-time hooks around warmup and graph capture."""
    num_warmup_iters = 2
    modules = (
        torch.nn.Linear(8, 8, device="cuda"),
        torch.nn.Linear(8, 8, device="cuda"),
    )
    sample_args = tuple((torch.ones(4, 8, device="cuda", requires_grad=True),) for _ in modules)
    records = []
    hook_order = [
        (0, "forward_pre_hooks"),
        (0, "forward_hooks"),
        (1, "forward_pre_hooks"),
        (1, "forward_hooks"),
        (1, "backward_pre_hooks"),
        (1, "backward_hooks"),
        (0, "backward_pre_hooks"),
        (0, "backward_hooks"),
    ]

    graphed_callables = make_graphed_callables(
        modules,
        sample_args,
        num_warmup_iters=num_warmup_iters,
        _order=[1, 2, -2, -1] if with_order else None,
        capture_time_hooks=_make_capture_time_hooks(modules, records),
    )

    assert records == hook_order * (num_warmup_iters + 1)

    for graphed in graphed_callables:
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        y = graphed(x)
        y.backward(torch.ones_like(y))
    assert records == hook_order * (num_warmup_iters + 1)
    reset_graphs(graphed_callables)


def _test_cuda_graphs_with_interleaved_pipeline_parallelism(
    *,
    with_graph: bool,
    model_config: ModelConfig,
    dtype: torch.dtype,
    reuse_graph_input_output_buffers: bool = False,
    clone_param_grads_on_return: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
            _reuse_graph_input_output_buffers=reuse_graph_input_output_buffers,
            clone_param_grads_on_return=clone_param_grads_on_return,
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
        output_snapshots = {} if reuse_graph_input_output_buffers else None

        def forward(layer_idx: int, microbatch_idx: int):
            """Helper function for forward steps"""
            idxs = (layer_idx, microbatch_idx)
            outputs[idxs] = layer_forwards[idxs](inputs[idxs])
            if output_snapshots is not None:
                # Reused graph output buffers are only valid until their corresponding backward.
                output_snapshots[idxs] = outputs[idxs].detach().clone()

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

    output_values = output_snapshots if output_snapshots is not None else outputs
    output_values = [y for _, y in sorted(output_values.items())]
    outputs = get_outputs(model, output_values)
    final_weights = [param.detach().clone() for param in model.parameters()]
    if with_graph:
        reset_graphs(layer_forwards)
    return outputs, final_weights


def test_make_graphed_callables_with_interleaved_pipeline_parallelism(
    *,
    model_config: str = "small",
    dtype: torch.dtype = torch.float16,
) -> None:
    """Test CUDA graphs with Megatron-LM interleaved pipeline parallelism."""
    model_config = model_configs[model_config]
    kwargs = dict(model_config=model_config, dtype=dtype)
    outputs, weights = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=False,
        **kwargs,
    )
    graph_outputs, graph_weights = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=True,
        **kwargs,
    )
    assert_all_equal(outputs, graph_outputs)
    assert_all_equal(weights, graph_weights)


def test_make_graphed_callables_with_interleaved_pipeline_parallelism_reused_buffers(
    *,
    model_config: str = "small",
    dtype: torch.dtype = torch.float16,
) -> None:
    """Test CUDA graphs with reused input/output buffers."""
    model_config = model_configs[model_config]
    kwargs = dict(model_config=model_config, dtype=dtype)
    outputs, weights = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=False,
        **kwargs,
    )
    graph_outputs, graph_weights = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=True,
        reuse_graph_input_output_buffers=True,
        **kwargs,
    )
    assert_all_equal(outputs, graph_outputs)
    assert_all_equal(weights, graph_weights)


def test_make_graphed_callables_with_interleaved_pipeline_parallelism_reused_buffers_no_param_grad_clone(
    *,
    model_config: str = "small",
    dtype: torch.dtype = torch.float16,
) -> None:
    """Test reused input/output buffers when returned parameter grad clones are disabled."""
    model_config = model_configs[model_config]
    kwargs = dict(model_config=model_config, dtype=dtype)
    outputs, weights = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=False,
        **kwargs,
    )
    graph_outputs, graph_weights = _test_cuda_graphs_with_interleaved_pipeline_parallelism(
        with_graph=True,
        reuse_graph_input_output_buffers=True,
        clone_param_grads_on_return=False,
        **kwargs,
    )
    assert_all_equal(outputs, graph_outputs)
    assert_all_equal(weights, graph_weights)


def _slot(saved_arena, branch, io_arena, overlap=0, frame=0, warmup=0, user_grad=None):
    """Build one private graph-memory slot used by the focused tests below."""
    if user_grad is None:
        user_grad = io_arena
    return (saved_arena, io_arena, branch, overlap, frame, warmup, user_grad)


def _variant_major_order(base_order, variants, model_chunks=1):
    """Repeat one complete PP/VPP schedule for each mutually exclusive variant."""
    return [
        (1 if chunk > 0 else -1) * (abs(chunk) + variant * model_chunks)
        for variant in range(variants)
        for chunk in base_order
    ]


def test_slot_memory_capture_time_hooks_with_warmup_aliases() -> None:
    """Hooks follow representative warmup order, then every captured branch."""

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, 8, device="cuda"))
            self.forward_calls = 0

        def forward(self, inp):
            self.forward_calls += 1
            return (inp * self.weight).square()

    module = Module()
    modules = (module,) * 3
    samples = tuple((torch.ones(4, 8, device="cuda", requires_grad=True),) for _ in modules)
    records = []
    num_warmup_iters = 2
    hook_names = (
        "forward_pre_hooks",
        "forward_hooks",
        "backward_pre_hooks",
        "backward_hooks",
    )
    # Branches 0 and 2 share a representative. Alias IDs determine warmup order,
    # so branch 1 warms first even though branch 0 is first in the capture order.
    expected_records = [
        (branch, name) for branch in (1, 0) for _ in range(num_warmup_iters) for name in hook_names
    ] + [(branch, name) for branch in range(3) for name in hook_names]
    graphed = make_graphed_callables(
        modules,
        samples,
        num_warmup_iters=num_warmup_iters,
        _order=_variant_major_order([1, -1], 3),
        _num_layers_per_chunk=[1] * 3,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, branch, 1, warmup=alias) for branch, alias in enumerate((1, 0, 1))
        ),
        capture_time_hooks=_make_capture_time_hooks(modules, records),
    )
    try:
        assert records == expected_records
        assert module.forward_calls == 2 * num_warmup_iters + 3
        for step, branch in enumerate((2, 0, 1, 2)):
            module.zero_grad(set_to_none=True)
            inp = (torch.arange(32, device="cuda").reshape(4, 8) / 32 + step).requires_grad_()
            weight = module.weight.detach()
            output = graphed[branch](inp)
            output.sum().backward()
            torch.testing.assert_close(output, (inp.detach() * weight).square())
            torch.testing.assert_close(inp.grad, 2 * inp.detach() * weight.square())
            torch.testing.assert_close(
                module.weight.grad, (2 * inp.detach().square() * weight).sum(0)
            )
        assert records == expected_records
        assert module.forward_calls == 2 * num_warmup_iters + 3
    finally:
        module.zero_grad(set_to_none=True)
        reset_graphs(graphed)


@pytest.mark.parametrize("num_groups", (1, 3))
def test_slot_memory_handles_saved_grouped_tensor_activations(num_groups: int) -> None:
    """Internal grouped wrappers preserve BF16 data and GPU metadata across slot reuse."""
    from transformer_engine.pytorch.quantized_tensor import (
        prepare_for_saving,
        restore_from_func_ctx,
    )
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    class GroupedSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, first_dims, tensor_offsets):
            activation = inp + 1
            grouped = GroupedTensor(
                tuple(inp.shape),
                inp.dtype,
                num_tensors=num_groups,
                data=activation.flatten(),
                first_dims=first_dims.clone(),
                tensor_offsets=tensor_offsets.clone(),
            )
            # Both metadata arrays affect the result and must survive saving, too.
            factor = (
                (grouped.first_dims.square().sum() + grouped.tensor_offsets.sum()).float() / 128
            ).to(inp.dtype)
            output = activation.square() * factor
            tensors, ctx.tensor_objects = prepare_for_saving(grouped)
            ctx.save_for_backward(*tensors)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (grouped,) = restore_from_func_ctx(ctx)
            assert isinstance(grouped, GroupedTensor)
            assert grouped.num_tensors == num_groups
            factor = (
                (grouped.first_dims.square().sum() + grouped.tensor_offsets.sum()).float() / 128
            ).to(grad_output.dtype)
            activation = grouped.rowwise_data.reshape(grad_output.shape)
            return grad_output * (2 * activation) * factor, None, None

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            splits = (8,) if num_groups == 1 else (1, 3, 4)
            self.register_buffer(
                "first_dims", torch.tensor(splits, device="cuda", dtype=torch.int64)
            )
            self.register_buffer(
                "tensor_offsets",
                torch.cat((self.first_dims.new_zeros(1), self.first_dims.cumsum(0) * 8)),
            )

        def forward(self, inp):
            return GroupedSquare.apply(inp, self.first_dims, self.tensor_offsets)

    module = Module()
    samples = tuple(
        (torch.ones(8, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True),)
        for _ in range(2)
    )
    # Verify the same custom autograd function eagerly before observing the slot path.
    eager_input = torch.full_like(samples[0][0], 0.5, requires_grad=True)
    eager_output = module(eager_input)
    eager_output.sum().backward()
    factor = 1.0 if num_groups == 1 else 130 / 128
    torch.testing.assert_close(eager_output, (eager_input.detach() + 1).square() * factor)
    torch.testing.assert_close(eager_input.grad, 2 * (eager_input.detach() + 1) * factor)

    graphed = make_graphed_callables(
        (module,) * 2,
        samples,
        num_warmup_iters=2,
        _order=_variant_major_order([1, -1], 2),
        _num_layers_per_chunk=[1] * 2,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(_slot(0, branch, 1) for branch in range(2)),
    )
    try:
        split_variants = (
            ((8,),) * 4 if num_groups == 1 else ((1, 3, 4), (2, 2, 4), (4, 3, 1), (1, 3, 4))
        )
        for step, (branch, splits) in enumerate(zip((1, 0, 1, 0), split_variants)):
            offsets = [0]
            for count in splits:
                offsets.append(offsets[-1] + count * 8)
            module.first_dims.copy_(module.first_dims.new_tensor(splits))
            module.tensor_offsets.copy_(module.tensor_offsets.new_tensor(offsets))
            factor = (sum(count * count for count in splits) + sum(offsets)) / 128
            inp = torch.full_like(samples[0][0], step / 4, requires_grad=True)
            output = graphed[branch](inp)
            output.sum().backward()
            torch.testing.assert_close(output, (inp.detach() + 1).square() * factor)
            torch.testing.assert_close(inp.grad, 2 * (inp.detach() + 1) * factor)
    finally:
        reset_graphs(graphed)


def test_slot_memory_releases_warmup_saved_activations_before_capture() -> None:
    """Warmup observation must not keep eager activations alive beside the arenas."""
    observed = []

    class Module(torch.nn.Module):
        def forward(self, inp):
            intermediate = inp.sin()
            if not torch.cuda.is_current_stream_capturing():
                observed.append(weakref.ref(intermediate))
            return intermediate.square()

    def check_warmup_release():
        assert len(observed) == 2
        assert all(ref() is None for ref in observed)

    module = Module().cuda()
    sample = (torch.ones(4096, device="cuda", requires_grad=True),)
    graphed = make_graphed_callables(
        module,
        sample,
        num_warmup_iters=2,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 0),),
        post_warmup_hook=check_warmup_release,
    )
    try:
        inp = torch.full((4096,), 0.5, device="cuda", requires_grad=True)
        output = graphed(inp)
        output.sum().backward()
        torch.testing.assert_close(output, inp.detach().sin().square())
        torch.testing.assert_close(inp.grad, 2 * inp.detach().sin() * inp.detach().cos())
    finally:
        reset_graphs(graphed)


def test_graph_capture_contexts_restore_process_state_on_error(monkeypatch) -> None:
    """Capture failures must restore GC and input gradients."""
    gc_was_enabled = gc.isenabled()
    gc.enable()
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: contextlib.nullcontext())
    try:
        with pytest.raises(RuntimeError, match="capture failed"):
            with te_graph._graph_context_wrapper():
                raise RuntimeError("capture failed")
        assert gc.isenabled()
    finally:
        if not gc_was_enabled:
            gc.disable()

    inp = torch.ones(1, requires_grad=True)
    original_grad = torch.full_like(inp, 2.0)
    inp.grad = original_grad
    with pytest.raises(RuntimeError, match="capture failed"):
        with te_graph._none_grad_context_wrapper((inp,)):
            assert inp.grad is None
            raise RuntimeError("capture failed")
    assert inp.grad is original_grad


def test_temporary_forward_hooks_are_removed_on_error() -> None:
    """Warmup failures must not leave hooks installed on user modules."""
    module = torch.nn.Sequential(torch.nn.Identity())

    with pytest.raises(RuntimeError, match="warmup failed"):
        with te_graph._module_forward_hooks(module.modules(), lambda *args: None):
            assert module._forward_hooks
            assert module[0]._forward_hooks
            raise RuntimeError("warmup failed")

    assert not module._forward_hooks
    assert not module[0]._forward_hooks


def test_allocator_settings_guard_restores_once() -> None:
    """Temporary allocator settings have idempotent failure cleanup."""
    settings = []
    guard = te_graph._AllocatorSettingsGuard()

    guard.apply(settings.append, "expandable_segments:False", "expandable_segments:True")
    guard.restore()
    guard.restore()

    assert settings == ["expandable_segments:False", "expandable_segments:True"]


def test_make_graphed_callables_restores_process_state_on_error(monkeypatch) -> None:
    """The public graph API must unwind every process-wide capture mutation."""

    class TestModule(torch.nn.Module):
        def forward(self, inp):
            return inp

    module = TestModule()
    original_call = TestModule.__call__
    fp8_state = object()
    rng_state = object()
    restored_fp8 = []
    restored_rng = []
    allocator_settings = []
    warmup_hooks = []

    monkeypatch.setattr(te_graph, "save_fp8_tensors", lambda *args, **kwargs: fp8_state)
    monkeypatch.setattr(
        te_graph,
        "restore_fp8_tensors",
        lambda modules, state: restored_fp8.append((modules, state)),
    )
    monkeypatch.setattr(te_graph, "graph_safe_rng_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", restored_rng.append)

    def fail_capture(*args, **kwargs):
        assert te_graph.is_graph_capturing()
        kwargs["pre_warmup_hook"]()
        assert warmup_hooks == ["pre"]
        kwargs["_allocator_settings_guard"].apply(
            allocator_settings.append,
            "expandable_segments:False",
            "expandable_segments:True",
        )
        raise RuntimeError("capture failed")

    monkeypatch.setattr(te_graph, "_make_graphed_callables", fail_capture)

    assert not te_graph.is_graph_capturing()
    with pytest.raises(RuntimeError, match="capture failed"):
        te_graph.make_graphed_callables(
            module,
            (torch.ones(1),),
            pre_warmup_hook=lambda: warmup_hooks.append("pre"),
            post_warmup_hook=lambda: warmup_hooks.append("post"),
        )

    assert not te_graph.is_graph_capturing()
    assert TestModule.__call__ is original_call
    assert restored_fp8 == [((module,), fp8_state)]
    assert restored_rng == [rng_state]
    assert allocator_settings == ["expandable_segments:False", "expandable_segments:True"]
    assert warmup_hooks == ["pre", "post"]


def test_make_graphed_callables_restores_wrappers_on_preparation_error(monkeypatch) -> None:
    """Preparation failures before capture starts must also restore global wrappers."""

    class TestModule(torch.nn.Module):
        def forward(self, inp):
            return inp

    module = TestModule()
    original_call = TestModule.__call__
    fp8_state = object()
    restored_fp8 = []

    monkeypatch.setattr(te_graph, "save_fp8_tensors", lambda *args, **kwargs: fp8_state)
    monkeypatch.setattr(
        te_graph,
        "restore_fp8_tensors",
        lambda modules, state: restored_fp8.append((modules, state)),
    )

    def fail_rng_preparation():
        raise RuntimeError("rng preparation failed")

    monkeypatch.setattr(te_graph, "graph_safe_rng_available", fail_rng_preparation)

    with pytest.raises(RuntimeError, match="rng preparation failed"):
        te_graph.make_graphed_callables(module, (torch.ones(1),))

    assert not te_graph.is_graph_capturing()
    assert TestModule.__call__ is original_call
    assert restored_fp8 == [((module,), fp8_state)]


@pytest.mark.parametrize("elements", (0, 4096), ids=("empty", "nonempty"))
def test_slot_memory_variants_share_one_backing(elements: int) -> None:
    """Mutually exclusive variants must use identical slot storage and output addresses."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square().sum().unsqueeze(0) * 3.0

    variants = 5
    module = Module().cuda()
    samples = tuple(
        (torch.ones(elements, device="cuda", requires_grad=True),) for _ in range(variants)
    )
    order = [
        value for variant in reversed(range(variants)) for value in (variant + 1, -variant - 1)
    ]
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=order,
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, variant, 1, warmup=variant) for variant in range(variants)
        ),
    )

    try:
        pool = graphed[0]._te_cuda_graph_allocator_pool
        assert all(graph._te_cuda_graph_allocator_pool is pool for graph in graphed)
        output_ptrs = []
        for graph in graphed:
            # A physical slot is replayed by later logical microbatches after its matching
            # backward has drained, so exercise two complete lifetimes per callable.
            for _ in range(2):
                inp = torch.randn(elements, device="cuda", requires_grad=True)
                output = graph(inp)
                output_ptrs.append(output.data_ptr())
                output.sum().backward()
                torch.testing.assert_close(inp.grad, 6.0 * inp.detach())
        assert len(set(output_ptrs)) == 1
    finally:
        reset_graphs(graphed)


@pytest.mark.parametrize(
    "num_layers_per_chunk",
    ([1, 0], [1, 0, 1], [1, 0, 0, 1]),
    ids=("trailing", "middle", "consecutive-middle"),
)
def test_slot_memory_skips_zero_layer_pipeline_chunks(num_layers_per_chunk) -> None:
    """Pipeline schedule entries without graphable layers must not consume slot metadata."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square()

    module = Module().cuda()
    num_chunks = len(num_layers_per_chunk)
    num_graphable_layers = sum(num_layers_per_chunk)
    samples = tuple(
        (torch.ones(16, device="cuda", requires_grad=True),) for _ in range(num_graphable_layers)
    )
    graphed = make_graphed_callables(
        (module,) * num_graphable_layers,
        samples,
        num_warmup_iters=2,
        _order=[*range(1, num_chunks + 1), *range(-num_chunks, 0)],
        _num_layers_per_chunk=num_layers_per_chunk,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(layer, layer, layer, overlap=layer, warmup=layer)
            for layer in range(num_graphable_layers)
        ),
    )

    try:
        for graph in graphed:
            inp = torch.randn(16, device="cuda", requires_grad=True)
            graph(inp).sum().backward()
            torch.testing.assert_close(inp.grad, 2.0 * inp.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_preserves_aliased_public_outputs() -> None:
    """Slot output arenas retain overlapping public views across CP branches."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            output = inp.square()
            return output, output[4:]

    module = Module().cuda()
    samples = tuple((torch.ones(16, device="cuda", requires_grad=True),) for _ in range(2))
    graphed = make_graphed_callables(
        (module, module),
        samples,
        num_warmup_iters=2,
        _order=[1, 2, -1, -2],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 0), _slot(0, 1, 0, warmup=1)),
    )
    try:
        for graph in graphed:
            inp = torch.randn(16, device="cuda", requires_grad=True)
            output, output_view = graph(inp)
            assert output_view.data_ptr() - output.data_ptr() == 4 * output.element_size()
            assert output_view.data_ptr() < (
                output.data_ptr() + output.numel() * output.element_size()
            )
            (output.sum() + output_view.sum()).backward()
            expected_grad = 2.0 * inp.detach()
            expected_grad[4:] *= 2.0
            torch.testing.assert_close(inp.grad, expected_grad)
            previous = output[4:].clone()
            with torch.no_grad():
                output_view.add_(1.0)
            torch.testing.assert_close(output[4:], previous + 1.0)
    finally:
        reset_graphs(graphed)


@pytest.mark.parametrize(
    "first_mode,second_mode",
    (("independent", "offset8"), ("offset8", "independent"), ("offset4", "offset8")),
)
def test_slot_memory_rejects_branch_output_alias_mismatch(first_mode, second_mode) -> None:
    """CP branches sharing an output slot must expose the same storage aliases."""

    class Module(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def forward(self, inp):
            output = inp.square()
            first = output[:8]
            if self.mode == "independent":
                return first, output[8:].clone()
            offset = 4 if self.mode == "offset4" else 8
            return first, output[offset : offset + 8]

    modules = (Module(first_mode).cuda(), Module(second_mode).cuda())
    samples = tuple((torch.ones(16, device="cuda", requires_grad=True),) for _ in modules)
    graphed = None
    try:
        with pytest.raises(RuntimeError, match="incompatible output storage aliases"):
            graphed = make_graphed_callables(
                modules,
                samples,
                num_warmup_iters=2,
                _order=[1, 2, -1, -2],
                _num_layers_per_chunk=[1, 1],
                _reuse_graph_input_output_buffers=True,
                _graph_memory_slots=(_slot(0, 0, 0), _slot(0, 1, 0, warmup=1)),
            )
        assert not te_graph.is_graph_capturing()
    finally:
        if graphed is not None:
            reset_graphs(graphed)


@pytest.mark.parametrize("state_kind", ("parameter", "buffer"))
def test_slot_memory_preserves_public_outputs_aliased_to_module_state(state_kind) -> None:
    """Persistent module-state views remain outside the slot allocator pool."""

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            state = torch.randn(16, device="cuda")
            if state_kind == "parameter":
                self.state = torch.nn.Parameter(state)
            else:
                self.register_buffer("state", state)

        def forward(self, inp):
            return inp.square(), self.state.view_as(self.state)

    module = Module()
    samples = tuple((torch.ones(16, device="cuda", requires_grad=True),) for _ in range(2))
    graphed = make_graphed_callables(
        (module, module),
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=[1, 2, -1, -2],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 0), _slot(0, 1, 0, warmup=1)),
    )
    try:
        for graph in graphed:
            inp = torch.randn(16, device="cuda", requires_grad=True)
            output, state_output = graph(inp)
            assert (
                state_output.untyped_storage().data_ptr()
                == module.state.untyped_storage().data_ptr()
            )
            (output.sum() + state_output.sum()).backward()
            torch.testing.assert_close(inp.grad, 2.0 * inp.detach())
            if state_kind == "parameter":
                torch.testing.assert_close(module.state.grad, torch.ones_like(module.state))
                module.state.grad = None
    finally:
        reset_graphs(graphed)


def test_slot_memory_rejects_public_cuda_tensor_subclass_outputs() -> None:
    """Unsupported CUDA outputs must fail before graph capture starts."""

    class OutputTensor(torch.Tensor):
        pass

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square().as_subclass(OutputTensor)

    module = Module().cuda()
    sample = (torch.ones(16, device="cuda", requires_grad=True),)
    with pytest.raises(RuntimeError, match="tensor subclasses"):
        make_graphed_callables(
            module,
            sample,
            num_warmup_iters=2,
            _order=[1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(_slot(0, 0, 0),),
        )
    assert not te_graph.is_graph_capturing()


def test_slot_memory_input_staging_respects_overlapping_liveness() -> None:
    """Live microbatches must not overwrite inputs still needed by backward."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square()

    module = Module().cuda()
    samples = tuple((torch.ones(4096, device="cuda", requires_grad=True),) for _ in range(2))
    graphed = make_graphed_callables(
        (module,),
        samples,
        num_warmup_iters=2,
        _order=[1, 1, -1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(
            _slot(0, 0, 0, warmup=0),
            _slot(1, 1, 1, warmup=0),
        ),
    )

    try:
        inp0 = torch.full((4096,), 2.0, device="cuda", requires_grad=True)
        inp1 = torch.full((4096,), 3.0, device="cuda", requires_grad=True)
        out0 = graphed[0](inp0)
        out1 = graphed[1](inp1)
        out0.sum().backward()
        out1.sum().backward()
        torch.testing.assert_close(inp0.grad, 2.0 * inp0.detach())
        torch.testing.assert_close(inp1.grad, 2.0 * inp1.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_rebinds_mutable_sample_args_to_staging_surfaces() -> None:
    """Mutable sample banks expose the exact static inputs selected by slot staging."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square()

    module = Module().cuda()
    shared = torch.ones(4096, device="cuda", requires_grad=True)
    samples = [(shared,), (shared,)]
    graphed = make_graphed_callables(
        (module,),
        samples,
        num_warmup_iters=2,
        _order=[1, 1, -1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(
            _slot(0, 0, 0, warmup=0),
            _slot(1, 1, 1, warmup=0),
        ),
    )

    try:
        assert any(tensor is shared for tensor in (samples[0][0], samples[1][0]))
        assert samples[0][0].data_ptr() != samples[1][0].data_ptr()

        inp0 = torch.full((4096,), 2.0, device="cuda", requires_grad=True)
        inp1 = torch.full((4096,), 3.0, device="cuda", requires_grad=True)
        out0 = graphed[0](inp0)
        out1 = graphed[1](inp1)
        out0.sum().backward()
        out1.sum().backward()
        torch.testing.assert_close(inp0.grad, 2.0 * inp0.detach())
        torch.testing.assert_close(inp1.grad, 2.0 * inp1.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_rejects_warmup_input_alias_merge() -> None:
    """Warmup-plan aliases must preserve the full input-storage alias topology."""

    class Module(torch.nn.Module):
        def forward(self, left, right):
            return left.square() + right.square()

    module = Module().cuda()
    left = torch.ones(4096, device="cuda", requires_grad=True)
    right = torch.ones(4096, device="cuda", requires_grad=True)
    merged = torch.ones(4096, device="cuda", requires_grad=True)

    with pytest.raises(RuntimeError, match="merge distinct input storages"):
        make_graphed_callables(
            (module,),
            ((left, right), (merged, merged)),
            num_warmup_iters=2,
            _order=[1, 1, -1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(
                _slot(0, 0, 0, warmup=0),
                _slot(1, 1, 1, warmup=0),
            ),
        )
    assert not te_graph.is_graph_capturing()


def test_slot_memory_staging_preserves_nonleading_input_aliases() -> None:
    """Staging must not choose a leading-input address used by another argument."""

    class Module(torch.nn.Module):
        def forward(self, left, right):
            return left.square() + right.square()

    module = Module().cuda()
    branch0_left = torch.ones(4096, device="cuda", requires_grad=True)
    branch0_right = torch.ones(4096, device="cuda", requires_grad=True)
    branch1_left = torch.ones(4096, device="cuda", requires_grad=True)
    samples = [
        (branch0_left, branch0_right),
        (branch1_left, branch0_left),
    ]
    graphed = make_graphed_callables(
        (module, module),
        samples,
        num_warmup_iters=2,
        _order=[1, 2, -1, -2],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(
            _slot(0, 0, 0, warmup=0),
            _slot(0, 1, 0, warmup=1),
        ),
    )

    try:
        assert samples[1][0].data_ptr() != samples[1][1].data_ptr()
        left = torch.full((4096,), 2.0, device="cuda", requires_grad=True)
        right = torch.full((4096,), 3.0, device="cuda", requires_grad=True)
        output = graphed[1](left, right)
        torch.testing.assert_close(output, left.detach().square() + right.detach().square())
        output.sum().backward()
        torch.testing.assert_close(left.grad, 2.0 * left.detach())
        torch.testing.assert_close(right.grad, 2.0 * right.detach())
    finally:
        reset_graphs(graphed)


@pytest.mark.parametrize("state_kind", ("parameter", "buffer"))
def test_slot_memory_rejects_user_inputs_aliased_to_module_state(state_kind) -> None:
    """Runtime input staging must not overwrite parameters or buffers."""

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            state = torch.ones(16, device="cuda")
            if state_kind == "parameter":
                self.state = torch.nn.Parameter(state)
            else:
                self.register_buffer("state", state)

        def forward(self, inp):
            return 2.0 * inp + self.state

    module = Module()
    sample = module.state.detach().view_as(module.state).requires_grad_(True)
    with pytest.raises(RuntimeError, match="sharing storage with a module parameter or buffer"):
        make_graphed_callables(
            module,
            (sample,),
            num_warmup_iters=2,
            _order=[1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(_slot(0, 0, 0),),
        )
    assert not te_graph.is_graph_capturing()


@pytest.mark.parametrize("state_kind", ("parameter", "buffer"))
def test_slot_memory_rejects_user_inputs_aliased_to_other_callable_state(state_kind) -> None:
    """Staging candidates must not alias state owned by another graph callable."""

    class StatelessModule(torch.nn.Module):
        def forward(self, inp):
            return inp.square()

    class StatefulModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            state = torch.ones(16, device="cuda")
            if state_kind == "parameter":
                self.state = torch.nn.Parameter(state)
            else:
                self.register_buffer("state", state)

        def forward(self, inp):
            return 2.0 * inp + self.state

    stateless = StatelessModule()
    stateful = StatefulModule()
    cross_callable_alias = stateful.state.detach().view_as(stateful.state).requires_grad_(True)
    independent = torch.ones_like(stateful.state, requires_grad=True)
    with pytest.raises(RuntimeError, match="sharing storage with a module parameter or buffer"):
        make_graphed_callables(
            (stateless, stateful),
            ((cross_callable_alias,), (independent,)),
            num_warmup_iters=2,
            _order=[1, 2, -1, -2],
            _num_layers_per_chunk=[1, 1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(_slot(0, 0, 0), _slot(0, 1, 0, warmup=1)),
        )
    assert not te_graph.is_graph_capturing()


@pytest.mark.parametrize("alias_state", (False, True))
def test_slot_memory_handles_wrapper_subclass_module_state(alias_state) -> None:
    """Alias validation must inspect physical storage owned by wrapper subclasses."""

    class WrapperTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, backing):
            tensor = torch.Tensor._make_wrapper_subclass(
                cls,
                backing.shape,
                strides=backing.stride(),
                storage_offset=0,
                dtype=backing.dtype,
                layout=backing.layout,
                device=backing.device,
                requires_grad=False,
            )
            tensor.backing = backing
            return tensor

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError("WrapperTensor is module state and must not be dispatched")

    class Module(torch.nn.Module):
        def __init__(self, backing):
            super().__init__()
            self.register_buffer("state", WrapperTensor(backing))

        def forward(self, inp):
            return inp.square()

    backing = torch.ones(16, device="cuda")
    module = Module(backing)
    sample = (
        backing.view_as(backing).requires_grad_(True)
        if alias_state
        else torch.ones_like(backing, requires_grad=True)
    )
    if alias_state:
        with pytest.raises(RuntimeError, match="sharing storage with a module parameter or buffer"):
            make_graphed_callables(
                module,
                (sample,),
                num_warmup_iters=2,
                _order=[1, -1],
                _num_layers_per_chunk=[1],
                _reuse_graph_input_output_buffers=True,
                _graph_memory_slots=(_slot(0, 0, 0),),
            )
        assert not te_graph.is_graph_capturing()
        return

    graphed = make_graphed_callables(
        module,
        (sample,),
        num_warmup_iters=2,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 0),),
    )
    try:
        inp = torch.randn_like(backing, requires_grad=True)
        graphed(inp).sum().backward()
        torch.testing.assert_close(inp.grad, 2.0 * inp.detach())
    finally:
        reset_graphs(graphed)


@pytest.mark.parametrize("parameter", (False, True))
def test_slot_memory_empty_tensor_ignores_unrelated_attributes(parameter) -> None:
    value = torch.empty(0, device="cuda")
    if parameter:
        value = torch.nn.Parameter(value)
    value.main_grad = torch.ones(16, device="cuda")

    assert te_graph._tensor_storage_ptrs(value) == set()
    assert te_graph._tensor_storage_identity(value) == value.untyped_storage().data_ptr()


@pytest.mark.skipif(not mxfp8_available, reason="MXFP8 is not supported")
def test_slot_memory_tracks_mxfp8_physical_storage_identity() -> None:
    with quantized_model_init(enabled=True, recipe=recipe.MXFP8BlockScaling()):
        module = Linear(32, 32, device="cuda", params_dtype=torch.bfloat16)
    weight = module.weight
    assert weight.untyped_storage().nbytes() == 0
    fields = ("_rowwise_data", "_columnwise_data", "_rowwise_scale_inv", "_columnwise_scale_inv")
    expected = {
        value.untyped_storage().data_ptr()
        for name in fields
        if (value := getattr(weight, name)) is not None and value.numel() > 0
    }
    assert expected
    assert te_graph._tensor_storage_ptrs(weight) == expected
    assert te_graph._tensor_storage_identity(weight) == frozenset(expected)

    # The placeholder storage stays empty when a real physical allocation changes.
    original = weight._rowwise_data
    try:
        weight._rowwise_data = original.clone()
        assert te_graph._tensor_storage_identity(weight) != frozenset(expected)
        assert weight._rowwise_data.untyped_storage().data_ptr() in te_graph._tensor_storage_ptrs(
            weight
        )
    finally:
        weight._rowwise_data = original
    assert te_graph._tensor_storage_identity(weight) == frozenset(expected)


@pytest.mark.skipif(not mxfp8_available, reason="MXFP8 is not supported")
def test_slot_memory_handles_mxfp8_module_parameters() -> None:
    """Slot bookkeeping must use the physical storage behind MXFP8 parameters."""
    fp8_recipe = recipe.MXFP8BlockScaling()
    with quantized_model_init(enabled=True, recipe=fp8_recipe):
        module = Linear(32, 32, device="cuda", params_dtype=torch.bfloat16)
    for param in module.parameters():
        param.grad = torch.empty_like(param)

    sample = torch.ones(32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (sample,),
        num_warmup_iters=2,
        enabled=True,
        recipe=fp8_recipe,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 0),),
    )
    try:
        inp = torch.randn_like(sample, requires_grad=True)
        with autocast(enabled=True, recipe=fp8_recipe):
            output = graphed(inp)
        output.sum().backward()
        assert torch.isfinite(output).all()
        assert torch.isfinite(inp.grad).all()
    finally:
        for param in module.parameters():
            param.grad = None
        reset_graphs(graphed)


@pytest.mark.skipif(not mxfp8_available, reason="MXFP8 is not supported")
def test_slot_memory_handles_saved_mxfp8_activations() -> None:
    """Slot capture decomposes saved quantized wrappers into arena-backed tensors."""

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(32, 32, device="cuda", params_dtype=torch.bfloat16)
            self.input_quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)

        def forward(self, inp):
            quantized_input = self.input_quantizer(inp)
            return self.linear(quantized_input)

    fp8_recipe = recipe.MXFP8BlockScaling()
    module = Module()
    for param in module.parameters():
        param.grad = torch.empty_like(param)

    variants = 4
    samples = tuple(
        (torch.ones(32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True),)
        for _ in range(variants)
    )
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        enabled=True,
        recipe=fp8_recipe,
        _order=_variant_major_order([1, -1], variants),
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, variant, 1, warmup=variant) for variant in range(variants)
        ),
    )
    try:
        for graph in graphed:
            inp = torch.randn_like(samples[0][0], requires_grad=True)
            with autocast(enabled=True, recipe=fp8_recipe):
                output = graph(inp)
            output.sum().backward()
            assert torch.isfinite(output).all()
            assert torch.isfinite(inp.grad).all()
            for param in module.parameters():
                if param.grad is not None:
                    param.grad.zero_()
    finally:
        for param in module.parameters():
            param.grad = None
        reset_graphs(graphed)


def test_slot_memory_rejects_input_promoted_to_module_state_during_warmup() -> None:
    """Lazy module state must be rechecked before input staging is selected."""

    class LazyStateModule(torch.nn.Module):
        def forward(self, inp):
            if "state" not in self._buffers:
                self.register_buffer("state", inp.detach())
            return 2.0 * inp + self.state

    module = LazyStateModule()
    sample = torch.ones(16, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="module parameter or buffer.*after warmup"):
        make_graphed_callables(
            module,
            (sample,),
            num_warmup_iters=2,
            _order=[1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(_slot(0, 0, 0),),
        )
    assert not te_graph.is_graph_capturing()


def test_slot_memory_rejects_non_tensor_sample_kwargs() -> None:
    """Slot-memory static surfaces must reject non-Tensor kwargs explicitly."""

    class Module(torch.nn.Module):
        def forward(self, inp, use_bias=False):
            output = inp.square()
            if use_bias:
                output = output + 1.0
            return output.sum().unsqueeze(0)

    module = Module().cuda()
    sample = torch.ones(8, device="cuda", requires_grad=True)
    with pytest.raises(TypeError, match="slot memory sample_kwargs must contain only Tensors"):
        make_graphed_callables(
            (module,),
            ((sample,),),
            sample_kwargs=({"use_bias": True},),
            num_warmup_iters=1,
            _order=[1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(_slot(0, 0, 0),),
        )
    assert not te_graph.is_graph_capturing()


def test_slot_memory_snapshots_shared_kwarg_across_alternate_liveness() -> None:
    """A later forward must not overwrite a shared kwarg needed by backward."""

    class Module(torch.nn.Module):
        def forward(self, inp, scale):
            return inp * scale

    module = Module().cuda()
    samples = tuple((torch.ones(4096, device="cuda", requires_grad=True),) for _ in range(2))
    shared_scale = torch.ones(4096, device="cuda")
    sample_kwargs = ({"scale": shared_scale}, {"scale": shared_scale})
    graphed = make_graphed_callables(
        (module,),
        samples,
        sample_kwargs=sample_kwargs,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=[1, -1, 1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 0), _slot(1, 1, 1)),
    )

    try:
        inp0 = torch.ones(4096, device="cuda", requires_grad=True)
        inp1 = torch.ones(4096, device="cuda", requires_grad=True)
        out0 = graphed[0](inp0, scale=torch.full_like(inp0, 2.0))
        out1 = graphed[1](inp1, scale=torch.full_like(inp1, 3.0))
        out0.sum().backward()
        out1.sum().backward()
        torch.testing.assert_close(inp0.grad, torch.full_like(inp0, 2.0))
        torch.testing.assert_close(inp1.grad, torch.full_like(inp1, 3.0))
    finally:
        reset_graphs(graphed)


@pytest.mark.parametrize("reverse_replay", (False, True), ids=("forward", "reverse"))
@pytest.mark.parametrize("debug_slots", (False, True))
def test_slot_memory_reuses_variant_major_branches(
    reverse_replay, debug_slots, monkeypatch
) -> None:
    """Complete CP schedules must reuse one slot backing across variants."""
    monkeypatch.setenv("NVTE_CUDA_GRAPH_SLOT_DEBUG", str(int(debug_slots)))

    class Module(torch.nn.Module):
        def forward(self, inp):
            transient = torch.cat((inp.square(), inp.sin(), inp.cos()), dim=0)
            return transient[: inp.numel()] * 3.0

    variants = 5
    elements = 4096
    module = Module().cuda()
    samples = tuple(
        (torch.ones(elements, device="cuda", requires_grad=True),) for _ in range(variants)
    )
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=_variant_major_order([1, -1], variants),
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, variant, 1, warmup=variant) for variant in range(variants)
        ),
    )

    try:
        pool = graphed[0]._te_cuda_graph_allocator_pool
        assert all(graph._te_cuda_graph_allocator_pool is pool for graph in graphed)
        replay_order = [1, 0, 2, 3, 4] if reverse_replay else range(variants)
        for variant in replay_order:
            graph = graphed[variant]
            inp = torch.randn(elements, device="cuda", requires_grad=True)
            output = graph(inp)
            output.sum().backward()
            expected = 6.0 * inp.detach()
            torch.testing.assert_close(
                inp.grad,
                expected,
                msg=lambda message: f"reverse_replay={reverse_replay}, variant={variant}: {message}",
            )
    finally:
        reset_graphs(graphed)


@pytest.mark.parametrize("conflict", ("saved", "output"))
def test_slot_memory_debug_rejects_live_branch_reuse(conflict, monkeypatch) -> None:
    """An invalid branch forward must fail before corrupting the first branch's backward."""
    monkeypatch.setenv("NVTE_CUDA_GRAPH_SLOT_DEBUG", "1")

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square()

    module = Module().cuda()
    samples = tuple((torch.ones(32, device="cuda", requires_grad=True),) for _ in range(2))
    slots = (
        _slot(0, 0, 0),
        _slot(0 if conflict == "saved" else 1, 1, 1 if conflict == "saved" else 0),
    )
    graphed = make_graphed_callables(
        (module,) * 2,
        samples,
        num_warmup_iters=2,
        _order=_variant_major_order([1, -1], 2),
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=slots,
    )
    try:
        first = torch.full_like(samples[0][0], 2.0, requires_grad=True)
        second = torch.full_like(samples[1][0], 3.0, requires_grad=True)
        output = graphed[0](first)
        with pytest.raises(RuntimeError, match="still live"):
            graphed[1](second)
        output.sum().backward()
        torch.testing.assert_close(first.grad, 2 * first.detach())
        graphed[1](second).sum().backward()
        torch.testing.assert_close(second.grad, 2 * second.detach())
        with torch.no_grad():
            graphed[0](first)
            graphed[1](second)
    finally:
        reset_graphs(graphed)


def test_slot_memory_debug_rejects_repeated_backward(monkeypatch) -> None:
    """The guard must not replay an old backward after a newer forward reuses its arena."""
    monkeypatch.setenv("NVTE_CUDA_GRAPH_SLOT_DEBUG", "1")

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square()

    sample = torch.ones(32, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        Module().cuda(),
        (sample,),
        num_warmup_iters=2,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 0),),
    )
    try:
        first = torch.full_like(sample, 2.0, requires_grad=True)
        old_output = graphed(first)
        old_output.sum().backward(retain_graph=True)
        second = torch.full_like(sample, 3.0, requires_grad=True)
        new_output = graphed(second)
        with pytest.raises(RuntimeError, match="no longer owns"):
            old_output.sum().backward()
        new_output.sum().backward()
        torch.testing.assert_close(second.grad, 2 * second.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_reclaims_retained_variant_outputs() -> None:
    """A module-held source output must not pin a mutually exclusive branch allocation."""

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.retained_output = None

        def forward(self, inp):
            self.retained_output = inp.square() * 3.0
            return self.retained_output

    variants = 5
    elements = 4096
    module = Module().cuda()
    samples = tuple(
        (torch.ones(elements, device="cuda", requires_grad=True),) for _ in range(variants)
    )
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=_variant_major_order([1, -1], variants),
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, variant, 1, warmup=variant) for variant in range(variants)
        ),
    )

    try:
        output_ptrs = []
        for graph in graphed:
            inp = torch.randn(elements, device="cuda", requires_grad=True)
            output = graph(inp)
            output_ptrs.append(output.data_ptr())
            output.sum().backward()
            torch.testing.assert_close(inp.grad, 6.0 * inp.detach())
        assert len(set(output_ptrs)) == 1
    finally:
        module.retained_output = None
        reset_graphs(graphed)


def test_slot_memory_tracks_native_saved_storage() -> None:
    """Variant-major capture must preserve native autograd saved tensors."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            hidden = inp * 2.0
            return hidden.square()

    variants = 5
    module = Module().cuda()
    samples = tuple((torch.ones(4096, device="cuda", requires_grad=True),) for _ in range(variants))
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        _order=_variant_major_order([1, -1], variants),
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, variant, 1, warmup=variant) for variant in range(variants)
        ),
    )

    try:
        for graph in graphed:
            inp = torch.randn(4096, device="cuda", requires_grad=True)
            graph(inp).sum().backward()
            torch.testing.assert_close(inp.grad, 8.0 * inp.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_tracks_expanded_native_saved_storage() -> None:
    """Arena copies must support saved expanded views with internally overlapping strides."""

    class ExpandedSave(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            saved = (inp[:1] + 3.0).expand_as(inp)
            ctx.save_for_backward(saved)
            return inp + 1.0

        @staticmethod
        def backward(ctx, grad_output):
            (saved,) = ctx.saved_tensors
            return grad_output * saved

    class Module(torch.nn.Module):
        def forward(self, inp):
            return ExpandedSave.apply(inp)

    variants = 2
    module = Module().cuda()
    samples = tuple((torch.ones(4096, device="cuda", requires_grad=True),) for _ in range(variants))
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        _order=_variant_major_order([1, -1], variants),
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, variant, 1, warmup=variant) for variant in range(variants)
        ),
    )

    try:
        for graph in graphed:
            inp = torch.randn(4096, device="cuda", requires_grad=True)
            graph(inp).sum().backward()
            torch.testing.assert_close(inp.grad, torch.full_like(inp, inp[0].detach() + 3.0))
    finally:
        reset_graphs(graphed)


def test_slot_memory_variants_reuse_native_saved_allocations() -> None:
    """Additional variants must reuse native saved-tensor allocations, not grow the pool."""

    class Module(torch.nn.Module):
        def __init__(self, canonical):
            super().__init__()
            self.canonical = canonical

        def forward(self, inp):
            if self.canonical:
                return inp.square()
            hidden = inp.sin()
            return hidden.square()

    def capture(variants):
        microbatches = 2
        modules = tuple(Module(variant == 0).cuda() for variant in range(variants))
        samples = tuple(
            (torch.ones(4096, device="cuda", requires_grad=True),)
            for _ in range(variants * microbatches)
        )
        graphed = make_graphed_callables(
            modules,
            samples,
            num_warmup_iters=2,
            _order=_variant_major_order([1, 1, -1, -1], variants),
            _num_layers_per_chunk=[1] * variants,
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=tuple(
                _slot(
                    microbatch,
                    variant * microbatches + microbatch,
                    microbatch,
                    warmup=variant,
                )
                for variant in range(variants)
                for microbatch in range(microbatches)
            ),
        )
        pool = graphed[0]._te_cuda_graph_allocator_pool
        snapshot = pool.snapshot(include_traces=False)
        segments = snapshot["segments"] if isinstance(snapshot, dict) else snapshot
        return graphed, sum(segment["total_size"] for segment in segments)

    baseline, baseline_pool_bytes = capture(2)
    try:
        for graph_idx, graph in enumerate(baseline):
            inp = torch.randn(4096, device="cuda", requires_grad=True)
            graph(inp).sum().backward()
            expected = (
                2.0 * inp.detach()
                if graph_idx // 2 == 0
                else 2.0 * inp.detach().sin() * inp.detach().cos()
            )
            torch.testing.assert_close(inp.grad, expected)
    finally:
        reset_graphs(baseline)

    graphed, variant_pool_bytes = capture(5)
    try:
        for graph_idx, graph in enumerate(graphed):
            inp = torch.randn(4096, device="cuda", requires_grad=True)
            graph(inp).sum().backward()
            expected = (
                2.0 * inp.detach()
                if graph_idx // 2 == 0
                else 2.0 * inp.detach().sin() * inp.detach().cos()
            )
            torch.testing.assert_close(inp.grad, expected)
        assert variant_pool_bytes == baseline_pool_bytes
    finally:
        reset_graphs(graphed)


def test_slot_memory_returns_correct_parameter_gradients() -> None:
    """Mutually exclusive branches must return correct parameter gradients."""

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4096, device="cuda"))

        def forward(self, inp):
            return inp * self.weight

    variants = 5
    module = Module()
    samples = tuple((torch.ones_like(module.weight, requires_grad=True),) for _ in range(variants))
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=_variant_major_order([1, -1], variants),
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, variant, 1, warmup=variant) for variant in range(variants)
        ),
    )

    try:
        for graph in graphed:
            module.weight.grad = None
            inp = torch.randn_like(module.weight, requires_grad=True)
            graph(inp).sum().backward()
            torch.testing.assert_close(inp.grad, module.weight.detach())
            torch.testing.assert_close(module.weight.grad, inp.detach())

        module.weight.grad = None
        accumulated_inputs = []
        for _ in range(2):
            inp = torch.randn_like(module.weight, requires_grad=True)
            graphed[0](inp).sum().backward()
            accumulated_inputs.append(inp.detach())
        torch.testing.assert_close(module.weight.grad, sum(accumulated_inputs))
    finally:
        module.weight.grad = None
        reset_graphs(graphed)


def test_slot_memory_returns_owned_parameter_gradients() -> None:
    """Replaying one physical slot must not overwrite a retained parameter gradient."""

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4096, device="cuda"))

        def forward(self, inp):
            return inp * self.weight

    module = Module()
    samples = ((torch.ones_like(module.weight, requires_grad=True),),)
    graphed = make_graphed_callables(
        (module,),
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 1),),
    )
    retained_grads = []
    hook = module.weight.register_hook(lambda grad: retained_grads.append(grad) or grad)

    try:
        first_input = torch.randn_like(module.weight, requires_grad=True)
        graphed[0](first_input).sum().backward()
        first_grad = retained_grads[0]
        first_grad_snapshot = first_grad.clone()

        module.weight.grad = None
        second_input = torch.randn_like(module.weight, requires_grad=True)
        graphed[0](second_input).sum().backward()

        assert len(retained_grads) == 2
        assert retained_grads[1].data_ptr() != first_grad.data_ptr()
        torch.testing.assert_close(first_grad, first_grad_snapshot, rtol=0, atol=0)
        torch.testing.assert_close(first_grad, first_input.detach(), rtol=0, atol=0)
        torch.testing.assert_close(retained_grads[1], second_input.detach(), rtol=0, atol=0)
    finally:
        hook.remove()
        module.weight.grad = None
        reset_graphs(graphed)


def test_slot_memory_explicit_outputs_share_slot_address() -> None:
    """DCP variants share their explicit output-arena address within a physical slot."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square() * 3.0

    variants = 5
    elements = 4096
    module = Module().cuda()
    samples = tuple(
        (torch.ones(elements, device="cuda", requires_grad=True),) for _ in range(variants)
    )
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=_variant_major_order([1, -1], variants),
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            _slot(0, variant, 1, warmup=variant) for variant in range(variants)
        ),
    )

    try:
        assert not hasattr(graphed[0], "_te_cuda_graph_slot_memory_pool")
        output_ptrs = []
        for graph in graphed:
            inp = torch.randn(elements, device="cuda", requires_grad=True)
            output = graph(inp)
            output_ptrs.append(output.data_ptr())
            output.sum().backward()
            torch.testing.assert_close(inp.grad, 6.0 * inp.detach())
        assert len(set(output_ptrs)) == 1
    finally:
        reset_graphs(graphed)


def test_slot_memory_releases_outputs_before_next_forward_group() -> None:
    """A completed backward group must not pin frame-local outputs into the next forward."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square() * 3.0

    variants = 3
    microbatches = 2
    elements = 4096
    module = Module().cuda()
    samples = tuple(
        (torch.ones(elements, device="cuda", requires_grad=True),)
        for _ in range(variants * microbatches)
    )
    variant_group = [variant + 1 for variant in range(variants)]
    order = [
        *variant_group,
        *(-value for value in variant_group),
        *variant_group,
        *(-value for value in variant_group),
    ]
    slots = tuple(
        _slot(
            microbatch,
            variant * microbatches + microbatch,
            microbatch,
            warmup=variant,
        )
        for variant in range(variants)
        for microbatch in range(microbatches)
    )
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=order,
        _num_layers_per_chunk=[1] * variants,
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=slots,
    )

    try:
        for graph in graphed:
            inp = torch.randn(elements, device="cuda", requires_grad=True)
            graph(inp).sum().backward()
            torch.testing.assert_close(inp.grad, 6.0 * inp.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_releases_transients_across_vpp_tail_backward() -> None:
    """PP/VPP tail backward groups must not inherit transient owners from prior events."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square() * 3.0

    variants = 3
    model_chunks = 2
    microbatches = 4
    elements = 4096
    module = Module().cuda()
    samples = tuple(
        (torch.ones(elements, device="cuda", requires_grad=True),)
        for _ in range(variants * model_chunks * microbatches)
    )
    slots = []
    for variant in range(variants):
        for model_chunk in range(model_chunks):
            logical_chunk = variant * model_chunks + model_chunk
            for microbatch in range(microbatches):
                frame = model_chunk * microbatches + microbatch
                branch = variant * model_chunks * microbatches + frame
                slots.append(
                    _slot(
                        frame,
                        branch,
                        microbatch,
                        overlap=model_chunk,
                        frame=0,
                        warmup=logical_chunk,
                    )
                )

    # PP=2, VPP=2, rank 0, four-microbatch schedule repeated per CP variant.
    base_order = [1, 1, 2, 2, 1, -2, 1, -2, 2, -1, 2, -1, -2, -2, -1, -1]
    order = _variant_major_order(base_order, variants, model_chunks)

    graphed = make_graphed_callables(
        (module,) * (variants * model_chunks),
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=order,
        _num_layers_per_chunk=[1] * (variants * model_chunks),
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(slots),
    )

    try:
        inp = torch.randn(elements, device="cuda", requires_grad=True)
        graphed[0](inp).sum().backward()
        torch.testing.assert_close(inp.grad, 6.0 * inp.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_snapshots_live_inputs_across_slot_wrap() -> None:
    """A drained slot can wrap while another slot's forward remains live."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square() * 3.0

    module = Module().cuda()
    samples = tuple((torch.ones(4096, device="cuda", requires_grad=True),) for _ in range(2))
    graphed = make_graphed_callables(
        (module,),
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=[1, 1, -1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 2), _slot(1, 0, 3)),
    )

    try:
        inp0 = torch.randn(4096, device="cuda", requires_grad=True)
        inp1 = torch.randn(4096, device="cuda", requires_grad=True)
        out0 = graphed[0](inp0)
        out1 = graphed[1](inp1)
        out0.sum().backward()
        # Returned input-grad surfaces are valid until their physical slot wraps.
        torch.testing.assert_close(inp0.grad, 6.0 * inp0.detach())

        inp2 = torch.randn(4096, device="cuda", requires_grad=True)
        out2 = graphed[0](inp2)
        out1.sum().backward()
        torch.testing.assert_close(inp1.grad, 6.0 * inp1.detach())
        out2.sum().backward()
        torch.testing.assert_close(inp2.grad, 6.0 * inp2.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_saved_arenas_cover_alternate_schedule() -> None:
    """Saved tensors must follow union liveness, not only the capture schedule."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            hidden = inp.sin()
            return hidden.square()

    module = Module().cuda()
    samples = tuple((torch.ones(4096, device="cuda", requires_grad=True),) for _ in range(4))
    graphed = make_graphed_callables(
        (module,),
        samples,
        num_warmup_iters=2,
        _order=[1, 1, -1, 1, -1, 1, -1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(_slot(index, index, index) for index in range(4)),
    )

    try:
        inputs = [torch.randn(4096, device="cuda", requires_grad=True) for _ in range(4)]
        outputs = [graphed[index](inputs[index]) for index in range(3)]
        outputs[0].sum().backward()
        torch.testing.assert_close(
            inputs[0].grad, 2.0 * inputs[0].detach().sin() * inputs[0].detach().cos()
        )
        outputs.append(graphed[3](inputs[3]))
        for index in (1, 2, 3):
            outputs[index].sum().backward()
            torch.testing.assert_close(
                inputs[index].grad,
                2.0 * inputs[index].detach().sin() * inputs[index].detach().cos(),
            )
    finally:
        reset_graphs(graphed)


def test_slot_memory_does_not_duplicate_output_backed_saves() -> None:
    """A saved public output must not also consume spill space in its arena."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.sigmoid()

    elements = 4096
    module = Module().cuda()
    samples = ((torch.ones(elements, device="cuda", requires_grad=True),),)
    graphed = make_graphed_callables(
        (module,),
        samples,
        num_warmup_iters=2,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 0),),
    )

    try:
        arenas = graphed[0]._te_cuda_graph_saved_arenas
        assert len(arenas) == 1
        assert next(iter(arenas.values())).numel() == elements * samples[0][0].element_size()

        inp = torch.randn(elements, device="cuda", requires_grad=True)
        graphed[0](inp).sum().backward()
        expected = inp.detach().sigmoid()
        torch.testing.assert_close(inp.grad, expected * (1.0 - expected))
    finally:
        reset_graphs(graphed)


def test_slot_memory_honors_user_grad_liveness_groups() -> None:
    """The private plan may keep adjacent asynchronous gradient consumers disjoint."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return inp.square()

    module = Module().cuda()
    samples = tuple((torch.ones(4096, device="cuda", requires_grad=True),) for _ in range(2))
    graphed = make_graphed_callables(
        (module, module),
        samples,
        num_warmup_iters=2,
        _order=[1, 2, -2, -1],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(
            _slot(0, 0, 0, overlap=0, warmup=0, user_grad=0),
            _slot(1, 1, 0, overlap=1, warmup=1, user_grad=1),
        ),
    )

    try:
        inp = torch.randn(4096, device="cuda", requires_grad=True)
        graphed[1](graphed[0](inp)).sum().backward()
        torch.testing.assert_close(inp.grad, 4.0 * inp.detach().pow(3))
        assert tuple(graphed[-1]._te_cuda_graph_user_grad_arenas) == (0, 1)
    finally:
        reset_graphs(graphed)


def test_slot_memory_distinguishes_reused_allocation_after_checkpoint_discard() -> None:
    """Freed checkpoint storage and a later allocation may share an address, not an arena."""

    class SavedSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, offset, holder):
            activation = inp + offset
            holder.append(activation)
            ctx.save_for_backward(activation)
            return activation.square()

        @staticmethod
        def backward(ctx, grad_output):
            (activation,) = ctx.saved_tensors
            return 2 * activation * grad_output, None, None

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.cuda.MemPool()
            self.reused_storages = []

        def forward(self, inp):
            capturing = torch.cuda.is_current_stream_capturing()
            pool_context = (
                contextlib.nullcontext() if capturing else torch.cuda.use_mem_pool(self.pool)
            )
            first, second = [], []
            with pool_context:
                left = SavedSquare.apply(inp, 1, first)
                original = first[0]
                old_ptr = original.data_ptr()
                original.untyped_storage().resize_(0)
                right = SavedSquare.apply(inp, 2, second)
                replacement = second[0]
                if not capturing:
                    # The isolated allocator has no competing free block of this size.
                    assert old_ptr == replacement.data_ptr()
                    assert original.untyped_storage()._cdata != replacement.untyped_storage()._cdata
                    self.reused_storages.append(old_ptr)
            output = left + right

            def recompute(grad):
                # Model checkpoint recomputation: refill the original StorageImpl before
                # its autograd consumer executes, without bumping the tensor version.
                restored = inp.detach() + 1
                storage = original.untyped_storage()
                storage.resize_(restored.untyped_storage().nbytes())
                storage.copy_(restored.untyped_storage())
                return grad

            output.register_hook(recompute)
            return output

    module = Module().cuda()
    samples = ((torch.ones(16384, device="cuda", requires_grad=True),),)
    graphed = make_graphed_callables(
        (module,),
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 1),),
    )
    try:
        assert len(module.reused_storages) == 2
        for offset in (0.0, 0.5, 2.0):
            inp = (
                torch.arange(16384, device="cuda", dtype=torch.float32) / 16384 + offset
            ).requires_grad_()
            output = graphed[0](inp)
            output.sum().backward()
            torch.testing.assert_close(
                output, (inp.detach() + 1).square() + (inp.detach() + 2).square()
            )
            torch.testing.assert_close(inp.grad, 4 * inp.detach() + 6)
    finally:
        reset_graphs(graphed)


def test_slot_memory_preserves_versions_of_mutated_saved_storage() -> None:
    """Distinct mutation versions of one StorageImpl need independent saved snapshots."""

    class SavedSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx, activation):
            ctx.save_for_backward(activation)
            return activation.square()

        @staticmethod
        def backward(ctx, grad_output):
            (activation,) = ctx.saved_tensors
            return 2 * activation * grad_output

    class Module(torch.nn.Module):
        def forward(self, inp):
            activation = inp + 1
            left = SavedSquare.apply(activation)
            activation.add_(2)
            return left + SavedSquare.apply(activation)

    module = Module().cuda()
    graphed = make_graphed_callables(
        (module,),
        ((torch.ones(4096, device="cuda", requires_grad=True),),),
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 1),),
    )
    try:
        for offset in (0.0, 1.0):
            inp = (
                torch.arange(4096, device="cuda", dtype=torch.float32) / 4096 + offset
            ).requires_grad_()
            output = graphed[0](inp)
            output.sum().backward()
            torch.testing.assert_close(
                output, (inp.detach() + 1).square() + (inp.detach() + 3).square()
            )
            torch.testing.assert_close(inp.grad, 4 * inp.detach() + 8)
    finally:
        reset_graphs(graphed)


def test_slot_memory_coalesces_overlapping_saved_views() -> None:
    """Saved views of one storage should occupy only their byte union in each live slot."""

    class OverlappingSaves(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            backing = torch.cat(tuple(inp + value for value in (1.0, 2.0, 3.0, 4.0)))
            ctx.save_for_backward(backing[: 3 * inp.numel()], backing[inp.numel() :])
            ctx.input_elements = inp.numel()
            return inp + 0.25

        @staticmethod
        def backward(ctx, grad_output):
            first, second = ctx.saved_tensors
            saved = (first[: ctx.input_elements] + second[: ctx.input_elements]) / 2.0
            return grad_output * saved

    class Module(torch.nn.Module):
        def forward(self, inp):
            return OverlappingSaves.apply(inp)

    elements = 4096
    module = Module().cuda()
    samples = tuple((torch.ones(elements, device="cuda", requires_grad=True),) for _ in range(2))
    graphed = make_graphed_callables(
        (module,),
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=[1, 1, -1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 2), _slot(1, 0, 3)),
    )

    try:
        inp0 = torch.randn(elements, device="cuda", requires_grad=True)
        inp1 = torch.randn(elements, device="cuda", requires_grad=True)
        out0 = graphed[0](inp0)
        out1 = graphed[1](inp1)
        out1.sum().backward()
        out0.sum().backward()
        torch.testing.assert_close(inp0.grad, inp0.detach() + 1.5)
        torch.testing.assert_close(inp1.grad, inp1.detach() + 1.5)
    finally:
        reset_graphs(graphed)


def test_slot_memory_preserves_strided_public_outputs() -> None:
    """Static outputs keep their strides after capture drops strong references to them."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            return (inp * 2.0).t()

    module = Module().cuda()
    samples = tuple((torch.randn(4, 8, device="cuda", requires_grad=True),) for _ in range(2))
    graphed = make_graphed_callables(
        (module, module),
        samples,
        allow_unused_input=True,
        _order=[2, -2, 1, -1],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 1), _slot(0, 1, 1, warmup=1)),
    )
    try:
        for graph in graphed:
            inp = torch.randn(4, 8, device="cuda", requires_grad=True)
            output = graph(inp)
            torch.testing.assert_close(output, (inp.detach() * 2.0).t())
            output.sum().backward()
            torch.testing.assert_close(inp.grad, torch.full_like(inp, 2.0))
    finally:
        reset_graphs(graphed)


def test_slot_memory_rejects_delayed_wgrad_order() -> None:
    """Delayed wgrad keeps activations outside save_for_backward, which arenas cannot protect."""
    module = torch.nn.Linear(8, 8, device="cuda")
    samples = ((torch.randn(4, 8, device="cuda", requires_grad=True),),)
    with pytest.raises(ValueError, match="delayed wgrad"):
        make_graphed_callables(
            (module,),
            samples,
            _order=[1, -1, -1.5],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(_slot(0, 0, 1),),
        )


def test_slot_memory_fused_wgrad_without_main_grad_flag_accumulates_zeros() -> None:
    """Without grad_added_to_main_grad, the replayed placeholder wgrad adds only zeros to .grad."""
    dtype = torch.bfloat16
    module = Linear(32, 32, params_dtype=dtype, fuse_wgrad_accumulation=True, device="cuda")
    module.weight.main_grad = torch.zeros_like(module.weight)
    samples = tuple(
        (torch.randn(8, 32, device="cuda", dtype=dtype, requires_grad=True),) for _ in range(2)
    )
    graphed = make_graphed_callables(
        (module, module),
        samples,
        allow_unused_input=True,
        _order=[2, -2, 1, -1],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 1), _slot(0, 1, 1, warmup=1)),
    )
    try:
        for _ in range(2):
            module.weight.grad = None
            module.weight.main_grad.zero_()
            inp = torch.randn(8, 32, device="cuda", dtype=dtype, requires_grad=True)
            graphed[0](inp).sum().backward()
            torch.cuda.synchronize()
            assert module.weight.grad is None or not torch.count_nonzero(module.weight.grad)
            assert torch.count_nonzero(module.weight.main_grad) > 0
    finally:
        reset_graphs(graphed)


def test_slot_memory_preserves_fused_wgrad_hook() -> None:
    """Fused wgrad must retain the parameter's autograd edge during replay."""
    dtype = torch.bfloat16
    module = Linear(
        32,
        32,
        params_dtype=dtype,
        fuse_wgrad_accumulation=True,
        device="cuda",
    )
    module.weight.main_grad = torch.zeros_like(module.weight)
    module.weight.grad_added_to_main_grad = False
    samples = tuple(
        (torch.randn(8, 32, device="cuda", dtype=dtype, requires_grad=True),) for _ in range(2)
    )
    graphed = make_graphed_callables(
        (module, module),
        samples,
        allow_unused_input=True,
        _order=[2, -2, 1, -1],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=(_slot(0, 0, 1), _slot(0, 1, 1, warmup=1)),
    )

    seen_grads = []

    def count_hook(grad):
        seen_grads.append(grad)
        return grad

    hook = module.weight.register_hook(count_hook)
    try:
        for _ in range(2):
            hook_count = len(seen_grads)
            module.weight.grad = None
            module.weight.main_grad.zero_()
            inp = torch.randn(8, 32, device="cuda", dtype=dtype, requires_grad=True)
            graphed[0](inp).sum().backward()
            torch.cuda.synchronize()
            assert len(seen_grads) == hook_count + 1
            assert torch.count_nonzero(module.weight.main_grad) > 0
        assert seen_grads[0].data_ptr() == seen_grads[1].data_ptr()
    finally:
        hook.remove()
        reset_graphs(graphed)
