# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import contextlib
import gc
import weakref
from typing import Callable, Dict, Iterable, List, Tuple, Union
import pytest

import torch
from transformer_engine.pytorch import (
    DotProductAttention,
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
    is_bf16_available,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    _get_cp_p2p_transport_group,
    set_cp_p2p_transport_group,
)
import transformer_engine.pytorch.ops as te_ops
import transformer_engine.pytorch.graph as te_graph
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


def test_cp_p2p_transport_group_override():
    class Group:
        pass

    logical_group = Group()
    transport_group = Group()

    assert _get_cp_p2p_transport_group(logical_group) == (logical_group, False)
    set_cp_p2p_transport_group(logical_group, transport_group)
    assert _get_cp_p2p_transport_group(logical_group) == (transport_group, True)
    set_cp_p2p_transport_group(logical_group, None)
    assert _get_cp_p2p_transport_group(logical_group) == (logical_group, False)

    set_cp_p2p_transport_group(logical_group, transport_group)
    logical_group_ref = weakref.ref(logical_group)
    del logical_group
    gc.collect()
    assert logical_group_ref() is None

    self_transport_group = Group()
    self_transport_group_ref = weakref.ref(self_transport_group)
    set_cp_p2p_transport_group(self_transport_group, self_transport_group)
    del self_transport_group
    gc.collect()
    assert self_transport_group_ref() is None


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
    outputs = get_outputs(model, outputs)
    if with_graph:
        reset_graphs(layer_forwards)
    return outputs


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


def _slot(saved_arena, branch, io_arena, overlap=0, frame=0, warmup=0, user_grad=None):
    """Build one private graph-memory slot used by the focused tests below."""
    if user_grad is None:
        user_grad = io_arena
    return (saved_arena, io_arena, branch, overlap, frame, warmup, user_grad)


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


def test_slot_memory_rejects_non_native_allocator(monkeypatch) -> None:
    """Allocator checkpoints are only defined for the native caching allocator."""

    module = torch.nn.Identity()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "cudaMallocAsync")

    with pytest.raises(RuntimeError, match="requires the native CUDA caching allocator"):
        te_graph._make_graphed_callables(
            module,
            (torch.ones(1),),
            _order=[1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(_slot(0, 0, 0),),
        )


def test_slot_memory_rejects_missing_allocator_api(monkeypatch) -> None:
    """Every private allocator API must be checked before slot preparation."""

    module = torch.nn.Identity()
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.delattr(torch._C, "_has_Standard_Deleter")

    with pytest.raises(RuntimeError, match="_has_Standard_Deleter"):
        te_graph._make_graphed_callables(
            module,
            (torch.ones(1),),
            _order=[1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=(_slot(0, 0, 0),),
        )


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
    """Unsupported CUDA outputs must fail before allocator checkpoint mutation."""

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
def test_slot_memory_checkpoint_reuses_lockstep_branches(reverse_replay) -> None:
    """Lockstep CP branches must restore one live slot boundary between captures."""

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
        _order=[
            *(variant + 1 for variant in range(variants)),
            *(-variant - 1 for variant in range(variants)),
        ],
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
            if not torch.allclose(inp.grad, expected):
                print(
                    "CHECKPOINT_GRAD_MISMATCH",
                    {
                        "reverse_replay": reverse_replay,
                        "variant": variant,
                        "input_ptr": inp.data_ptr(),
                        "output_ptr": output.data_ptr(),
                        "grad_ptr": inp.grad.data_ptr(),
                        "grad_head": inp.grad[:4].tolist(),
                        "expected_head": expected[:4].tolist(),
                        "grad_head_i64": inp.grad[:4].view(torch.int64).tolist(),
                    },
                    flush=True,
                )
            torch.testing.assert_close(
                inp.grad,
                expected,
                msg=lambda message: f"variant={variant}: {message}",
            )
    finally:
        reset_graphs(graphed)


@pytest.mark.parametrize("failure_timing", ("detach", "before", "after"))
def test_slot_memory_checkpoint_rolls_back_restore_failure(monkeypatch, failure_timing) -> None:
    """A failed allocator restore must put the original owners back before raising."""

    class Module(torch.nn.Module):
        def forward(self, inp):
            transient = torch.cat((inp.square(), inp.sin(), inp.cos()), dim=0)
            return transient[: inp.numel()] * 3.0

    module = Module().cuda()
    variants = 2
    samples = tuple((torch.ones(4096, device="cuda", requires_grad=True),) for _ in range(variants))
    slots = tuple(_slot(0, variant, 1, warmup=variant) for variant in range(variants))
    real_set_state = torch._C._cuda_setCheckpointPoolState
    real_detach = te_graph.tex._graph_checkpoint_detach_storage
    detached_storage_impls = []
    original_owner_impls = []
    set_state_calls = 0

    def record_detach(storage_impl_ptr):
        detached_storage_impls.append(storage_impl_ptr)
        real_detach(storage_impl_ptr)
        if failure_timing == "detach" and len(detached_storage_impls) == 1:
            original_owner_impls.extend(detached_storage_impls)
            raise RuntimeError("injected checkpoint detach failure")

    def fail_first_set_state(*args, **kwargs):
        nonlocal set_state_calls
        set_state_calls += 1
        if set_state_calls == 1:
            original_owner_impls.extend(detached_storage_impls)
            if failure_timing == "before":
                raise RuntimeError("injected checkpoint restore failure")
        result = real_set_state(*args, **kwargs)
        if set_state_calls == 1:
            raise RuntimeError("injected checkpoint restore failure")
        return result

    monkeypatch.setattr(te_graph.tex, "_graph_checkpoint_detach_storage", record_detach)
    monkeypatch.setattr(torch._C, "_cuda_setCheckpointPoolState", fail_first_set_state)

    with pytest.raises(
        RuntimeError, match="injected checkpoint (detach|restore) failure"
    ) as exc_info:
        make_graphed_callables(
            (module,) * variants,
            samples,
            num_warmup_iters=2,
            allow_unused_input=True,
            _order=[1, 2, -1, -2],
            _num_layers_per_chunk=[1, 1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=slots,
        )
    assert set_state_calls == (1 if failure_timing == "detach" else 2)
    if failure_timing == "detach":
        original_owner_impls = list(dict.fromkeys(detached_storage_impls))
    assert original_owner_impls
    assert all(
        torch._C._has_Standard_Deleter(storage_impl_ptr)
        for storage_impl_ptr in original_owner_impls
    )

    del exc_info
    gc.collect()
    graphed = make_graphed_callables(
        (module,) * variants,
        samples,
        num_warmup_iters=2,
        allow_unused_input=True,
        _order=[1, 2, -1, -2],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=slots,
    )
    try:
        for graph in graphed:
            inp = torch.randn(4096, device="cuda", requires_grad=True)
            graph(inp).sum().backward()
            torch.testing.assert_close(inp.grad, 6.0 * inp.detach())
    finally:
        reset_graphs(graphed)


def test_slot_memory_checkpoint_reclaims_retained_branch_outputs() -> None:
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
        _order=[
            *(variant + 1 for variant in range(variants)),
            *(-variant - 1 for variant in range(variants)),
        ],
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


def test_slot_memory_checkpoint_tracks_native_saved_storage() -> None:
    """Checkpoint branches must retain allocator owners hidden in autograd saved tensors."""

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
        _order=[
            *(variant + 1 for variant in range(variants)),
            *(-variant - 1 for variant in range(variants)),
        ],
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


def test_slot_memory_fork_reuses_native_saved_allocations() -> None:
    """Forked branches must reuse native saved-tensor allocations, not grow the pool."""

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
        variant_group = [variant + 1 for variant in range(variants)]
        graphed = make_graphed_callables(
            modules,
            samples,
            num_warmup_iters=2,
            _order=[
                *variant_group,
                *(-value for value in variant_group),
                *variant_group,
                *(-value for value in variant_group),
            ],
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

    graphed, forked_pool_bytes = capture(5)
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
        assert forked_pool_bytes == baseline_pool_bytes
    finally:
        reset_graphs(graphed)


def test_slot_memory_checkpoint_aliases_parameter_gradients() -> None:
    """Mutually exclusive branches must return parameter grads from one slot address."""

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
        _order=[
            *(variant + 1 for variant in range(variants)),
            *(-variant - 1 for variant in range(variants)),
        ],
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
    finally:
        module.weight.grad = None
        reset_graphs(graphed)


def test_slot_memory_native_io_aliases_graph_pool_storage() -> None:
    """DCP slot plans default to forked graph-pool I/O aliases."""

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
        _order=[
            *(variant + 1 for variant in range(variants)),
            *(-variant - 1 for variant in range(variants)),
        ],
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

    # PP=2, VPP=2, rank 0, four-microbatch schedule with lockstep CP branches.
    base_order = [1, 1, 2, 2, 1, -2, 1, -2, 2, -1, 2, -1, -2, -2, -1, -1]
    order = []
    for chunk_id in base_order:
        for variant in range(variants):
            remapped = abs(chunk_id) + variant * model_chunks
            order.append(remapped if chunk_id > 0 else -remapped)

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

    hook_calls = 0

    def count_hook(grad):
        nonlocal hook_calls
        hook_calls += 1
        return grad

    hook = module.weight.register_hook(count_hook)
    try:
        for graph in graphed:
            hook_calls = 0
            module.weight.grad = None
            module.weight.main_grad.zero_()
            inp = torch.randn(8, 32, device="cuda", dtype=dtype, requires_grad=True)
            graph(inp).sum().backward()
            torch.cuda.synchronize()
            assert hook_calls == 1
            assert torch.count_nonzero(module.weight.main_grad) > 0
    finally:
        hook.remove()
        reset_graphs(graphed)
