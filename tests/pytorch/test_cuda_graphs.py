# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Tuple
import pytest

import torch
from transformer_engine.pytorch import (
    DotProductAttention, LayerNormLinear, LayerNormMLP, Linear, make_graphed_callables,
    MultiheadAttention, TransformerLayer, fp8_autocast, fp8_model_init,
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


class ModelConfig:
    def __init__(self, hidden_size, nheads, kv, seq_len):
        self.h = hidden_size
        self.nheads = nheads
        self.kv = kv
        self.s = seq_len

model_configs = {
    "small": ModelConfig(64, 2, 32, 32),
}

modules = ["transformer", "layernorm_mlp", "layernorm_linear", "linear", "mha", "dpa"]

optimizers = [torch.optim.SGD, torch.optim.Adam]

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
        with torch.no_grad():
            t1.masked_fill_(t1.isnan(), 1.0)
            t2.masked_fill_(t2.isnan(), 1.0)
        if not torch.equal(t1, t2):
            failed = True
            failed_tensors += f"    {names[i]}\n" if names is not None else f"    tensor at idx={i}\n"
    assert not failed, "Output mismatches in:\n" + failed_tensors


def generate_data(
    s: int, b: int, h: int, nheads: int, kv: int, dtype: torch.dtype,
    dpa: bool = False, warmup: bool = False, gen_labels: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data."""
    gen_func = torch.ones if warmup else torch.randn
    if dpa:
        inputs = [gen_func(s, b, nheads, kv, device="cuda", requires_grad=True, dtype=dtype) for _ in range(3)]
    else:
        inputs = [gen_func(s, b, h, device="cuda", requires_grad=True, dtype=dtype)]

    if not gen_labels:
        return inputs

    target = torch.randn(s, b, h, device="cuda", dtype=dtype)
    return inputs, target


def get_outputs(model, output):
    """Return grads and params for comparsion."""
    values = []
    for param in model.parameters():
        values.append(param)
        if param.grad is not None:
            values.append(param.grad)
    values.append(output)
    return values


def _test_cuda_graphs(config, bs, num_layers, dtype, fp8, fp8_params, graph, module, optimizer, graph_mode=""):
    """Helper function for test."""
    reset_rng_states()
    FP8GlobalStateManager.reset()
    dpa = module == "dpa"

    with fp8_model_init(enabled=fp8_params):
        # Create modules.
        if module == "transformer":
            modules = [TransformerLayer(
                            config.h,
                            config.h,
                            config.nheads,
                            hidden_dropout=0.0,
                            attention_dropout=0.0,
                            fuse_qkv_params=True,
                            params_dtype=dtype,
                       ) for _ in range(num_layers)]
        elif module == "layernorm_mlp":
            modules = [LayerNormMLP(
                config.h, config.h, params_dtype=dtype
            ) for _ in range(num_layers)]
        elif module == "layernorm_linear":
            modules = [LayerNormLinear(
                config.h, config.h, params_dtype=dtype
            ) for _ in range(num_layers)]
        elif module == "mha":
            modules = [MultiheadAttention(
                            config.h,
                            config.nheads,
                            attention_dropout=0.0,
                            params_dtype=dtype,
                            fuse_qkv_params=True,
                       ) for _ in range(num_layers)]
        elif dpa:
            assert config.h % config.nheads == 0, "Err."
            assert num_layers == 1, "Err."
            modules = [DotProductAttention(
                        config.nheads, config.kv, attention_dropout=0.0
                        ) for _ in range(num_layers)]
        else:
            modules = [Linear(
                config.h, config.h, device="cuda", params_dtype=dtype
            ) for _ in range(num_layers)]

        # Generate model and wrap API to return graphed version.
        if graph:
            # Graph entire module at once.
            if graph_mode == "full":
                model = modules[0] if dpa else torch.nn.Sequential(*modules)
                model = make_graphed_callables(
                        model,
                        generate_data(config.s, bs, config.h, config.nheads, config.kv, dtype, dpa=dpa, warmup=True),
                        num_warmup_iters=10,
                        fp8_enabled=fp8)
            else:
                modules = [make_graphed_callables(
                    module,
                    generate_data(config.s, bs, config.h, config.nheads, config.kv, dtype, dpa=dpa, warmup=True),
                    num_warmup_iters=10,
                    fp8_enabled=fp8) for module in modules]
                model = modules[0] if dpa else torch.nn.Sequential(*modules)
        else:
            model = modules[0] if dpa else torch.nn.Sequential(*modules)

    # Loss function and optimizer.
    loss_fn = torch.nn.MSELoss()
    if not dpa:
        optimizer = optimizer(model.parameters(), lr=0.001)

    # Launch.
    for _ in range(10):
        inputs, target = generate_data(config.s, bs, config.h, config.nheads, config.kv, dtype, dpa=dpa, gen_labels=True)
        with fp8_autocast(enabled=fp8):
            output = model(*inputs)
        loss = loss_fn(output, target)
        loss.backward()
        if not dpa:
            optimizer.step()
            optimizer.zero_grad()

    return get_outputs(model, output)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("num_layers", [1, 10])
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("fp8_params", all_boolean)
@pytest.mark.parametrize("module", modules)
@pytest.mark.parametrize("optimizer", optimizers)
def test_gpt_make_graphed_callables(dtype, bs, model, num_layers, fp8, fp8_params, module, optimizer):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if fp8_params and not fp8:
        pytest.skip("FP8 needed for FP8 parameters.")
    if module == "dpa" and num_layers > 1:
        pytest.skip("Max 1 layer for DPA.")

    config = model_configs[model]

    outputs = _test_cuda_graphs(config, bs, num_layers, dtype, fp8, fp8_params, False, module, optimizer)
    graph_outputs_mode1 = _test_cuda_graphs(config, bs, num_layers, dtype, fp8, fp8_params, True, module, optimizer, graph_mode="full")
    graph_outputs_mode2 = _test_cuda_graphs(config, bs, num_layers, dtype, fp8, fp8_params, True, module, optimizer, graph_mode="individual")

    # Check that results match
    assert_all_equal(outputs, graph_outputs_mode1)
    assert_all_equal(outputs, graph_outputs_mode2)
