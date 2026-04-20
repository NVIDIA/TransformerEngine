#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FSDP2 + FusedAdam compatibility tests.

Run all tests (via torchrun + pytest):
  torchrun -m pytest <this_file> -v --tb=short

Run a single test standalone (for debugging):
  torchrun <this_file> --test <name> --recipe <recipe>

Available --test values:
  fused_adam_fp8_master_weights, fused_adam_fp8_master_weights_no_meta,
  fused_adam_fp8_high_precision_init,
  fused_adam_bf16, fused_adam_fp8_no_master, fused_adam_bf16_store_param_remainders,
  fuse_wgrad_accumulation, dcp_output_parity, dcp_output_parity_async,
  dcp_resharding_save, dcp_resharding_load, safetensors_fp32_export

Available --recipe values:
  DelayedScaling, Float8CurrentScaling, Float8BlockScaling,
  MXFP8BlockScaling, NVFP4BlockScaling

Note: dcp_resharding_save and dcp_resharding_load are two phases of a single
cross-topology test.  Run dcp_resharding_save under a larger world_size first
(e.g. --nproc_per_node=4), then run dcp_resharding_load under a smaller one
(e.g. --nproc_per_node=2).  The orchestration is handled automatically by
test_fsdp2_fused_adam_dcp_resharding in test_torch_fsdp2.py.
"""

import argparse
import functools
import os
import shutil
import pytest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

import transformer_engine.pytorch as te
from transformer_engine.pytorch import QuantizedTensor
import transformer_engine.common.recipe

from fsdp2_utils import get_recipe_from_string, save_custom_attrs, restore_custom_attrs


HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
NUM_LAYERS = 2
SEQ_LEN = 32
BATCH_PER_RANK = 2
NUM_STEPS = 3


def _build_model(
    fp8_init,
    fuse_wgrad_accumulation=False,
    recipe=None,
    use_meta_device=True,
    preserve_high_precision_init_val=False,
    params_dtype=torch.bfloat16,
):
    """Build a Sequential of TransformerLayers, optionally with FP8 init.

    When fp8_init=True and use_meta_device=True (the default), the model is
    created on the meta device to avoid FSDP2 incompatibility with
    QuantizedTensor wrapper subclasses (e.g. MXFP8Tensor) whose storage is
    inaccessible via data_ptr().  Parameters are materialized after FSDP2
    sharding via reset_parameters() in _shard_model().

    When use_meta_device=False, the model is created directly on CUDA.
    This is the legacy path that does NOT work for block-scaling quantized
    tensors (MXFP8, Float8Blockwise, NVFP4) because FSDP2's
    reset_sharded_param() crashes on wrapper subclass tensors with
    data_ptr() == 0.
    """
    if fp8_init:
        ctx = te.quantized_model_init(
            enabled=True,
            recipe=recipe,
            preserve_high_precision_init_val=preserve_high_precision_init_val,
        )
    else:
        from contextlib import nullcontext

        ctx = nullcontext()
    kwargs = dict(
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        fuse_qkv_params=True,
        params_dtype=params_dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    if fp8_init and use_meta_device:
        kwargs["device"] = "meta"
    with ctx:
        model = torch.nn.Sequential(
            *[
                te.TransformerLayer(
                    HIDDEN_SIZE,
                    FFN_HIDDEN_SIZE,
                    NUM_ATTENTION_HEADS,
                    **kwargs,
                )
                for _ in range(NUM_LAYERS)
            ]
        )
    return model


def _shard_model(model, world_size):
    """Apply FSDP2 sharding with save/restore custom attrs.

    If the model was created on the meta device (e.g. for FP8 init),
    parameters are materialized after sharding via reset_parameters().

    restore_custom_attrs is called last so it applies to the final parameter
    objects. For meta-device models, reset_parameters() replaces params via
    module_setattr (base.py:1336-1339), so attrs must be restored afterward.
    """
    has_meta_params = any(p.is_meta for p in model.parameters())
    custom_attrs = save_custom_attrs(model)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    for child in model.children():
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)
    if has_meta_params:
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    # Restore after reset_parameters so attrs land on the final param objects.
    # save_custom_attrs skips private attrs (_*) on QuantizedTensor params;
    # reset_parameters fully reinitializes quantizer state from
    # self.param_init_meta, so no private attrs need restoring.
    restore_custom_attrs(model, custom_attrs)
    return model


def _get_dist_info():
    """Get world_size and device from environment (PG already initialized by session fixture)."""
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    return world_size, device


def test_fused_adam_fp8_master_weights(recipe_name):
    """FusedAdam with master_weights + FSDP2 + quantized_model_init (FP8 params).

    Verifies:
    - Optimizer states are created with correct dtype (float32)
    - Training loop completes without error
    - DTensor wrapping and QuantizedTensor local tensors are preserved
    """
    recipe = get_recipe_from_string(recipe_name)

    world_size, device = _get_dist_info()

    model = _build_model(fp8_init=True, recipe=recipe)
    model = _shard_model(model, world_size)

    # Verify params are DTensors with QuantizedTensor local shards
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} is not DTensor"
    qt_count = sum(
        1
        for _, p in model.named_parameters()
        if isinstance(p, DTensor) and isinstance(p._local_tensor, QuantizedTensor)
    )
    assert qt_count > 0, "No QuantizedTensor local tensors after sharding"

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    # Verify optimizer states
    for param in model.parameters():
        state = optimizer.state[param]
        assert (
            state["exp_avg"].dtype == torch.float32
        ), f"exp_avg dtype {state['exp_avg'].dtype}, expected float32"
        assert (
            state["exp_avg_sq"].dtype == torch.float32
        ), f"exp_avg_sq dtype {state['exp_avg_sq'].dtype}, expected float32"
        if "master_param" in state:
            assert (
                state["master_param"].dtype == torch.float32
            ), f"master_param dtype {state['master_param'].dtype}, expected float32"

    # Verify FP8 params preserved
    qt_count = sum(
        1
        for _, p in model.named_parameters()
        if isinstance(p, DTensor) and isinstance(p._local_tensor, QuantizedTensor)
    )
    assert qt_count > 0, "No QuantizedTensor local tensors after training"


def test_fused_adam_fp8_master_weights_no_meta(recipe_name):
    """FusedAdam with master_weights + FSDP2 + quantized_model_init WITHOUT meta device.

    This is the legacy path that creates quantized params directly on CUDA.
    FSDP2's reset_sharded_param() crashes on block-scaling QuantizedTensor
    wrapper subclasses (data_ptr() == 0). This test documents that failure.

    For per-tensor FP8 (DelayedScaling, Float8CurrentScaling) this works
    because Float8Tensor's storage is accessible via data_ptr().
    """
    recipe = get_recipe_from_string(recipe_name)

    if recipe_name in ("MXFP8BlockScaling", "Float8BlockScaling", "NVFP4BlockScaling"):
        pytest.xfail(
            f"{recipe_name}: FSDP2 without meta-device init crashes on block-scaling "
            "QuantizedTensor wrapper subclasses (data_ptr() == 0). "
            "Use device='meta' + reset_parameters() after sharding."
        )

    world_size, device = _get_dist_info()

    model = _build_model(fp8_init=True, recipe=recipe, use_meta_device=False)
    model = _shard_model(model, world_size)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()


def test_fused_adam_fp8_high_precision_init(recipe_name):
    """FusedAdam with master_weights seeded from high-precision init values.

    Tests the preserve_high_precision_init_val=True path demonstrated in the
    fully_shard.py example:
    1. Model is created with preserve_high_precision_init_val=True on meta device
    2. After FSDP2 sharding + materialization, each QuantizedTensor param has
       a high-precision init value accessible via get_high_precision_init_val()
    3. These values seed the optimizer's FP32 master weights (avoiding FP8
       round-trip precision loss)
    4. Training completes successfully with correct optimizer state dtypes
    """
    recipe = get_recipe_from_string(recipe_name)

    if recipe_name == "NVFP4BlockScaling":
        pytest.xfail(
            f"{recipe_name}: quantized_model_init and FSDP2 is not currently supported, since the "
            "block tensor is dequantized before we flatten it for FSDP2."
        )

    world_size, device = _get_dist_info()

    model = _build_model(
        fp8_init=True,
        recipe=recipe,
        preserve_high_precision_init_val=True,
        params_dtype=torch.float32,
    )
    model = _shard_model(model, world_size)

    # Verify params are DTensors with QuantizedTensor local shards
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} is not DTensor"
    qt_count = sum(
        1
        for _, p in model.named_parameters()
        if isinstance(p, DTensor) and isinstance(p._local_tensor, QuantizedTensor)
    )
    assert qt_count > 0, "No QuantizedTensor local tensors after sharding"

    # Verify high-precision init values exist for all QuantizedTensor params
    hp_val_count = 0
    for name, param in model.named_parameters():
        local = param._local_tensor if isinstance(param, DTensor) else param
        if isinstance(local, QuantizedTensor):
            hp_val = getattr(local, "get_high_precision_init_val", lambda: None)()
            assert (
                hp_val is not None
            ), f"{name}: QuantizedTensor param missing high-precision init value"
            assert (
                hp_val.dtype == torch.float32
            ), f"{name}: HP init val dtype {hp_val.dtype}, expected float32"
            hp_val_count += 1
    assert hp_val_count > 0, "No high-precision init values found"

    # Create optimizer and seed master weights from high-precision init values
    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    for name, param in model.named_parameters():
        optimizer.initialize_state(param, store_param_remainders=False)
        local = param._local_tensor if isinstance(param, DTensor) else param
        hp_val = getattr(local, "get_high_precision_init_val", lambda: None)()
        if hp_val is not None:
            optimizer.set_scaled_state(
                param, "master_param", hp_val.to(device=device, dtype=torch.float32)
            )
            local.clear_high_precision_init_val()

    # Verify high-precision init values are cleared after seeding
    for name, param in model.named_parameters():
        local = param._local_tensor if isinstance(param, DTensor) else param
        if isinstance(local, QuantizedTensor):
            hp_val = getattr(local, "get_high_precision_init_val", lambda: None)()
            assert (
                hp_val is None
            ), f"{name}: high-precision init value not cleared after seeding optimizer"

    # Verify optimizer master weights are float32
    for param in model.parameters():
        state = optimizer.state[param]
        if "master_param" in state:
            assert (
                state["master_param"].dtype == torch.float32
            ), f"master_param dtype {state['master_param'].dtype}, expected float32"

    # Training loop
    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.float32, device=device)
    target = torch.randn_like(x)

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    # Verify optimizer states after training
    for param in model.parameters():
        state = optimizer.state[param]
        assert (
            state["exp_avg"].dtype == torch.float32
        ), f"exp_avg dtype {state['exp_avg'].dtype}, expected float32"
        assert (
            state["exp_avg_sq"].dtype == torch.float32
        ), f"exp_avg_sq dtype {state['exp_avg_sq'].dtype}, expected float32"
        if "master_param" in state:
            assert (
                state["master_param"].dtype == torch.float32
            ), f"master_param dtype {state['master_param'].dtype}, expected float32"

    # Verify FP8 params preserved after training
    qt_count = sum(
        1
        for _, p in model.named_parameters()
        if isinstance(p, DTensor) and isinstance(p._local_tensor, QuantizedTensor)
    )
    assert qt_count > 0, "No QuantizedTensor local tensors after training"


def test_fused_adam_bf16(recipe_name):
    """FusedAdam with master_weights + FSDP2 + bf16 params (no FP8).

    Verifies the non-FP8 DTensor param path in step() works correctly.
    """
    recipe = get_recipe_from_string(recipe_name)

    world_size, device = _get_dist_info()

    model = _build_model(fp8_init=False)
    model = _shard_model(model, world_size)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    losses = []
    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Verify optimizer states are float32
    for param in model.parameters():
        state = optimizer.state[param]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32

    # Verify loss decreased (basic sanity)
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_fused_adam_fp8_no_master(recipe_name):
    """FusedAdam without master_weights + FSDP2 + FP8 params.

    Verifies FusedAdam works with FSDP2 even without master weights enabled.
    """
    recipe = get_recipe_from_string(recipe_name)

    if recipe_name in ("MXFP8BlockScaling", "Float8BlockScaling", "NVFP4BlockScaling"):
        pytest.xfail(
            f"{recipe_name}: FusedAdam without master_weights does not support "
            "block-scaling quantized tensors. Use master_weights=True."
        )

    world_size, device = _get_dist_info()

    model = _build_model(fp8_init=True, recipe=recipe)
    model = _shard_model(model, world_size)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=False,
    )

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    # Verify DTensors preserved
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} lost DTensor wrapping"


def test_fused_adam_bf16_store_param_remainders(recipe_name):
    """FusedAdam with master_weights + store_param_remainders + FSDP2 + bf16 params.

    store_param_remainders stores only the trailing 16 remainder bits (int16)
    instead of full FP32 master params. The FP32 master can be reconstructed
    from BF16 params + int16 remainders. Only works with bf16 params + fp32
    master weights.

    Verifies:
    - Training loop completes without error
    - Optimizer master_param states are int16 (remainder bits)
    - exp_avg and exp_avg_sq are float32
    - Loss decreases (basic sanity)
    """
    recipe = get_recipe_from_string(recipe_name)
    world_size, device = _get_dist_info()

    model = _build_model(fp8_init=False)
    model = _shard_model(model, world_size)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
        store_param_remainders=True,
    )

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    losses = []
    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Verify model params are bf16 (required for store_param_remainders)
    for name, param in model.named_parameters():
        assert (
            param.dtype == torch.bfloat16
        ), f"{name}: param dtype {param.dtype}, expected bfloat16"

    # Verify optimizer states
    for name, param in model.named_parameters():
        state = optimizer.state[param]
        assert (
            state["exp_avg"].dtype == torch.float32
        ), f"{name}: exp_avg dtype {state['exp_avg'].dtype}, expected float32"
        assert (
            state["exp_avg_sq"].dtype == torch.float32
        ), f"{name}: exp_avg_sq dtype {state['exp_avg_sq'].dtype}, expected float32"
        # store_param_remainders stores master_param as int16 remainder bits
        if "master_param" in state:
            assert (
                state["master_param"].dtype == torch.int16
            ), f"{name}: master_param dtype {state['master_param'].dtype}, expected int16"

    # Verify loss decreased (basic sanity)
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


@pytest.mark.xfail(
    reason=(
        "fuse_wgrad_accumulation is incompatible with vanilla FSDP2: "
        "autograd Function.apply unwraps DTensors to local tensors, so "
        "main_grad (set on the DTensor) is inaccessible during backward. "
        "Additionally, the fused wgrad GEMM bypasses FSDP2's reduce-scatter."
    ),
    raises=AttributeError,
    strict=True,
)
def test_fuse_wgrad_accumulation(recipe_name):
    """fuse_wgrad_accumulation=True + FSDP2 -- expected to fail.

    With vanilla FSDP2, PyTorch's autograd Function.apply unwraps DTensor
    inputs to local tensors. The local Float8Tensor inside the autograd
    function does not have the `main_grad` attribute (which is set on the
    DTensor parameter). This causes an AttributeError during backward.

    Additionally, even if main_grad were accessible, fuse_wgrad_accumulation
    writes the gradient directly into main_grad and returns None to autograd,
    bypassing FSDP2's reduce-scatter.
    """
    recipe = get_recipe_from_string(recipe_name)
    world_size, device = _get_dist_info()
    model = _build_model(fp8_init=True, fuse_wgrad_accumulation=True, recipe=recipe)

    # Allocate main_grad buffers on the DTensor params
    for param in model.parameters():
        param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device=param.device)

    model = _shard_model(model, world_size)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
        use_decoupled_grad=True,
    )

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    # This is currently failing during backward because the local Float8Tensor
    # inside the autograd function doesn't have main_grad.
    optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.main_grad.zero_()

    with te.autocast(enabled=True, recipe=recipe):
        output = model(x)

    loss = F.mse_loss(output, target)
    loss.backward()  # Expected to raise AttributeError


def test_safetensors_fp32_export(recipe_name):
    """Export full-precision (FP32) model to safetensors from optimizer master weights.

    Verifies:
    - get_model_state_dict with full_state_dict gathers all params
    - get_optimizer_state_dict with full_state_dict gathers optimizer state
    - FP32 state dict is built from optimizer master weights
    - All saved tensors are float32
    - Saved tensor shapes match expected (unsharded) shapes
    """
    recipe = get_recipe_from_string(recipe_name)
    if recipe_name == "MXFP8BlockScaling":
        pytest.xfail(
            "MXFP8BlockScaling: FusedAdam CUDA kernel does not support "
            "MXFP8 quantized tensors, causing illegal memory access. "
            "Fixed by https://github.com/NVIDIA/TransformerEngine/pull/2789."
        )

    from safetensors.torch import load_file, save_file
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    world_size, device = _get_dist_info()
    model = _build_model(fp8_init=True, recipe=recipe)
    model = _shard_model(model, world_size)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    # Train a few steps.
    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    # Gather full state dicts (all ranks participate).
    full_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_model_state = get_model_state_dict(model, options=full_opts)
    full_opt_state = get_optimizer_state_dict(model, optimizer, options=full_opts)

    rank = int(os.environ.get("RANK", "0"))
    save_path = f"/tmp/te_test_fsdp2_model_fp32_{recipe_name}.safetensors"

    if rank == 0:
        if os.path.exists(save_path):
            os.remove(save_path)

        try:
            fp32_state = {}
            opt_param_states = full_opt_state.get("state", {})

            for key, value in full_model_state.items():
                if key in opt_param_states and "master_param" in opt_param_states[key]:
                    fp32_state[key] = opt_param_states[key]["master_param"].float()
                else:
                    fp32_state[key] = value.float()

            assert len(fp32_state) > 0, "FP32 state dict is empty"

            save_file(fp32_state, save_path)
            loaded = load_file(save_path)

            assert len(loaded) == len(
                fp32_state
            ), f"Loaded {len(loaded)} tensors, expected {len(fp32_state)}"
            for k, v in loaded.items():
                assert v.dtype == torch.float32, f"{k}: expected float32, got {v.dtype}"
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


@pytest.mark.parametrize("async_save", [False, True], ids=["sync", "async"])
def test_dcp_output_parity(recipe_name, async_save):
    """DCP save/load round-trip produces bitwise-identical model outputs.

    1. Builds and trains a model for NUM_STEPS
    2. Runs a forward pass and records the output
    3. Saves model + optimizer state via DCP
    4. Builds a *fresh* model + optimizer (same architecture)
    5. Loads the DCP checkpoint into the fresh model
    6. Runs the same forward pass and asserts outputs are identical
    7. Runs one more training step on both models and asserts outputs still match
    """
    recipe = get_recipe_from_string(recipe_name)

    if recipe_name == "MXFP8BlockScaling":
        pytest.xfail(
            "MXFP8BlockScaling: FusedAdam CUDA kernel does not support "
            "MXFP8 quantized tensors, causing illegal memory access: "
            "/transformer_engine/common/multi_tensor/multi_tensor_apply.cuh:92 in function "
            "multi_tensor_apply: CUDA Error: an illegal memory access was encountered. "
            "Fixed by https://github.com/NVIDIA/TransformerEngine/pull/2789."
        )

    if recipe_name == "NVFP4BlockScaling":
        pytest.xfail(
            "NVFP4BlockScaling: DCP load_state_dict triggers reset_sharded_param() "
            "which calls data_ptr() on NVFP4Tensor wrapper subclass with invalid storage"
        )

    if (
        recipe_name == "Float8BlockScaling"
        and not async_save
        and torch.cuda.get_device_capability()[0] == 12
    ):
        pytest.xfail(
            "Float8BlockScaling is failing on SM120 with RuntimeError: "
            "transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu:534 "
            "in function quantize_transpose_vector_blockwise: Assertion failed: pow2_scale. On "
            "Blackwell and newer, the FP8 block scaling recipe is emulated with MXFP8, which "
            "requires using power of two scaling factors."
        )
    if recipe_name == "Float8BlockScaling" and async_save:
        pytest.xfail(
            "Float8BlockScaling: async DCP save/load round-trip produces different model "
            "outputs — quantization metadata (scales) is not correctly persisted through "
            "async distributed checkpointing. On SM120, additionally fails with pow2_scale "
            "assertion in quantize_transpose_vector_blockwise."
        )

    import torch.distributed.checkpoint as dcp

    world_size, device = _get_dist_info()
    rank = int(os.environ.get("RANK", "0"))
    save_mode = "async" if async_save else "sync"
    checkpoint_dir = f"/tmp/te_test_fsdp2_dcp_parity_{recipe_name}_{save_mode}"

    if rank == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    dist.barrier()

    try:
        # ── Build and train the original model ───────────────────────────
        model = _build_model(fp8_init=True, recipe=recipe)
        model = _shard_model(model, world_size)

        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        target = torch.randn_like(x)

        for _ in range(NUM_STEPS):
            optimizer.zero_grad(set_to_none=True)
            with te.autocast(enabled=True, recipe=recipe):
                output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        # Record reference output from the trained model.
        with torch.no_grad():
            with te.autocast(enabled=True, recipe=recipe):
                ref_output = model(x).clone()

        # ── Save checkpoint ──────────────────────────────────────────────
        if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling):
            # We need to remove the _extra_state keys from the model state dict for
            # DelayedScaling, since otherwise we'll run into an error that the tensor
            # sizes are different. The alternative is a LoadPlanner that dynamically
            # re-sizes the input tensors, see NVIDIA/TransformerEngine#1860 for more
            # details.
            model_state = {
                k: v for k, v in model.state_dict().items() if not k.endswith("_extra_state")
            }
        else:
            model_state = model.state_dict()

        save_state = {"model": model_state, "optimizer": optimizer.state_dict()}

        if not async_save:
            dcp.save(save_state, checkpoint_id=checkpoint_dir)
        else:
            future = dcp.async_save(save_state, checkpoint_id=checkpoint_dir)
            future.result()

        # ── Build a fresh model and load the checkpoint ──────────────────
        model2 = _build_model(fp8_init=True, recipe=recipe)
        model2 = _shard_model(model2, world_size)

        optimizer2 = te.optimizers.FusedAdam(
            model2.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        # Populate optimizer state so load_state_dict has matching structure.
        optimizer2.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            out_tmp = model2(x)
        F.mse_loss(out_tmp, target).backward()
        optimizer2.step()

        if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling):
            model2_state = {
                k: v for k, v in model2.state_dict().items() if not k.endswith("_extra_state")
            }
        else:
            model2_state = model2.state_dict()

        state_to_load = {"model": model2_state, "optimizer": optimizer2.state_dict()}

        dcp.load(state_to_load, checkpoint_id=checkpoint_dir)
        model2.load_state_dict(
            state_to_load["model"],
            strict=(
                False
                if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling)
                else True
            ),
        )
        optimizer2.load_state_dict(state_to_load["optimizer"])

        # ── Verify identical forward-pass output ─────────────────────────
        with torch.no_grad():
            with te.autocast(enabled=True, recipe=recipe):
                loaded_output = model2(x)

        if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling):
            # DelayedScaling stores amax history and scaling factors in _extra_state,
            # which cannot be saved via DCP due to non-deterministic pickle sizes
            # across ranks. The fresh model therefore uses default scaling factors,
            # producing small numerical differences from FP8 re-quantization.
            torch.testing.assert_close(
                loaded_output,
                ref_output,
                rtol=0.05,
                atol=0.1,
                msg=lambda x: f"Fresh model loaded from DCP checkpoint produces different output: {x}",
            )
        else:
            torch.testing.assert_close(
                loaded_output,
                ref_output,
                rtol=0,
                atol=0,
                msg=lambda x: f"Fresh model loaded from DCP checkpoint produces different output: {x}",
            )

        # ── Verify one more training step produces identical results ─────
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            out1 = model(x)
        loss1 = F.mse_loss(out1, target)
        loss1.backward()
        optimizer.step()

        optimizer2.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            out2 = model2(x)
        loss2 = F.mse_loss(out2, target)
        loss2.backward()
        optimizer2.step()

        if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling):
            torch.testing.assert_close(
                out2,
                out1,
                rtol=0.05,
                atol=0.1,
                msg="Training step after DCP load produces different output",
            )
        else:
            torch.testing.assert_close(
                out2, out1, msg="Training step after DCP load produces different output"
            )
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def test_dcp_resharding_save(recipe_name):
    """Phase 1 of the DCP resharding test: train with current world_size and save checkpoint.

    Trains a model for NUM_STEPS, records the forward-pass output, and writes:
    - A DCP checkpoint to /tmp/te_test_fsdp2_dcp_resharding_<recipe>/
    - A reference output tensor to /tmp/te_test_fsdp2_dcp_resharding_<recipe>_ref.pt

    These artifacts are consumed by test_dcp_resharding_load, which runs under
    a *different* world_size (typically half as many ranks) to verify that DCP
    correctly reshards the checkpoint into the new topology.

    The two phases are orchestrated by test_fsdp2_fused_adam_dcp_resharding in
    test_torch_fsdp2.py using two sequential plain torchrun invocations.
    """
    recipe = get_recipe_from_string(recipe_name)

    import torch.distributed.checkpoint as dcp

    world_size, device = _get_dist_info()
    rank = int(os.environ.get("RANK", "0"))
    checkpoint_dir = f"/tmp/te_test_fsdp2_dcp_resharding_{recipe_name}"
    ref_output_path = f"/tmp/te_test_fsdp2_dcp_resharding_{recipe_name}_ref.pt"

    if rank == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        if os.path.exists(ref_output_path):
            os.remove(ref_output_path)
    dist.barrier()

    model = _build_model(fp8_init=True, recipe=recipe)
    model = _shard_model(model, world_size)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    # Fixed seed so the load phase reproduces the exact same input tensor.
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    for _ in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    # Record the reference output before saving.
    with torch.no_grad():
        with te.autocast(enabled=True, recipe=recipe):
            ref_output = model(x).clone().cpu()

    dist.barrier()
    if rank == 0:
        torch.save(ref_output, ref_output_path)

    if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling):
        model_state = {
            k: v for k, v in model.state_dict().items() if not k.endswith("_extra_state")
        }
    else:
        model_state = model.state_dict()

    dcp.save(
        {"model": model_state, "optimizer": optimizer.state_dict()},
        checkpoint_id=checkpoint_dir,
    )
    dist.barrier()


def test_dcp_resharding_load(recipe_name):
    """Phase 2 of the DCP resharding test: load into a different world_size and verify parity.

    Loads the DCP checkpoint written by test_dcp_resharding_save (which ran
    under a larger world_size, e.g. 4 ranks) into a fresh model sharded over
    the current, smaller world_size (e.g. 2 ranks).  Asserts that the model
    output after loading is bitwise-identical to the reference saved in phase 1,
    confirming that DCP resharding correctly reconstructs all parameter shards.
    """
    recipe = get_recipe_from_string(recipe_name)

    import torch.distributed.checkpoint as dcp

    world_size, device = _get_dist_info()
    rank = int(os.environ.get("RANK", "0"))
    checkpoint_dir = f"/tmp/te_test_fsdp2_dcp_resharding_{recipe_name}"
    ref_output_path = f"/tmp/te_test_fsdp2_dcp_resharding_{recipe_name}_ref.pt"

    try:
        model2 = _build_model(fp8_init=True, recipe=recipe)
        model2 = _shard_model(model2, world_size)

        optimizer2 = te.optimizers.FusedAdam(
            model2.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        # Same fixed seed as the save phase to reproduce identical x/target.
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)
        x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        target = torch.randn_like(x)

        # Populate optimizer state so load_state_dict has a matching structure.
        optimizer2.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            out_tmp = model2(x)
        F.mse_loss(out_tmp, target).backward()
        optimizer2.step()

        if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling):
            model2_state = {
                k: v for k, v in model2.state_dict().items() if not k.endswith("_extra_state")
            }
        else:
            model2_state = model2.state_dict()

        state_to_load = {"model": model2_state, "optimizer": optimizer2.state_dict()}
        dcp.load(state_to_load, checkpoint_id=checkpoint_dir)
        model2.load_state_dict(
            state_to_load["model"],
            strict=(
                False
                if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling)
                else True
            ),
        )
        optimizer2.load_state_dict(state_to_load["optimizer"])

        with torch.no_grad():
            with te.autocast(enabled=True, recipe=recipe):
                loaded_output = model2(x).cpu()

        if rank == 0:
            ref_output = torch.load(ref_output_path, weights_only=True)

            if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling):
                torch.testing.assert_close(
                    loaded_output,
                    ref_output,
                    rtol=0.05,
                    atol=0.1,
                    msg=lambda m: f"Resharded model output differs from reference: {m}",
                )
            else:
                torch.testing.assert_close(
                    loaded_output,
                    ref_output,
                    rtol=0,
                    atol=0,
                    msg=lambda m: f"Resharded model output differs from reference: {m}",
                )
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            if os.path.exists(ref_output_path):
                os.remove(ref_output_path)


TESTS = {
    "fused_adam_fp8_master_weights": test_fused_adam_fp8_master_weights,
    "fused_adam_fp8_master_weights_no_meta": test_fused_adam_fp8_master_weights_no_meta,
    "fused_adam_fp8_high_precision_init": test_fused_adam_fp8_high_precision_init,
    "fused_adam_bf16": test_fused_adam_bf16,
    "fused_adam_fp8_no_master": test_fused_adam_fp8_no_master,
    "fused_adam_bf16_store_param_remainders": test_fused_adam_bf16_store_param_remainders,
    "fuse_wgrad_accumulation": test_fuse_wgrad_accumulation,
    "dcp_output_parity": functools.partial(test_dcp_output_parity, async_save=False),
    "dcp_output_parity_async": functools.partial(test_dcp_output_parity, async_save=True),
    "dcp_resharding_save": test_dcp_resharding_save,
    "dcp_resharding_load": test_dcp_resharding_load,
    "safetensors_fp32_export": test_safetensors_fp32_export,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, choices=sorted(TESTS.keys()))
    parser.add_argument(
        "--recipe",
        type=str,
        default="MXFP8BlockScaling",
        help="Quantizer type.",
        choices=[
            "DelayedScaling",
            "Float8CurrentScaling",
            "Float8BlockScaling",
            "MXFP8BlockScaling",
            "NVFP4BlockScaling",
        ],
    )
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    try:
        TESTS[args.test](args.recipe)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
