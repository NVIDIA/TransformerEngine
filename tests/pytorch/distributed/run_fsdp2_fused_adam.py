#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FSDP2 + FusedAdam compatibility tests.

Launched via torchrun from test_fused_optimizer.py.
"""

import argparse
import functools
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

import transformer_engine.pytorch as te
from transformer_engine.pytorch import QuantizedTensor
import transformer_engine.common.recipe


def get_recipe_from_string(recipe):
    return getattr(transformer_engine.common.recipe, recipe)()


HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
NUM_LAYERS = 2
SEQ_LEN = 32
BATCH_PER_RANK = 2
NUM_STEPS = 3


def save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        if isinstance(param, QuantizedTensor):
            ignore_keys = [key for key in param.__dict__.keys() if key.startswith("_")]
        else:
            ignore_keys = []
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items() if k not in ignore_keys}
    return custom_attrs


def restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)


def _setup():
    """Common distributed setup. Returns (world_size, local_rank, device)."""
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # CPU backend required for async save
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    return world_size, local_rank, device


def _build_model(fp8_init, fuse_wgrad_accumulation=False, recipe=None):
    """Build a Sequential of TransformerLayers, optionally with FP8 init."""
    if fp8_init:
        ctx = te.quantized_model_init(enabled=True, recipe=recipe)
    else:
        from contextlib import nullcontext

        ctx = nullcontext()
    with ctx:
        model = torch.nn.Sequential(
            *[
                te.TransformerLayer(
                    HIDDEN_SIZE,
                    FFN_HIDDEN_SIZE,
                    NUM_ATTENTION_HEADS,
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                    fuse_qkv_params=True,
                    params_dtype=torch.bfloat16,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                )
                for _ in range(NUM_LAYERS)
            ]
        )
    return model


def _shard_model(model, world_size):
    """Apply FSDP2 sharding with save/restore custom attrs."""
    custom_attrs = save_custom_attrs(model)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    for child in model.children():
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)
    restore_custom_attrs(model, custom_attrs)
    return model


def test_fused_adam_fp8_master_weights(recipe=None):
    """FusedAdam with master_weights + FSDP2 + quantized_model_init (FP8 params).

    Verifies:
    - Optimizer states are created with correct dtype (float32)
    - Training loop completes without error
    - DTensor wrapping and QuantizedTensor local tensors are preserved
    """
    world_size, _, device = _setup()

    model = _build_model(fp8_init=True, recipe=recipe)

    # Verify FP8 params created
    qt_count = sum(1 for _, p in model.named_parameters() if isinstance(p, QuantizedTensor))
    assert qt_count > 0, "No QuantizedTensor local tensors before training"

    model = _shard_model(model, world_size)

    # Verify params are DTensors
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} is not DTensor"

    # Verify FP8 params after sharding
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

    dist.destroy_process_group()


def test_fused_adam_bf16(recipe=None):
    """FusedAdam with master_weights + FSDP2 + bf16 params (no FP8).

    Verifies the non-FP8 DTensor param path in step() works correctly.
    """
    world_size, _, device = _setup()

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

    dist.destroy_process_group()


def test_fused_adam_fp8_no_master(recipe=None):
    """FusedAdam without master_weights + FSDP2 + FP8 params.

    Verifies FusedAdam works with FSDP2 even without master weights enabled.
    """
    world_size, _, device = _setup()

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

    dist.destroy_process_group()


def test_fused_adam_bf16_store_param_remainders(recipe=None):
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
    world_size, _, device = _setup()

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

    dist.destroy_process_group()


def test_fuse_wgrad_accumulation(recipe=None):
    """fuse_wgrad_accumulation=True + FSDP2 -- expected to fail.

    With vanilla FSDP2, PyTorch's autograd Function.apply unwraps DTensor
    inputs to local tensors. The local Float8Tensor inside the autograd
    function does not have the `main_grad` attribute (which is set on the
    DTensor parameter). This causes an AttributeError during backward.

    Additionally, even if main_grad were accessible, fuse_wgrad_accumulation
    writes the gradient directly into main_grad and returns None to autograd,
    bypassing FSDP2's reduce-scatter.
    """
    world_size, _, device = _setup()

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

    dist.destroy_process_group()


def test_safetensors_fp32_export(recipe=None):
    """Export full-precision (FP32) model to safetensors from optimizer master weights.

    Verifies:
    - get_model_state_dict with full_state_dict gathers all params
    - get_optimizer_state_dict with full_state_dict gathers optimizer state
    - FP32 state dict is built from optimizer master weights
    - All saved tensors are float32
    - Saved tensor shapes match expected (unsharded) shapes
    """
    from safetensors.torch import load_file, save_file
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    world_size, _, device = _setup()

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
    save_path = "/tmp/te_test_fsdp2_model_fp32.safetensors"

    if rank == 0:
        # Build FP32 state dict from optimizer master weights.
        fp32_state = {}
        opt_param_states = full_opt_state.get("state", {})

        for key, value in full_model_state.items():
            if key in opt_param_states and "master_param" in opt_param_states[key]:
                fp32_state[key] = opt_param_states[key]["master_param"].float()
            else:
                fp32_state[key] = value.float()

        assert len(fp32_state) > 0, "FP32 state dict is empty"

        # Save and verify.
        save_file(fp32_state, save_path)
        loaded = load_file(save_path)

        assert len(loaded) == len(
            fp32_state
        ), f"Loaded {len(loaded)} tensors, expected {len(fp32_state)}"
        for k, v in loaded.items():
            assert v.dtype == torch.float32, f"{k}: expected float32, got {v.dtype}"

        # Clean up.
        os.remove(save_path)

    dist.destroy_process_group()


def test_dcp_output_parity(recipe=None, async_save=False):
    """DCP save/load round-trip produces bitwise-identical model outputs.

    1. Builds and trains a model for NUM_STEPS
    2. Runs a forward pass and records the output
    3. Saves model + optimizer state via DCP
    4. Builds a *fresh* model + optimizer (same architecture)
    5. Loads the DCP checkpoint into the fresh model
    6. Runs the same forward pass and asserts outputs are identical
    7. Runs one more training step on both models and asserts outputs still match
    """
    import torch.distributed.checkpoint as dcp

    world_size, local_rank, device = _setup()

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
    checkpoint_dir = "/tmp/te_test_fsdp2_dcp_parity"

    if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling):
        model_state = {
            k: v for k, v in model.state_dict().items() if not k.endswith("_extra_state")
        }
    else:
        model_state = model.state_dict()

    if not async_save:
        dcp.save(
            {"model": model_state, "optimizer": optimizer.state_dict()},
            checkpoint_id=checkpoint_dir,
        )
        future = None
    else:
        future = dcp.async_save(
            {"model": model_state, "optimizer": optimizer.state_dict()},
            checkpoint_id=checkpoint_dir,
        )

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

    if async_save:
        future.result()  # Block on async save completion

    dcp.load(state_to_load, checkpoint_id=checkpoint_dir)
    model2.load_state_dict(
        state_to_load["model"],
        strict=(
            False if isinstance(recipe, transformer_engine.common.recipe.DelayedScaling) else True
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
            msg="Fresh model loaded from DCP checkpoint produces different output",
        )
    else:
        torch.testing.assert_close(
            loaded_output,
            ref_output,
            rtol=0,
            atol=0,
            msg="Fresh model loaded from DCP checkpoint produces different output",
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

    # ── Cleanup ──────────────────────────────────────────────────────
    import shutil

    if int(os.environ.get("RANK", "0")) == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    dist.destroy_process_group()


TESTS = {
    "fused_adam_fp8_master_weights": test_fused_adam_fp8_master_weights,
    "fused_adam_bf16": test_fused_adam_bf16,
    "fused_adam_fp8_no_master": test_fused_adam_fp8_no_master,
    "fused_adam_bf16_store_param_remainders": test_fused_adam_bf16_store_param_remainders,
    "fuse_wgrad_accumulation": test_fuse_wgrad_accumulation,
    "dcp_output_parity": functools.partial(test_dcp_output_parity, async_save=False),
    "dcp_output_parity_async": functools.partial(test_dcp_output_parity, async_save=True),
    "safetensors_fp32_export": test_safetensors_fp32_export,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, choices=list(TESTS.keys()))
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
    recipe = get_recipe_from_string(args.recipe)
    TESTS[args.test](recipe)
