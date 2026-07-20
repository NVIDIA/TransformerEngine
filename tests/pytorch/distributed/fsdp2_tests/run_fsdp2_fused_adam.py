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
from transformer_engine.pytorch import HybridQuantizedTensor, QuantizedTensor
from transformer_engine.pytorch.tensor import (
    Float8BlockwiseQTensorStorage,
    Float8TensorStorage,
    MXFP8TensorStorage,
)
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
    created on the meta device so parameters are materialized after FSDP2
    sharding via reset_parameters() in _shard_model().  This ensures the
    sharded parameter format is compatible with the FSDP2 all-gather hooks.

    When use_meta_device=False, the model is created directly on CUDA.
    This only works for per-tensor FP8 (DelayedScaling, Float8CurrentScaling).
    Block-scaling types (MXFP8, Float8Blockwise, NVFP4) fail because their
    FSDP2 all-gather hooks do not support CUDA-initialized parameters.
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


def _shard_model(model, world_size, reshard_after_forward=None):
    """Apply FSDP2 sharding with save/restore custom attrs.

    If the model was created on the meta device (e.g. for FP8 init),
    parameters are materialized after sharding via reset_parameters().

    restore_custom_attrs is called last so it applies to the final parameter
    objects. For meta-device models, reset_parameters() replaces params via
    module_setattr (base.py:1336-1339), so attrs must be restored afterward.

    Parameters
    ----------
    reshard_after_forward : bool, optional
        Passed through to ``fully_shard``. ``None`` (default) keeps FSDP2's
        own default: ``True`` for child modules, ``False`` for the root.
        ``False`` on child modules keeps the full-precision gathered weight
        alive through backward, exercising the iter-2+ buffer-reuse path
        inside the same forward/backward rather than across training steps.
    """
    has_meta_params = any(p.is_meta for p in model.parameters())
    custom_attrs = save_custom_attrs(model)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    shard_kwargs = {"mesh": mesh}
    if reshard_after_forward is not None:
        shard_kwargs["reshard_after_forward"] = reshard_after_forward
    for child in model.children():
        fully_shard(child, **shard_kwargs)
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


def _collective_assert(condition, message):
    """Raise on every rank only after all ranks report structural status."""
    failed = torch.tensor([not condition], dtype=torch.uint8, device="cuda")
    if not condition:
        print(f"[rank {dist.get_rank()}] {message}", flush=True)
    dist.all_reduce(failed, dist.ReduceOp.MAX)
    assert not bool(failed.item()), f"{message}: failed on at least one rank"


def _collective_assert_same_shape(tensor, message):
    """Check tensor ranks/shapes before entering a shape-sensitive collective."""
    # Quantized parameter buffers in these tests are at most 2-D, but leave
    # room for future formats without introducing object collectives.
    layout = torch.full((9,), -1, dtype=torch.int64, device=tensor.device)
    _collective_assert(tensor.ndim < layout.numel(), f"{message}: ndim={tensor.ndim}")
    layout[0] = tensor.ndim
    layout[1 : tensor.ndim + 1] = torch.tensor(
        tensor.shape, dtype=torch.int64, device=tensor.device
    )
    min_layout = layout.clone()
    max_layout = layout.clone()
    dist.all_reduce(min_layout, dist.ReduceOp.MIN)
    dist.all_reduce(max_layout, dist.ReduceOp.MAX)
    _collective_assert(
        torch.equal(min_layout, max_layout),
        f"{message}: tensor shape differs across ranks",
    )


def _record_exact(errors, actual, expected, label):
    """Record an exact mismatch without interrupting later distributed work."""
    if actual is None or expected is None:
        if actual is not expected:
            errors.append(f"{label}: one tensor is None")
        return
    try:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    except (AssertionError, TypeError) as exc:
        errors.append(f"{label}: {exc}")


def _raise_collective_errors(errors, context):
    """Aggregate delayed exact failures after every rank finishes collectives."""
    if errors:
        print(f"[rank {dist.get_rank()}] {context}:\n" + "\n".join(errors), flush=True)
    failed = torch.tensor([bool(errors)], dtype=torch.uint8, device="cuda")
    dist.all_reduce(failed, dist.ReduceOp.MAX)
    assert not bool(failed.item()), f"{context}: exact check failed on at least one rank"


def _check_hybrid_direction_buffers(
    local_sub,
    full_sub,
    expected_type,
    *,
    direction,
    param_name,
    world_size,
    errors,
):
    """Compare one direction's gathered raw data, scale buffers, and metadata."""
    _collective_assert(
        isinstance(local_sub, expected_type),
        f"{param_name}: {direction} local storage is {type(local_sub).__name__}, "
        f"expected {expected_type.__name__}",
    )
    _collective_assert(
        isinstance(full_sub, expected_type),
        f"{param_name}: {direction} full storage is {type(full_sub).__name__}, "
        f"expected {expected_type.__name__}",
    )

    local_buffers, local_meta = local_sub.fsdp_extract_buffers()
    full_buffers, full_meta = full_sub.fsdp_extract_buffers()
    local_count = torch.tensor([len(local_buffers)], dtype=torch.int64, device="cuda")
    min_count = local_count.clone()
    max_count = local_count.clone()
    dist.all_reduce(min_count, dist.ReduceOp.MIN)
    dist.all_reduce(max_count, dist.ReduceOp.MAX)
    _collective_assert(
        min_count.item() == max_count.item() == len(full_buffers),
        f"{param_name}: {direction} buffer count mismatch: local={len(local_buffers)}, "
        f"full={len(full_buffers)}, rank range=({min_count.item()}, {max_count.item()})",
    )

    expected_buffers = []
    for buffer in local_buffers:
        _collective_assert(
            buffer is not None,
            f"{param_name}: {direction} unexpectedly exposed a None FSDP buffer",
        )
        _collective_assert_same_shape(buffer, f"{param_name}: {direction} FSDP buffer")
        gathered = [torch.zeros_like(buffer) for _ in range(world_size)]
        dist.all_gather(gathered, buffer)
        expected_buffers.append(torch.cat(gathered, dim=0))

    for key in ("field_names", "direction"):
        if key in local_meta or key in full_meta:
            if local_meta.get(key) != full_meta.get(key):
                errors.append(
                    f"{param_name}: {direction} metadata {key!r} differs: "
                    f"{full_meta.get(key)!r} != {local_meta.get(key)!r}"
                )
    for index, (actual, expected) in enumerate(zip(full_buffers, expected_buffers)):
        field_names = local_meta.get("field_names", ())
        field = field_names[index] if index < len(field_names) else f"buffer[{index}]"
        _record_exact(errors, actual, expected, f"{param_name}: {direction} {field}")

    # Per-tensor FP8 scale is metadata rather than an FSDP buffer. It must be
    # identical on every shard and preserved on the reconstructed full tensor.
    if isinstance(local_sub, Float8TensorStorage):
        local_scale_inv = getattr(local_sub, "_scale_inv", None)
        _collective_assert(
            isinstance(local_scale_inv, torch.Tensor),
            f"{param_name}: {direction} Float8 storage has no tensor _scale_inv",
        )
        local_scale = local_scale_inv.detach().clone()
        _collective_assert_same_shape(local_scale, f"{param_name}: {direction} Float8 scale")
        gathered_scales = [torch.zeros_like(local_scale) for _ in range(world_size)]
        dist.all_gather(gathered_scales, local_scale)
        for rank, scale in enumerate(gathered_scales[1:], start=1):
            _record_exact(
                errors,
                scale,
                gathered_scales[0],
                f"{param_name}: {direction} scale rank {rank}",
            )
        _record_exact(
            errors,
            full_sub._scale_inv,
            gathered_scales[0],
            f"{param_name}: {direction} reconstructed scale",
        )


def _manual_reconstruct_hybrid(local, *, param_name, world_size):
    """Run the Hybrid FSDP pre/post protocol on manually gathered raw buffers."""
    _collective_assert(
        isinstance(local, HybridQuantizedTensor),
        f"{param_name}: local shard is {type(local).__name__}, expected HybridQuantizedTensor",
    )
    sharded_tensors, metadata = local.fsdp_pre_all_gather(
        mesh=None,
        orig_size=local.shape,
        contiguous_orig_stride=None,
        module=None,
        mp_policy=None,
    )
    local_count = torch.tensor([len(sharded_tensors)], dtype=torch.int64, device=local.device)
    min_count = local_count.clone()
    max_count = local_count.clone()
    dist.all_reduce(min_count, dist.ReduceOp.MIN)
    dist.all_reduce(max_count, dist.ReduceOp.MAX)
    _collective_assert(
        min_count.item() == max_count.item() and min_count.item() > 0,
        f"{param_name}: Hybrid FSDP buffer count range is ({min_count.item()}, {max_count.item()})",
    )

    gathered_outputs = []
    for index, shard in enumerate(sharded_tensors):
        _collective_assert(
            shard is not None,
            f"{param_name}: Hybrid FSDP buffer {index} is None",
        )
        _collective_assert_same_shape(shard, f"{param_name}: Hybrid FSDP buffer {index}")
        gathered = [torch.zeros_like(shard) for _ in range(world_size)]
        dist.all_gather(gathered, shard)
        gathered_outputs.append(torch.cat(gathered, dim=0))

    reconstructed, _ = local.fsdp_post_all_gather(
        tuple(gathered_outputs), metadata, local.dtype, out=None
    )
    _collective_assert(
        isinstance(reconstructed, HybridQuantizedTensor),
        f"{param_name}: Hybrid FSDP reconstruction returned {type(reconstructed).__name__}",
    )
    return reconstructed


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
    FSDP2's forward-time all-gather hooks for block-scaling QuantizedTensor
    subclasses fail when parameters are initialized directly on CUDA rather
    than on the meta device. NVFP4Tensor does not implement the FSDP all-gather
    hooks at all.

    For per-tensor FP8 (DelayedScaling, Float8CurrentScaling) the all-gather
    hooks handle CUDA-initialized Float8Tensor parameters correctly.
    """
    recipe = get_recipe_from_string(recipe_name)

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

    if recipe_name in (
        "MXFP8BlockScaling",
        "Float8BlockScaling",
        "NVFP4BlockScaling",
        "Float8CurrentScaling",
    ):
        pytest.xfail(
            f"{recipe_name}: FusedAdam without master_weights does not support "
            "this quantized tensor type. Use master_weights=True."
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

    if recipe_name == "Float8BlockScaling" and torch.cuda.get_device_capability()[0] == 12:
        pytest.xfail(
            "Float8BlockScaling is failing on SM120 with RuntimeError: "
            "transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu:534 "
            "in function quantize_transpose_vector_blockwise: Assertion failed: pow2_scale. On "
            "Blackwell and newer, the FP8 block scaling recipe is emulated with MXFP8, which "
            "requires using power of two scaling factors."
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

        # DelayedScaling: amax history and scaling factors live in _extra_state,
        # which cannot be saved via DCP due to non-deterministic pickle sizes
        # across ranks; the fresh model uses default scaling factors, producing
        # small numerical differences from FP8 re-quantization.
        # Float8CurrentScaling: Float8Tensor._scale_inv is passed via
        # fsdp_pre_all_gather metadata rather than as a sharded tensor, so DCP
        # saves it cast to the model's param_dtype (bf16) instead of fp32; the
        # precision loss in the reloaded scale_inv prevents bitwise parity.
        if isinstance(
            recipe,
            (
                transformer_engine.common.recipe.DelayedScaling,
                transformer_engine.common.recipe.Float8CurrentScaling,
            ),
        ):
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

        if isinstance(
            recipe,
            (
                transformer_engine.common.recipe.DelayedScaling,
                transformer_engine.common.recipe.Float8CurrentScaling,
            ),
        ):
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

            # DelayedScaling and Float8CurrentScaling use loose tolerance because
            # Float8Tensor._scale_inv is passed via fsdp_pre_all_gather metadata
            # rather than as a sharded tensor, so DCP saves it cast to the model's
            # param_dtype (bf16) instead of fp32. The resulting precision loss in
            # the reloaded scale_inv prevents bitwise-identical output parity.
            if isinstance(
                recipe,
                (
                    transformer_engine.common.recipe.DelayedScaling,
                    transformer_engine.common.recipe.Float8CurrentScaling,
                ),
            ):
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


# ---------------------------------------------------------------------------
# Hybrid quantization + FSDP2 tests
# ---------------------------------------------------------------------------


def _build_hybrid_model(hybrid_recipe, use_meta_device=True):
    """Build a model with quantized_model_init using a hybrid CustomRecipe."""
    kwargs = dict(
        fuse_qkv_params=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    if use_meta_device:
        kwargs["device"] = "meta"
    with te.quantized_model_init(enabled=True, recipe=hybrid_recipe):
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


def test_fused_adam_hybrid_master_weights(hybrid_recipe_name):
    """FusedAdam + master_weights + FSDP2 + hybrid quantized_model_init.

    Verifies:
    - Params are DTensors wrapping HybridQuantizedTensor local shards
    - Training loop completes without error
    - Optimizer states are FP32
    - Loss decreases over training steps
    """
    from transformer_engine.pytorch import HybridQuantizedTensor
    from fsdp2_utils import get_hybrid_recipe_from_string

    hybrid_recipe = get_hybrid_recipe_from_string(hybrid_recipe_name)
    world_size, device = _get_dist_info()

    model = _build_hybrid_model(hybrid_recipe)
    model = _shard_model(model, world_size)

    hybrid_count = sum(
        1
        for _, p in model.named_parameters()
        if isinstance(p, DTensor) and isinstance(p._local_tensor, HybridQuantizedTensor)
    )
    assert hybrid_count > 0, "No HybridQuantizedTensor local tensors after sharding"

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
        with te.autocast(enabled=True, recipe=hybrid_recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    for param in model.parameters():
        state = optimizer.state[param]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32
        if "master_param" in state:
            assert state["master_param"].dtype == torch.float32

    # Strictly monotonic decrease
    assert all(
        losses[i + 1] < losses[i] for i in range(len(losses) - 1)
    ), f"Loss not strictly decreasing each step: {losses}"


def test_fused_adam_hybrid_reshard_variants(hybrid_recipe_name):
    """Hybrid FusedAdam training must be numerically invariant to FSDP2's
    ``reshard_after_forward`` schedule.

    ``reshard_after_forward`` only changes *when* the gathered weight is
    materialized/freed, not the math: ``True`` (FSDP2's child-module default)
    drops the gathered weight after forward and re-gathers it in backward --
    invoking ``fsdp_post_all_gather(out=...)`` twice per step -- while ``False``
    keeps the gathered copy alive through backward (one gather per step). The
    gathered quantized bytes are identical either way, so both schedules must
    produce bitwise-identical outputs, losses, input/parameter gradients,
    master parameters, moments, and step counters.

    Strictly stronger than "loss decreased": it locks in that the hybrid
    all-gather hooks are schedule-invariant across both FSDP2 passes, and
    regression-guards the future P1.1 buffer-split bandwidth optimization.
    """
    from fsdp2_utils import get_hybrid_recipe_from_string

    hybrid_recipe = get_hybrid_recipe_from_string(hybrid_recipe_name)
    world_size, device = _get_dist_info()

    # Shared, fixed input/target so the two schedules are compared on identical data.
    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    def assert_exact(actual, expected, path):
        if torch.is_tensor(expected):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.0,
                atol=0.0,
                msg=lambda m: f"reshard schedule changed {path}: {m}",
            )
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys(), f"{path}: keys differ"
            for key in expected:
                assert_exact(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected), f"{path}: lengths differ"
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                assert_exact(actual_item, expected_item, f"{path}[{index}]")
        else:
            assert actual == expected, f"{path}: {actual!r} != {expected!r}"

    def run(reshard_after_forward):
        # Re-seed so both schedules get identical weight init from reset_parameters().
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        model = _shard_model(
            _build_hybrid_model(hybrid_recipe),
            world_size,
            reshard_after_forward=reshard_after_forward,
        )
        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )
        run_x = x.detach().clone().requires_grad_()
        artifacts = {
            "outputs": [],
            "losses": [],
            "input_grads": [],
            "param_grads": [],
            "optimizer": [],
        }
        for _ in range(NUM_STEPS):
            optimizer.zero_grad(set_to_none=True)
            run_x.grad = None
            with te.autocast(enabled=True, recipe=hybrid_recipe):
                output = model(run_x)
            artifacts["outputs"].append(output.detach().clone())
            loss = F.mse_loss(output, target)
            artifacts["losses"].append(loss.detach().clone())
            loss.backward()
            artifacts["input_grads"].append(run_x.grad.detach().clone())
            step_grads = []
            for param in model.parameters():
                grad = param.grad
                if grad is not None:
                    grad = grad.to_local() if isinstance(grad, DTensor) else grad
                    grad = grad.detach().clone()
                step_grads.append(grad)
            artifacts["param_grads"].append(step_grads)
            optimizer.step()
            step_state = []
            for param in model.parameters():
                param_state = {}
                for key, value in optimizer.state[param].items():
                    if torch.is_tensor(value):
                        value = value.to_local() if isinstance(value, DTensor) else value
                        value = value.detach().clone()
                    param_state[key] = value
                step_state.append(param_state)
            artifacts["optimizer"].append(step_state)
        return artifacts

    artifacts_resharded = run(reshard_after_forward=True)  # re-gather in backward
    artifacts_kept = run(reshard_after_forward=False)  # keep gathered weight through backward

    losses_resharded = [loss.item() for loss in artifacts_resharded["losses"]]
    assert all(
        losses_resharded[i + 1] < losses_resharded[i] for i in range(NUM_STEPS - 1)
    ), f"reshard_after_forward=True loss not strictly decreasing: {losses_resharded}"
    assert_exact(artifacts_resharded, artifacts_kept, "training")


def test_fused_adam_hybrid_bf16_vs_hybrid_parity(hybrid_recipe_name):
    """Compare hybrid+FSDP2 loss trajectory against BF16+FSDP2 within tolerance.

    This is a sanity check that hybrid quantized training converges similarly
    to BF16 training, not a bitwise-exact comparison.
    """
    from fsdp2_utils import get_hybrid_recipe_from_string

    hybrid_recipe = get_hybrid_recipe_from_string(hybrid_recipe_name)
    world_size, device = _get_dist_info()

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    def run_training(model, recipe_for_autocast):
        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )
        losses = []
        for _ in range(NUM_STEPS):
            optimizer.zero_grad(set_to_none=True)
            with te.autocast(enabled=(recipe_for_autocast is not None), recipe=recipe_for_autocast):
                output = model(x)
            loss = F.mse_loss(output, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        return losses

    # BF16 baseline
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    bf16_model = _build_model(fp8_init=False)
    bf16_model = _shard_model(bf16_model, world_size)
    bf16_losses = run_training(bf16_model, None)

    # Hybrid
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    hybrid_model = _build_hybrid_model(hybrid_recipe)
    hybrid_model = _shard_model(hybrid_model, world_size)
    hybrid_losses = run_training(hybrid_model, hybrid_recipe)

    assert hybrid_losses[-1] < hybrid_losses[0], f"Hybrid loss did not decrease: {hybrid_losses}"
    assert bf16_losses[-1] < bf16_losses[0], f"BF16 loss did not decrease: {bf16_losses}"

    # Hybrid stays within a few % of bf16 (seed-fixed).
    rel_tol = 0.10
    for step, (h_loss, b_loss) in enumerate(zip(hybrid_losses, bf16_losses)):
        rel_diff = abs(h_loss - b_loss) / max(abs(b_loss), 1e-10)
        assert rel_diff < rel_tol, (
            f"Step {step}: hybrid loss ({h_loss:.4f}) vs bf16 ({b_loss:.4f}) "
            f"differ by {rel_diff * 100:.2f}% (> {rel_tol * 100:.0f}%)"
        )


# Same-format hybrid -> the vanilla recipe it must match bitwise. Cross-format
# hybrids (e.g. HybridMixed_MXFP8_FP8) have no single-format vanilla equivalent.
_HYBRID_TO_BASE_RECIPE = {
    "HybridFP8CurrentScaling": "Float8CurrentScaling",
    "HybridMXFP8": "MXFP8BlockScaling",
    "HybridFloat8BlockScaling": "Float8BlockScaling",
}


def _build_linear_parity_stack(recipe):
    """Two bare ``te.Linear`` layers under ``quantized_model_init`` for
    hybrid-vs-vanilla bitwise parity.
    """
    with te.quantized_model_init(enabled=True, recipe=recipe):
        return torch.nn.Sequential(
            te.Linear(HIDDEN_SIZE, HIDDEN_SIZE, params_dtype=torch.bfloat16, device="meta"),
            te.Linear(HIDDEN_SIZE, HIDDEN_SIZE, params_dtype=torch.bfloat16, device="meta"),
        )


def test_fused_adam_hybrid_vs_base_recipe_parity(hybrid_recipe_name):
    """Same-format hybrid must match its vanilla recipe bitwise through the full
    FSDP2 + FusedAdam loop.

    Every output, loss, input/parameter gradient, master parameter, optimizer
    moment, and step counter is asserted bitwise-identical -- a regression guard
    for both amax reduction and master-weight requantization. Uses a bare
    ``te.Linear`` stack (see ``_build_linear_parity_stack``) to isolate
    GEMM-operand quantization.
    """
    if hybrid_recipe_name not in _HYBRID_TO_BASE_RECIPE:
        pytest.skip(
            f"{hybrid_recipe_name} is cross-format; no single-format vanilla "
            "recipe to compare against."
        )

    from fsdp2_utils import get_hybrid_recipe_from_string

    base_recipe_name = _HYBRID_TO_BASE_RECIPE[hybrid_recipe_name]
    world_size, device = _get_dist_info()

    # Shared, fixed input/target; the comparison is per-rank (base vs hybrid on
    # the same rank), so cross-rank input consistency does not matter.
    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    def run_training(build_fn, recipe_for_autocast):
        def snapshot_optimizer(model, optimizer):
            snapshots = []
            for param in model.parameters():
                state = {}
                for key, value in optimizer.state[param].items():
                    if torch.is_tensor(value):
                        value = value.to_local() if isinstance(value, DTensor) else value
                        value = value.detach().clone()
                    state[key] = value
                snapshots.append(state)
            return snapshots

        # Re-seed so both models get identical init from reset_parameters() (run
        # after sharding); with same-format quantization and a dropout-free loop
        # the full trajectory is then deterministic.
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        model = _shard_model(build_fn(), world_size)
        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )
        run_x = x.detach().clone().requires_grad_()
        outputs = []
        losses = []
        input_grads = []
        grads_per_step = []
        optimizer_states = []
        for step in range(NUM_STEPS):
            optimizer.zero_grad(set_to_none=True)
            run_x.grad = None
            with te.autocast(enabled=True, recipe=recipe_for_autocast):
                output = model(run_x)
            outputs.append(output.detach().clone())
            loss = F.mse_loss(output, target)
            losses.append(loss.detach().clone())
            loss.backward()
            input_grads.append(run_x.grad.detach().clone())
            # Snapshot grad local shards before the optimizer consumes them
            # (p.grad is a DTensor under FSDP2) to assert backward parity directly.
            step_grads = []
            for p in model.parameters():
                g = p.grad
                if g is None:
                    step_grads.append(None)
                else:
                    g = g.to_local() if isinstance(g, DTensor) else g
                    step_grads.append(g.detach().clone())
            grads_per_step.append(step_grads)
            optimizer.step()
            optimizer_states.append(snapshot_optimizer(model, optimizer))
        return outputs, losses, input_grads, grads_per_step, optimizer_states

    base_recipe = get_recipe_from_string(base_recipe_name)
    hybrid_recipe = get_hybrid_recipe_from_string(hybrid_recipe_name)

    base_outputs, base_losses, base_input_grads, base_grads, base_opt_states = run_training(
        lambda: _build_linear_parity_stack(base_recipe), base_recipe
    )
    (
        hybrid_outputs,
        hybrid_losses,
        hybrid_input_grads,
        hybrid_grads,
        hybrid_opt_states,
    ) = run_training(lambda: _build_linear_parity_stack(hybrid_recipe), hybrid_recipe)

    # (1) Every forward: bitwise-identical before and after optimizer updates.
    for step, (base_output, hybrid_output) in enumerate(zip(base_outputs, hybrid_outputs)):
        torch.testing.assert_close(
            hybrid_output,
            base_output,
            rtol=0.0,
            atol=0.0,
            msg=lambda m, s=step: f"[{hybrid_recipe_name} vs {base_recipe_name}] step {s} forward output not bitwise-identical: {m}",
        )

    # (2) Every per-step loss: bitwise-identical across the whole optimizer loop.
    for step, (b_loss, h_loss) in enumerate(zip(base_losses, hybrid_losses)):
        torch.testing.assert_close(
            h_loss,
            b_loss,
            rtol=0.0,
            atol=0.0,
            msg=lambda m, s=step: f"[{hybrid_recipe_name} vs {base_recipe_name}] step {s} loss not bitwise-identical to the vanilla recipe: {m}",
        )

    # (3) Backward: every weight-gradient shard at every step bitwise-identical
    for step, (base_grad, hybrid_grad) in enumerate(zip(base_input_grads, hybrid_input_grads)):
        torch.testing.assert_close(
            hybrid_grad,
            base_grad,
            rtol=0.0,
            atol=0.0,
            msg=lambda m, s=step: f"[{hybrid_recipe_name} vs {base_recipe_name}] step {s} input gradient not bitwise-identical: {m}",
        )

    #     (implied by the loss trajectory, but asserted directly to be explicit).
    for step, (b_step, h_step) in enumerate(zip(base_grads, hybrid_grads)):
        for i, (b_grad, h_grad) in enumerate(zip(b_step, h_step)):
            assert (b_grad is None) == (h_grad is None), (
                f"[{hybrid_recipe_name} vs {base_recipe_name}] step {step} param {i}"
                " gradient presence differs between hybrid and vanilla"
            )
            if b_grad is None:
                continue
            torch.testing.assert_close(
                h_grad,
                b_grad,
                rtol=0.0,
                atol=0.0,
                msg=lambda m, s=step, i=i: f"[{hybrid_recipe_name} vs {base_recipe_name}] step {s} param {i} gradient not bitwise-identical to the vanilla recipe: {m}",
            )

    # Optimizer continuation state: FP32 master params, moments, and counters.
    assert len(base_opt_states) == len(hybrid_opt_states)
    for step, (base_step, hybrid_step) in enumerate(zip(base_opt_states, hybrid_opt_states)):
        assert len(base_step) == len(hybrid_step)
        for param_idx, (base_state, hybrid_state) in enumerate(zip(base_step, hybrid_step)):
            assert base_state.keys() == hybrid_state.keys(), (
                f"[{hybrid_recipe_name} vs {base_recipe_name}] step {step} param {param_idx} "
                f"optimizer state keys differ: {base_state.keys()} != {hybrid_state.keys()}"
            )
            for key in base_state:
                base_value = base_state[key]
                hybrid_value = hybrid_state[key]
                if torch.is_tensor(base_value):
                    torch.testing.assert_close(
                        hybrid_value,
                        base_value,
                        rtol=0.0,
                        atol=0.0,
                        msg=lambda m, s=step, i=param_idx, k=key: f"[{hybrid_recipe_name} vs {base_recipe_name}] step {s} param {i} optimizer state {k!r} not bitwise-identical: {m}",
                    )
                else:
                    assert hybrid_value == base_value, (
                        f"[{hybrid_recipe_name} vs {base_recipe_name}] step {step} "
                        f"param {param_idx} optimizer state {key!r} differs: "
                        f"{hybrid_value!r} != {base_value!r}"
                    )


def test_fused_adam_hybrid_scale_uniform_across_shards(hybrid_recipe_name):
    """Per-tensor hybrid weights must share ONE amax-reduced scale across FSDP2
    shards -- tolerance-free regression guard for the amax-reduction fix.

    Without cross-shard reduction each rank quantizes its shard with a local amax
    and the scales differ; with the fix they match. Checked directly on the
    sharded weight (no forward). Block-scaled formats (MXFP8) are skipped.
    """
    if hybrid_recipe_name != "HybridFP8CurrentScaling":
        pytest.skip("scale-uniformity check applies to per-tensor current scaling only")

    from transformer_engine.pytorch import HybridQuantizedTensor
    from fsdp2_utils import get_hybrid_recipe_from_string

    world_size, device = _get_dist_info()
    if world_size < 2:
        pytest.skip("needs >=2 ranks to compare shard scales")

    hybrid_recipe = get_hybrid_recipe_from_string(hybrid_recipe_name)
    model = _build_hybrid_model(hybrid_recipe)
    model = _shard_model(model, world_size)

    checked = {"rowwise": 0, "columnwise": 0}
    for name, param in model.named_parameters():
        if not (
            isinstance(param, DTensor) and isinstance(param._local_tensor, HybridQuantizedTensor)
        ):
            continue
        for direction in checked:
            sub_storage = getattr(param._local_tensor, f"_{direction}_storage")
            scale_inv = getattr(sub_storage, "_scale_inv", None)
            if scale_inv is None:
                continue
            local_scale = scale_inv.detach().reshape(-1).clone()
            gathered = [torch.zeros_like(local_scale) for _ in range(world_size)]
            dist.all_gather(gathered, local_scale)
            for r in range(1, world_size):
                torch.testing.assert_close(
                    gathered[r],
                    gathered[0],
                    rtol=0.0,
                    atol=0.0,
                    msg=lambda m, n=name, r=r, d=direction: f"{n}: rank {r} {d} _scale_inv differs from rank 0 -- cross-shard amax reduction was not applied to the hybrid current-scaling weight: {m}",
                )
            checked[direction] += 1
    assert all(
        count > 0 for count in checked.values()
    ), f"missing hybrid current-scaling directions: {checked}"


def test_fused_adam_hybrid_identity_fp8_master_weights():
    """FSDP2 + FusedAdam with Hybrid(FP8 current rowwise, Identity columnwise).

    Covers the Identity sub-storage in hybrid FSDP2 all-gather while the FP8
    current rowwise direction validates cross-shard amax reduction via scale
    uniformity.
    """
    from transformer_engine.pytorch import HybridQuantizedTensor
    from transformer_engine.pytorch.tensor.storage.identity_tensor_storage import (
        IdentityTensorStorage,
    )
    from fsdp2_utils import get_hybrid_recipe_from_string

    world_size, device = _get_dist_info()
    if world_size < 2:
        pytest.skip("needs >=2 ranks to validate cross-shard amax reduction")

    hybrid_recipe = get_hybrid_recipe_from_string("HybridFP8CurrentScalingIdentity")
    model = _build_linear_parity_stack(hybrid_recipe)
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
    identity_steps = 2
    for step in range(identity_steps):
        optimizer.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=hybrid_recipe):
            output = model(x)
        loss = F.mse_loss(output, target)
        losses.append(loss.item())
        loss.backward()
        if step < identity_steps - 1:
            optimizer.step()

    assert all(
        losses[i + 1] < losses[i] for i in range(len(losses) - 1)
    ), f"Hybrid Identity/FP8 loss not strictly decreasing: {losses}"

    checked_identity = 0
    checked_scale = 0
    for name, param in model.named_parameters():
        if not (
            isinstance(param, DTensor) and isinstance(param._local_tensor, HybridQuantizedTensor)
        ):
            continue
        local = param._local_tensor
        assert isinstance(local._columnwise_storage, IdentityTensorStorage)

        scale_inv = getattr(local._rowwise_storage, "_scale_inv", None)
        if scale_inv is not None:
            local_scale = scale_inv.detach().reshape(-1).clone()
            gathered_scales = [torch.zeros_like(local_scale) for _ in range(world_size)]
            dist.all_gather(gathered_scales, local_scale)
            for r in range(1, world_size):
                torch.testing.assert_close(
                    gathered_scales[r],
                    gathered_scales[0],
                    rtol=0.0,
                    atol=0.0,
                    msg=lambda m, n=name, r=r: f"{n}: rank {r} rowwise _scale_inv differs from rank 0 for Hybrid(FP8Current, Identity): {m}",
                )
            checked_scale += 1

        local_identity = local._columnwise_storage.dequantize().contiguous()
        gathered_identity = [torch.zeros_like(local_identity) for _ in range(world_size)]
        dist.all_gather(gathered_identity, local_identity)
        manual_full = torch.cat(gathered_identity, dim=0)

        sharded_tensors, metadata = local.fsdp_pre_all_gather(
            mesh=None,
            orig_size=local.shape,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        all_gather_outputs = []
        for shard in sharded_tensors:
            gathered = [torch.zeros_like(shard) for _ in range(world_size)]
            dist.all_gather(gathered, shard)
            all_gather_outputs.append(torch.cat(gathered, dim=0))
        fsdp_full, _ = local.fsdp_post_all_gather(
            tuple(all_gather_outputs), metadata, local.dtype, out=None
        )
        assert isinstance(fsdp_full, HybridQuantizedTensor)
        full_identity = fsdp_full._columnwise_storage.dequantize()
        torch.testing.assert_close(
            manual_full.float(),
            full_identity[: manual_full.shape[0]].float(),
            rtol=0.0,
            atol=0.0,
            msg=lambda m, n=name: f"{n}: Identity columnwise all-gather mismatch: {m}",
        )
        checked_identity += 1

    assert checked_identity > 0, "no Hybrid(FP8Current, Identity) params found"
    assert checked_scale > 0, "no FP8 current rowwise scales found to check"


def test_fused_adam_hybrid_allgather_correctness(hybrid_recipe_name):
    """Validate both Hybrid directions through FSDP2 at raw-buffer precision.

    The rowwise and columnwise sub-storages are checked independently. Every
    gathered data buffer, scale buffer, field layout, and per-tensor scale
    metadata value must exactly match a manual dim-0 all-gather.
    """
    from fsdp2_utils import get_hybrid_recipe_from_string

    expected_types = {
        "HybridFP8CurrentScaling": (Float8TensorStorage, Float8TensorStorage),
        "HybridMXFP8": (MXFP8TensorStorage, MXFP8TensorStorage),
        "HybridFloat8BlockScaling": (
            Float8BlockwiseQTensorStorage,
            Float8BlockwiseQTensorStorage,
        ),
        "HybridMixed_MXFP8_FP8": (MXFP8TensorStorage, Float8TensorStorage),
    }
    row_type, col_type = expected_types[hybrid_recipe_name]
    hybrid_recipe = get_hybrid_recipe_from_string(hybrid_recipe_name)
    world_size, device = _get_dist_info()

    model = _build_hybrid_model(hybrid_recipe)
    model = _shard_model(model, world_size)

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    with te.autocast(enabled=True, recipe=hybrid_recipe):
        _ = model(x)

    params = [
        (name, param)
        for name, param in model.named_parameters()
        if isinstance(param, DTensor) and isinstance(param._local_tensor, HybridQuantizedTensor)
    ]
    local_count = torch.tensor([len(params)], dtype=torch.int64, device=device)
    min_count = local_count.clone()
    max_count = local_count.clone()
    dist.all_reduce(min_count, dist.ReduceOp.MIN)
    dist.all_reduce(max_count, dist.ReduceOp.MAX)
    _collective_assert(
        min_count.item() == max_count.item() and min_count.item() > 0,
        f"{hybrid_recipe_name}: Hybrid parameter count range is "
        f"({min_count.item()}, {max_count.item()})",
    )

    errors = []
    for name, param in params:
        local = param._local_tensor
        reconstructed = _manual_reconstruct_hybrid(local, param_name=name, world_size=world_size)
        public_full = param.full_tensor()
        _record_exact(
            errors,
            public_full,
            reconstructed.dequantize(),
            f"{name}: public full_tensor vs Hybrid reconstruction",
        )
        _check_hybrid_direction_buffers(
            local._rowwise_storage,
            reconstructed._rowwise_storage,
            row_type,
            direction="rowwise",
            param_name=name,
            world_size=world_size,
            errors=errors,
        )
        _check_hybrid_direction_buffers(
            local._columnwise_storage,
            reconstructed._columnwise_storage,
            col_type,
            direction="columnwise",
            param_name=name,
            world_size=world_size,
            errors=errors,
        )

    _raise_collective_errors(errors, f"{hybrid_recipe_name} raw Hybrid all-gather")


def test_fused_adam_hybrid_mxfp8_awkward_shard_shape():
    """Exercise MXFP8 block-scale unpad/pad on a sharded Linear whose shard
    dim-0 is block-aligned (divisible by 32) but NOT divisible by 128.

    MXFP8 block scales are stored with ``[128, 4]`` / ``[4, 128]`` alignment
    padding, which must be stripped before FSDP2's dim-0 all-gather and
    re-applied after. With ``HIDDEN_SIZE`` and ``FFN_HIDDEN_SIZE`` both
    divisible by 128, the default model never forces this code path, so this
    test uses a hand-picked Linear size.

    Regression test for the "pre-fix" bug where
    ``HybridQuantizedTensor.fsdp_pre_all_gather`` pulled raw tensor fields via
    ``get_metadata()`` without unpadding the scale — the padded bytes would
    have been interleaved at every rank boundary in the gather output.
    """
    from fsdp2_utils import get_hybrid_recipe_from_string

    supported, reason = te.is_mxfp8_available(return_reason=True)
    if not supported:
        pytest.skip(f"MXFP8: {reason}")

    world_size, device = _get_dist_info()

    # FSDP2 shards a Linear weight of shape (out_features, in_features) along
    # dim-0, so each rank holds `out_features / world_size` rows. Pick
    # per-rank shard dim-0 = 96: divisible by MXFP8_BLOCK_SCALING_SIZE (32)
    # so data alignment holds, but NOT divisible by 128 so the rowwise
    # scale-inv needs alignment padding on the sharded copy. This is the
    # shape that exercises the unpad-before-gather / pad-after-gather
    # behaviour in MXFP8TensorStorage.fsdp_{extract,assign}_buffers.
    per_rank_out = 96
    out_features = per_rank_out * world_size
    in_features = 128  # arbitrary, divisible by 32; not sharded by FSDP2 here
    assert per_rank_out % 32 == 0, (
        f"Test setup error: per_rank_out={per_rank_out} (= out_features / world_size, "
        f"world_size={world_size}) must be a multiple of the MXFP8 block size (32) so the "
        "sharded weight's data stays block-aligned. Pick a per_rank_out divisible by 32."
    )
    assert per_rank_out % 128 != 0, (
        f"Test setup error: per_rank_out={per_rank_out} must NOT be a multiple of 128, or the "
        "rowwise scale-inv needs no alignment padding and this test stops exercising the MXFP8 "
        "unpad-before-gather / pad-after-gather path it exists to cover. Pick a per_rank_out "
        "divisible by 32 but not 128 (e.g. 96)."
    )

    for recipe_name in ("HybridMXFP8", "HybridMixed_MXFP8_FP8"):
        hybrid_recipe = get_hybrid_recipe_from_string(recipe_name)

        with te.quantized_model_init(enabled=True, recipe=hybrid_recipe):
            model = torch.nn.Sequential(
                te.Linear(
                    in_features,
                    out_features,
                    params_dtype=torch.bfloat16,
                    device="meta",
                ),
            )
        model = _shard_model(model, world_size)

        # Batch (leading) dim must be divisible by MXFP8_BLOCK_SCALING_SIZE (32).
        x = torch.randn(32, in_features, dtype=torch.bfloat16, device=device)
        with te.autocast(enabled=True, recipe=hybrid_recipe):
            out = model(x)
        out.sum().backward()

        col_type = MXFP8TensorStorage if recipe_name == "HybridMXFP8" else Float8TensorStorage
        params = [
            (name, param)
            for name, param in model.named_parameters()
            if isinstance(param, DTensor) and isinstance(param._local_tensor, HybridQuantizedTensor)
        ]
        local_count = torch.tensor([len(params)], dtype=torch.int64, device=device)
        min_count = local_count.clone()
        max_count = local_count.clone()
        dist.all_reduce(min_count, dist.ReduceOp.MIN)
        dist.all_reduce(max_count, dist.ReduceOp.MAX)
        _collective_assert(
            min_count.item() == max_count.item() and min_count.item() > 0,
            f"{recipe_name}: awkward-shape Hybrid parameter count range is "
            f"({min_count.item()}, {max_count.item()})",
        )

        errors = []
        for name, param in params:
            local = param._local_tensor
            label = f"{recipe_name}:{name}"
            reconstructed = _manual_reconstruct_hybrid(
                local, param_name=label, world_size=world_size
            )
            public_full = param.full_tensor()
            _record_exact(
                errors,
                public_full,
                reconstructed.dequantize(),
                f"{label}: public full_tensor vs Hybrid reconstruction",
            )
            _check_hybrid_direction_buffers(
                local._rowwise_storage,
                reconstructed._rowwise_storage,
                MXFP8TensorStorage,
                direction="rowwise",
                param_name=label,
                world_size=world_size,
                errors=errors,
            )
            _check_hybrid_direction_buffers(
                local._columnwise_storage,
                reconstructed._columnwise_storage,
                col_type,
                direction="columnwise",
                param_name=label,
                world_size=world_size,
                errors=errors,
            )
        _raise_collective_errors(errors, f"{recipe_name} awkward raw Hybrid all-gather")


def test_fused_adam_hybrid_float8_block_unaligned_shard_shape():
    """Unaligned local shards are rejected before gathering incompatible scale tiles."""
    from transformer_engine.pytorch import fp8
    from fsdp2_utils import get_hybrid_recipe_from_string

    supported, reason = fp8.check_fp8_block_scaling_support()
    if not supported:
        pytest.skip(reason)

    world_size, device = _get_dist_info()
    if world_size != 2:
        pytest.skip("test shape is defined for exactly two FSDP ranks")

    in_features = 256
    per_rank_out = 192
    out_features = per_rank_out * world_size
    assert out_features % 128 == 0
    assert per_rank_out % 128 != 0

    hybrid_recipe = get_hybrid_recipe_from_string("HybridFloat8BlockScaling")
    with te.quantized_model_init(enabled=True, recipe=hybrid_recipe):
        model = torch.nn.Sequential(
            te.Linear(
                in_features,
                out_features,
                params_dtype=torch.bfloat16,
                device="meta",
            )
        )
    model = _shard_model(model, world_size)

    x = torch.randn(128, in_features, dtype=torch.bfloat16, device=device)
    with pytest.raises(
        RuntimeError,
        match="local flattened M dimension.*not a multiple of 128",
    ):
        with te.autocast(enabled=True, recipe=hybrid_recipe):
            model(x)


def test_hybrid_dcp_output_parity(hybrid_recipe_name):
    """DCP roundtrip and exact forked optimizer continuation.

    Trains and checkpoints a hybrid model plus FusedAdam, loads both into fresh
    objects, and compares the identical next step against uninterrupted training:
    output, loss, input/parameter gradients, master params, moments, counters,
    and post-step output must all match bitwise.
    """
    import torch.distributed.checkpoint as dcp

    from fsdp2_utils import get_hybrid_recipe_from_string

    hybrid_recipe = get_hybrid_recipe_from_string(hybrid_recipe_name)
    world_size, device = _get_dist_info()
    rank = int(os.environ.get("RANK", "0"))
    # Deterministic, rank-agnostic checkpoint dir so all ranks read/write
    # the same DCP path. ``os.getpid()`` differs per rank under torchrun.
    checkpoint_dir = f"/tmp/te_test_fsdp2_hybrid_dcp_parity_{hybrid_recipe_name}"

    if rank == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    dist.barrier()

    try:
        model = _build_hybrid_model(hybrid_recipe)
        model = _shard_model(model, world_size)
        optimizer = te.optimizers.FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )

        x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        target = torch.randn_like(x)
        failures = []

        def snapshot_optimizer(current_model, current_optimizer):
            snapshots = []
            for param in current_model.parameters():
                state = {}
                for key, value in current_optimizer.state[param].items():
                    if torch.is_tensor(value):
                        value = value.to_local() if isinstance(value, DTensor) else value
                        value = value.detach().clone()
                    state[key] = value
                snapshots.append(state)
            return snapshots

        def check_optimizer_state(actual, expected, label):
            if len(actual) != len(expected):
                failures.append(
                    f"{label}: parameter count differs: {len(actual)} != {len(expected)}"
                )
            for param_idx, (actual_state, expected_state) in enumerate(zip(actual, expected)):
                if actual_state.keys() != expected_state.keys():
                    failures.append(
                        f"{label}: param {param_idx} state keys differ: "
                        f"{actual_state.keys()} != {expected_state.keys()}"
                    )
                for key in actual_state.keys() & expected_state.keys():
                    actual_value = actual_state[key]
                    expected_value = expected_state[key]
                    if torch.is_tensor(expected_value):
                        _record_exact(
                            failures,
                            actual_value,
                            expected_value,
                            f"{label}: param {param_idx} optimizer state {key!r}",
                        )
                    elif actual_value != expected_value:
                        failures.append(
                            f"{label}: param {param_idx} optimizer state {key!r} differs: "
                            f"{actual_value!r} != {expected_value!r}"
                        )

        def run_continuation_step(current_model, current_optimizer):
            step_x = x.detach().clone().requires_grad_()
            current_optimizer.zero_grad(set_to_none=True)
            with te.autocast(enabled=True, recipe=hybrid_recipe):
                step_output = current_model(step_x)
            step_loss = F.mse_loss(step_output, target)
            step_loss.backward()
            step_grads = []
            for param in current_model.parameters():
                grad = param.grad
                if grad is not None:
                    grad = grad.to_local() if isinstance(grad, DTensor) else grad
                    grad = grad.detach().clone()
                step_grads.append(grad)
            current_optimizer.step()
            with torch.no_grad(), te.autocast(enabled=True, recipe=hybrid_recipe):
                post_step_output = current_model(x).detach().clone()
            return {
                "output": step_output.detach().clone(),
                "loss": step_loss.detach().clone(),
                "input_grad": None if step_x.grad is None else step_x.grad.detach().clone(),
                "param_grads": step_grads,
                "optimizer": snapshot_optimizer(current_model, current_optimizer),
                "post_step_output": post_step_output,
            }

        for _ in range(NUM_STEPS):
            optimizer.zero_grad(set_to_none=True)
            with te.autocast(enabled=True, recipe=hybrid_recipe):
                output = model(x)
            F.mse_loss(output, target).backward()
            optimizer.step()

        with torch.no_grad():
            with te.autocast(enabled=True, recipe=hybrid_recipe):
                ref_output = model(x).clone()

        save_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        dcp.save(save_state, checkpoint_id=checkpoint_dir)
        saved_optimizer_state = snapshot_optimizer(model, optimizer)

        model2 = _build_hybrid_model(hybrid_recipe)
        model2 = _shard_model(model2, world_size)
        optimizer2 = te.optimizers.FusedAdam(
            model2.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.float32,
        )
        optimizer2.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=hybrid_recipe):
            out_tmp = model2(x)
        F.mse_loss(out_tmp, target).backward()
        optimizer2.step()

        state_to_load = {
            "model": model2.state_dict(),
            "optimizer": optimizer2.state_dict(),
        }
        dcp.load(state_to_load, checkpoint_id=checkpoint_dir)
        model2.load_state_dict(state_to_load["model"])
        optimizer2.load_state_dict(state_to_load["optimizer"])

        with torch.no_grad():
            with te.autocast(enabled=True, recipe=hybrid_recipe):
                loaded_output = model2(x)
        check_optimizer_state(
            snapshot_optimizer(model2, optimizer2),
            saved_optimizer_state,
            "DCP optimizer state immediately after load",
        )
        _record_exact(failures, loaded_output, ref_output, "DCP roundtrip output")
        _raise_collective_errors(failures, "DCP state immediately after load")
        failures.clear()

        # Fork from the checkpoint and execute the identical next step. This
        # catches an ignored optimizer checkpoint even when model-only output
        # parity succeeds immediately after load.
        torch.manual_seed(9876)
        torch.cuda.manual_seed(9876)
        reference_step = run_continuation_step(model, optimizer)
        torch.manual_seed(9876)
        torch.cuda.manual_seed(9876)
        resumed_step = run_continuation_step(model2, optimizer2)

        for key in ("output", "loss", "input_grad", "post_step_output"):
            _record_exact(
                failures,
                resumed_step[key],
                reference_step[key],
                f"DCP continuation {key}",
            )

        resumed_grads = resumed_step["param_grads"]
        reference_grads = reference_step["param_grads"]
        if len(resumed_grads) != len(reference_grads):
            failures.append(
                "DCP continuation parameter-gradient count differs: "
                f"{len(resumed_grads)} != {len(reference_grads)}"
            )
        for param_idx, (resumed_grad, reference_grad) in enumerate(
            zip(resumed_grads, reference_grads)
        ):
            _record_exact(
                failures,
                resumed_grad,
                reference_grad,
                f"DCP continuation param {param_idx} gradient",
            )

        check_optimizer_state(
            resumed_step["optimizer"],
            reference_step["optimizer"],
            "DCP optimizer state after continuation",
        )
        _raise_collective_errors(failures, "DCP forked continuation")
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


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
    "fused_adam_hybrid_master_weights": test_fused_adam_hybrid_master_weights,
    "fused_adam_hybrid_bf16_vs_hybrid_parity": test_fused_adam_hybrid_bf16_vs_hybrid_parity,
    "fused_adam_hybrid_vs_base_recipe_parity": test_fused_adam_hybrid_vs_base_recipe_parity,
    "fused_adam_hybrid_scale_uniform_across_shards": (
        test_fused_adam_hybrid_scale_uniform_across_shards
    ),
    "fused_adam_hybrid_identity_fp8_master_weights": (
        test_fused_adam_hybrid_identity_fp8_master_weights
    ),
    "fused_adam_hybrid_allgather_correctness": test_fused_adam_hybrid_allgather_correctness,
    "fused_adam_hybrid_mxfp8_awkward_shard_shape": test_fused_adam_hybrid_mxfp8_awkward_shard_shape,
    "hybrid_dcp_output_parity": test_hybrid_dcp_output_parity,
}

# Hybrid tests that are NOT parametrized by recipe (they sweep internally).
_HYBRID_NON_PARAMETRIZED_TESTS = {
    "fused_adam_hybrid_identity_fp8_master_weights",
    "fused_adam_hybrid_mxfp8_awkward_shard_shape",
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
            "HybridFP8CurrentScaling",
            "HybridMXFP8",
            "HybridFloat8BlockScaling",
            "HybridMixed_MXFP8_FP8",
            "HybridFP8CurrentScalingIdentity",
        ],
    )
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    try:
        if args.test in _HYBRID_NON_PARAMETRIZED_TESTS:
            TESTS[args.test]()
        else:
            TESTS[args.test](args.recipe)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
