#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FSDP2 memory leak detection tests.

These tests verify that temporary TE tensors (FP8 quantized weights, transpose
caches) are properly freed when moving between layers with FSDP2.

Related issues:
  - https://github.com/NVIDIA/TransformerEngine/issues/2681
    Quantized weights created during forward pass accumulate across layers.
  - https://github.com/NVIDIA/TransformerEngine/issues/2717
    _create_transpose tensors accumulate across training steps with
    quantized_model_init + FusedAdam + FSDP2.

Run all tests (via torchrun + pytest):
  torchrun -m pytest <this_file> -v --tb=short

Run a single test standalone (for debugging):
  torchrun <this_file> --test <name> --recipe <recipe>

Available --test values:
  bf16_no_excess_forward_memory, fp8_temp_accumulation_across_layers,
  transpose_cache_retained_after_backward

Available --recipe values:
  DelayedScaling, Float8CurrentScaling, Float8BlockScaling,
  MXFP8BlockScaling, NVFP4BlockScaling
"""

import argparse
import gc
import os
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh

import transformer_engine.pytorch as te

from fsdp2_utils import get_recipe_from_string, save_custom_attrs, restore_custom_attrs


# ── Constants ────────────────────────────────────────────────────────
HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
NUM_LAYERS = 8
SEQ_LEN = 32
BATCH_PER_RANK = 2
WARMUP_STEPS = 2


# ── Helpers ──────────────────────────────────────────────────────────
def _build_model(num_layers, fp8_init, recipe=None, use_meta_device=True):
    """Build a Sequential of TransformerLayers, optionally with FP8 init.

    When fp8_init=True and use_meta_device=True (the default), the model is
    created on the meta device so parameters are materialized after FSDP2
    sharding via reset_parameters().
    """
    if fp8_init:
        ctx = te.quantized_model_init(enabled=True, recipe=recipe)
    else:
        ctx = nullcontext()
    kwargs = dict(
        fuse_qkv_params=True,
        params_dtype=torch.bfloat16,
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
                for _ in range(num_layers)
            ]
        )
    return model


def _shard_model(model, world_size):
    """Apply FSDP2 sharding with save/restore of custom attrs."""
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
    restore_custom_attrs(model, custom_attrs)
    return model


def _get_dist_info():
    """Get world_size and device from environment."""
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    return world_size, device


def _run_training_step(model, optimizer, recipe, x, target):
    """Run one forward + backward + optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    with te.autocast(enabled=(recipe is not None), recipe=recipe):
        output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def _measure_backward_memory_delta(model, optimizer, recipe, x, target):
    """Run a training step and return (post_bwd - post_fwd) memory delta.

    This delta captures memory added during backward that persists afterward.
    In a healthy system, backward frees activations and adds only gradients.
    If transpose caches or other FP8 temps persist, the delta will be larger.
    """
    optimizer.zero_grad(set_to_none=True)
    with te.autocast(enabled=(recipe is not None), recipe=recipe):
        output = model(x)
    torch.cuda.synchronize()
    mem_post_fwd = torch.cuda.memory_allocated()

    loss = F.mse_loss(output, target)
    loss.backward()
    torch.cuda.synchronize()
    mem_post_bwd = torch.cuda.memory_allocated()

    optimizer.step()
    return mem_post_bwd - mem_post_fwd


def _maybe_skip(recipe_name, quantized_model_init):
    """Skip configurations that fail for reasons unrelated to memory leaks."""
    if recipe_name == "NVFP4BlockScaling" and quantized_model_init:
        pytest.skip(
            "NVFP4BlockScaling + quantized_model_init: not supported with FSDP2 "
            "(block tensor dequantized before FSDP2 flatten)"
        )


class _LayerMemoryTracker:
    """Register forward hooks on Sequential children to measure per-layer memory."""

    def __init__(self):
        self.post_forward_mem = []
        self._handles = []

    def attach(self, model):
        for i, layer in enumerate(model.children()):

            def make_hook(idx):
                def hook(module, args, output):
                    torch.cuda.synchronize()
                    self.post_forward_mem.append(torch.cuda.memory_allocated())

                return hook

            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def clear(self):
        self.post_forward_mem.clear()

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def per_layer_increments(self):
        """Return list of memory increments between consecutive post-forward hooks."""
        return [
            self.post_forward_mem[i] - self.post_forward_mem[i - 1]
            for i in range(1, len(self.post_forward_mem))
        ]


def _measure_forward_increments(model, optimizer, recipe, x, target):
    """Run a single training step with hooks and return per-layer forward memory increments."""
    tracker = _LayerMemoryTracker()
    tracker.attach(model)
    try:
        _run_training_step(model, optimizer, recipe, x, target)
        return tracker.per_layer_increments()
    finally:
        tracker.detach()


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(params=[False, True], ids=["no_quant_init", "quant_init"])
def quantized_model_init(request):
    return request.param


# ── Tests ────────────────────────────────────────────────────────────
def test_bf16_no_excess_forward_memory():
    """Control test: bf16 (no FP8) should have stable per-layer forward memory.

    With FSDP2 and bf16 params (no FP8), the per-layer memory growth during
    forward should only be activation saves for autograd. There should be no
    FP8 temporary accumulation. This test validates the measurement approach.
    """
    world_size, device = _get_dist_info()

    model = _build_model(NUM_LAYERS, fp8_init=False)
    model = _shard_model(model, world_size)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    # Warmup
    for _ in range(WARMUP_STEPS):
        _run_training_step(model, optimizer, None, x, target)

    # Measure
    increments = _measure_forward_increments(model, optimizer, None, x, target)

    # bf16 per-layer increments should be consistent (activation saves only)
    # and should NOT grow over layers (each layer saves similar activations).
    avg_increment = sum(increments) / len(increments)
    max_deviation = max(abs(inc - avg_increment) for inc in increments)

    # Allow 10% deviation from mean -- bf16 increments should be very uniform
    assert max_deviation <= 0.1 * abs(avg_increment) + 1024, (
        "bf16 per-layer increments are not uniform. "
        f"Increments (KiB): {[f'{inc/1024:.1f}' for inc in increments]}. "
        f"Average: {avg_increment/1024:.1f} KiB, max deviation: {max_deviation/1024:.1f} KiB"
    )


def test_fp8_temp_accumulation_across_layers(recipe_name, quantized_model_init):
    """Detect FP8 weight temporaries accumulating across layers during forward.

    Strategy: measure per-layer memory growth during forward for both bf16
    (baseline) and FP8. With FSDP2, per-layer params are unsharded then
    resharded, so the only per-layer memory growth should be activation saves
    for autograd (same as bf16). If FP8 adds excess per-layer growth, it means
    FP8 weight copies are accumulating across layers instead of being freed.
    """
    _maybe_skip(recipe_name, quantized_model_init)

    recipe = get_recipe_from_string(recipe_name)
    world_size, device = _get_dist_info()

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    # ── bf16 baseline ──
    bf16_model = _build_model(NUM_LAYERS, fp8_init=False)
    bf16_model = _shard_model(bf16_model, world_size)
    bf16_optimizer = te.optimizers.FusedAdam(
        bf16_model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )
    for _ in range(WARMUP_STEPS):
        _run_training_step(bf16_model, bf16_optimizer, None, x, target)
    bf16_increments = _measure_forward_increments(bf16_model, bf16_optimizer, None, x, target)
    bf16_avg = sum(bf16_increments) / len(bf16_increments)

    del bf16_model, bf16_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # ── FP8 model ──
    fp8_model = _build_model(NUM_LAYERS, fp8_init=quantized_model_init, recipe=recipe)
    fp8_model = _shard_model(fp8_model, world_size)
    fp8_optimizer = te.optimizers.FusedAdam(
        fp8_model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )
    for _ in range(WARMUP_STEPS):
        _run_training_step(fp8_model, fp8_optimizer, recipe, x, target)
    fp8_increments = _measure_forward_increments(fp8_model, fp8_optimizer, recipe, x, target)
    fp8_avg = sum(fp8_increments) / len(fp8_increments)

    # ── Assert: FP8 per-layer excess should be bounded ──
    # If FP8 temps are properly freed between layers, per-layer increment
    # should be similar to bf16 (just activation saves). Any excess indicates
    # FP8 weight copies accumulating.
    excess_per_layer = fp8_avg - bf16_avg

    # Allow up to 50 KiB per layer for FP8 scale/amax metadata.
    # FP8 weight copies (~0.68 MiB/layer for this model) should NOT persist.
    tolerance_per_layer = 50 * 1024  # 50 KiB

    assert excess_per_layer <= tolerance_per_layer, (
        "FP8 per-layer forward memory increment exceeds bf16 baseline by "
        f"{excess_per_layer/1024:.1f} KiB/layer (tolerance: {tolerance_per_layer/1024:.1f} KiB). "
        f"bf16 avg: {bf16_avg/1024:.1f} KiB/layer, FP8 avg: {fp8_avg/1024:.1f} KiB/layer. "
        f"FP8 increments (KiB): {[f'{inc/1024:.1f}' for inc in fp8_increments]}. "
        "FP8 weight copies are likely accumulating across layers (Issue #2681)."
    )


def test_bf16_no_excess_backward_memory():
    """Control test: two identical bf16 models should show zero backward excess.

    This mirrors the structure of test_transpose_cache_retained_after_backward
    but compares bf16 vs bf16 instead of FP8 vs bf16. The excess should be
    zero, proving the comparison methodology works.
    """
    world_size, device = _get_dist_info()

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    # Build and measure first bf16 model (acts as "baseline")
    model_a = _build_model(NUM_LAYERS, fp8_init=False)
    model_a = _shard_model(model_a, world_size)
    opt_a = te.optimizers.FusedAdam(
        model_a.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )
    for _ in range(WARMUP_STEPS):
        _run_training_step(model_a, opt_a, None, x, target)
    delta_a = _measure_backward_memory_delta(model_a, opt_a, None, x, target)

    del model_a, opt_a
    gc.collect()
    torch.cuda.empty_cache()

    # Build and measure second bf16 model (acts as "test")
    model_b = _build_model(NUM_LAYERS, fp8_init=False)
    model_b = _shard_model(model_b, world_size)
    opt_b = te.optimizers.FusedAdam(
        model_b.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )
    for _ in range(WARMUP_STEPS):
        _run_training_step(model_b, opt_b, None, x, target)
    delta_b = _measure_backward_memory_delta(model_b, opt_b, None, x, target)

    excess = delta_b - delta_a
    tolerance = 256 * 1024  # 256 KiB

    assert abs(excess) <= tolerance, (
        "Two identical bf16 models show backward delta excess of "
        f"{excess/1024:.1f} KiB (tolerance: {tolerance/1024:.0f} KiB). "
        f"delta_a={delta_a/1024**2:.2f} MiB, delta_b={delta_b/1024**2:.2f} MiB."
    )


def test_transpose_cache_retained_after_backward(recipe_name, quantized_model_init):
    """Detect transpose caches persisting after backward completes.

    When FP8 backward runs, _create_transpose allocates tensors for transposed
    weight copies. These should be freed when backward completes, but instead
    they persist until the next forward pass. This test measures the backward
    memory delta (post_bwd - post_fwd) and compares it to a bf16 baseline.
    In bf16, backward frees activations and adds gradients (net negative delta).
    With FP8, retained transpose caches make the delta significantly more positive.
    """
    _maybe_skip(recipe_name, quantized_model_init)

    recipe = get_recipe_from_string(recipe_name)
    world_size, device = _get_dist_info()

    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    # ── bf16 baseline ──
    bf16_model = _build_model(NUM_LAYERS, fp8_init=False)
    bf16_model = _shard_model(bf16_model, world_size)
    bf16_optimizer = te.optimizers.FusedAdam(
        bf16_model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )
    for _ in range(WARMUP_STEPS):
        _run_training_step(bf16_model, bf16_optimizer, None, x, target)
    bf16_bwd_delta = _measure_backward_memory_delta(
        bf16_model,
        bf16_optimizer,
        None,
        x,
        target,
    )

    del bf16_model, bf16_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # ── FP8 model ──
    fp8_model = _build_model(NUM_LAYERS, fp8_init=quantized_model_init, recipe=recipe)
    fp8_model = _shard_model(fp8_model, world_size)
    fp8_optimizer = te.optimizers.FusedAdam(
        fp8_model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )
    for _ in range(WARMUP_STEPS):
        _run_training_step(fp8_model, fp8_optimizer, recipe, x, target)
    fp8_bwd_delta = _measure_backward_memory_delta(
        fp8_model,
        fp8_optimizer,
        recipe,
        x,
        target,
    )

    # ── Assert: FP8 backward should not retain excess memory ──
    # In bf16, backward frees activations and adds gradients (typically net negative).
    # If FP8 transpose caches persist after backward, the FP8 delta will be
    # significantly more positive than bf16.
    excess = fp8_bwd_delta - bf16_bwd_delta

    # Allow 1 MiB for FP8 scale/amax bookkeeping and temporary workspace
    # re-creation during backward. The key check is that transpose caches
    # (~3 MiB for this 8-layer model) do NOT persist across steps.
    tolerance = 1024 * 1024

    assert excess <= tolerance, (
        f"FP8 backward retains {excess/1024**2:.2f} MiB more than bf16 baseline. "
        f"bf16 backward delta: {bf16_bwd_delta/1024**2:.2f} MiB, "
        f"FP8 backward delta: {fp8_bwd_delta/1024**2:.2f} MiB. "
        "Transpose caches from backward are likely not being freed (Issue #2717)."
    )


# ── Standalone runner ────────────────────────────────────────────────
TESTS = {
    "bf16_no_excess_forward_memory": test_bf16_no_excess_forward_memory,
    "bf16_no_excess_backward_memory": test_bf16_no_excess_backward_memory,
    "fp8_temp_accumulation_across_layers": test_fp8_temp_accumulation_across_layers,
    "transpose_cache_retained_after_backward": test_transpose_cache_retained_after_backward,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP2 memory leak tests (standalone)")
    parser.add_argument("--test", required=True, choices=list(TESTS.keys()))
    parser.add_argument(
        "--recipe",
        type=str,
        default="DelayedScaling",
        choices=[
            "DelayedScaling",
            "Float8CurrentScaling",
            "Float8BlockScaling",
            "MXFP8BlockScaling",
            "NVFP4BlockScaling",
        ],
    )
    parser.add_argument("--quantized-model-init", action="store_true", default=False)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    _PARAMETRIZED_TESTS = {
        "fp8_temp_accumulation_across_layers",
        "transpose_cache_retained_after_backward",
    }

    try:
        test_fn = TESTS[args.test]
        if args.test in _PARAMETRIZED_TESTS:
            test_fn(args.recipe, args.quantized_model_init)
        else:
            test_fn()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
