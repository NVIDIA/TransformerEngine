# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FSDP2 distributed training with quantized model initialization.

Extends the single-GPU ``main.py`` example to multi-GPU training using
PyTorch-native FSDP2 (``fully_shard``).  The script demonstrates:

1. **Meta-device initialization** -- Model parameters are created on the
   ``meta`` device (zero memory), then FSDP2 sharding is applied, and
   finally ``reset_parameters()`` materializes and quantizes only the
   local shards on each rank's GPU.
2. ``quantized_model_init`` -- Flags the model for FP8 weight initialization
   (actual quantization happens in ``reset_parameters`` after sharding).
3. ``preserve_high_precision_init_val`` -- Keeps the original BF16 weight
   values on CPU so they can seed the optimizer's FP32 master weights,
   avoiding the precision loss of round-tripping through FP8.
4. ``fully_shard`` -- PyTorch FSDP2 sharding of each TransformerLayer.
5. ``FusedAdam`` with FP32 master weights for full-precision training updates.

.. note::
   ``fuse_wgrad_accumulation`` is **not** used here.  That feature writes
   weight gradients directly into ``main_grad`` buffers, bypassing the
   autograd gradient flow.  FSDP2 requires gradients to go through its
   reduce-scatter, so ``fuse_wgrad_accumulation`` needs Megatron-Core's
   FSDP integration (which provides ``get_main_grad()``).

Usage::

    torchrun --nproc-per-node 2 fully_shard.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

import transformer_engine.pytorch as te
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

# ── Configuration ────────────────────────────────────────────────────
HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
NUM_LAYERS = 3
SEQ_LEN = 32
BATCH_PER_RANK = 2
NUM_STEPS = 5
# DTYPE is used for both params_dtype and activation tensors in this example.
# float32 is chosen for params_dtype so that the high-precision init values
# (which seed the optimizer's FP32 master weights) avoid a lossy BF16→FP8→FP32
# round-trip.  Using float32 for activations as well keeps the example simple;
# in production you would typically use BF16 activations inside te.autocast().
DTYPE = torch.float32


def dist_print(msg):
    """Print only on rank 0."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg)


def main():
    # ── 1. Distributed setup ─────────────────────────────────────────
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # ── 2. Create model on meta device (zero memory) ─────────────────
    # quantized_model_init flags parameters for FP8 quantization.
    # preserve_high_precision_init_val=True saves the original BF16
    # values on CPU so they can seed optimizer master weights later,
    # avoiding the precision loss of dequantizing from FP8.
    # We set DTYPE to float32 since these weights will actually be initialized as FP8,
    # but we want to seed the optimizer states (which will be in FP32) with the FP32 values.
    with te.quantized_model_init(enabled=True, preserve_high_precision_init_val=True):
        model = torch.nn.Sequential(
            *[
                te.TransformerLayer(
                    HIDDEN_SIZE,
                    FFN_HIDDEN_SIZE,
                    NUM_ATTENTION_HEADS,
                    fuse_qkv_params=True,
                    params_dtype=DTYPE,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                    device="meta",
                )
                for _ in range(NUM_LAYERS)
            ]
        )
    dist_print("Model created on meta device (zero GPU memory).")

    # ── 3. FSDP2 sharding ───────────────────────────────────────────
    # Apply sharding to the meta-device model.  FSDP2 wraps parameters
    # as DTensors but no GPU memory is allocated yet.
    mesh = DeviceMesh("cuda", list(range(world_size)))
    for child in model.children():
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)
    dist_print("FSDP2 sharding applied to meta-device model.")

    # ── 4. Materialize parameters on GPU ─────────────────────────────
    # reset_parameters() on each TE module materializes the local shard
    # on CUDA, applies weight initialization, and quantizes to FP8.
    # Because preserve_high_precision_init_val=True, the pre-quantization
    # BF16 values are saved on CPU for each local shard.
    for module in model.modules():
        if isinstance(module, TransformerEngineBaseModule):
            module.reset_parameters()
    dist_print("Parameters materialized on GPU.")

    # ── 5. Optimizer with FP32 master weights ────────────────────────
    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    # ── 6. Seed master weights from high-precision init values ───────
    # By default, FusedAdam initializes master weights by dequantizing
    # the FP8 parameters, which introduces quantization noise.  Instead,
    # we seed them from the original BF16 init values preserved in step 2.
    for name, param in model.named_parameters():
        optimizer.initialize_state(param, store_param_remainders=False)
        local = param._local_tensor if isinstance(param, DTensor) else param
        if isinstance(local, QuantizedTensor):
            hp_val = local.get_high_precision_init_val()
            assert hp_val.dtype == DTYPE, f"HP val dtype {hp_val.dtype}, expected {DTYPE}"
            optimizer.set_scaled_state(
                param, "master_param", hp_val.to(device=device, dtype=torch.float32)
            )
            local.clear_high_precision_init_val()

    dist_print("Optimizer master weights seeded from high-precision init values.")

    # ── 7. Training loop ─────────────────────────────────────────────
    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=DTYPE, device=device)
    target = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=DTYPE, device=device)

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)

        with te.autocast(enabled=True):
            output = model(x)

        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        dist_print(f"  Step {step}: loss = {loss.item():.6f}")

    # ── 8. Distributed checkpoint: save and load ─────────────────────
    # torch.distributed.checkpoint (DCP) saves sharded state — each rank
    # writes only its local shard, preserving FP8 compute weights and
    # the full optimizer state (master weights, moments, step count).
    import torch.distributed.checkpoint as dcp

    checkpoint_dir = "/tmp/te_fsdp2_example_checkpoint"
    dist_print(f"\nSaving distributed checkpoint to {checkpoint_dir} ...")

    dcp.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        checkpoint_id=checkpoint_dir,
    )
    dist_print("  Checkpoint saved (FP8 weights + optimizer state).")

    # Load checkpoint back.  Provide empty state dict containers with the
    # same structure; DCP fills them from the saved files.
    state_to_load = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.load(state_to_load, checkpoint_id=checkpoint_dir)
    model.load_state_dict(state_to_load["model"])
    optimizer.load_state_dict(state_to_load["optimizer"])
    dist_print("  Checkpoint loaded — FP8 weights and optimizer state restored.")

    # Verify training continues after checkpoint load.
    optimizer.zero_grad(set_to_none=True)
    with te.autocast(enabled=True):
        output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    dist_print(f"  Post-checkpoint training step: loss = {loss.item():.6f}")

    # ── 9. Save full-precision (FP32) model to safetensors ───────────
    # For inference or fine-tuning you typically want FP32 weights, not
    # FP8 compute weights.  The optimizer's master weight copies are the
    # authoritative FP32 values (more precise than dequantizing FP8).
    # All ranks must participate in gathering; only rank 0 saves.
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    full_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)

    full_model_state = get_model_state_dict(model, options=full_opts)
    full_opt_state = get_optimizer_state_dict(model, optimizer, options=full_opts)

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        fp32_state = {}
        opt_param_states = full_opt_state.get("state", {})

        for key, value in full_model_state.items():
            if key in opt_param_states and "master_param" in opt_param_states[key]:
                # Prefer optimizer's FP32 master weight.
                fp32_state[key] = opt_param_states[key]["master_param"].float()
            elif isinstance(value, te.QuantizedTensor):
                # Fallback: dequantize FP8 → FP32.
                fp32_state[key] = value.dequantize().float()
            else:
                # Non-FP8 params (e.g. LayerNorm weights): cast to FP32.
                fp32_state[key] = value.float()

        save_path = "/tmp/te_fsdp2_example_model_fp32.safetensors"
        save_file(fp32_state, save_path)
        dist_print(f"\nSaved FP32 model ({len(fp32_state)} params) to {save_path}")

    dist.barrier()  # wait for rank 0 to finish file I/O
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
