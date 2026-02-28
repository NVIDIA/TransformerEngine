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
3. ``fully_shard`` -- PyTorch FSDP2 sharding of each TransformerLayer.
4. ``FusedAdam`` with FP32 master weights for full-precision training updates.

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
from transformer_engine.pytorch import QuantizedTensor
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

# ── Configuration (matches main.py) ──────────────────────────────────
HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
NUM_LAYERS = 3
SEQ_LEN = 32
BATCH_PER_RANK = 2
NUM_STEPS = 5
DTYPE = torch.bfloat16


def dist_print(msg):
    """Print only on rank 0."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg)


def main():
    # ── 1. Distributed setup ─────────────────────────────────────────
    assert "TORCHELASTIC_RUN_ID" in os.environ, (
        "This script must be launched with torchrun, e.g.:\n"
        "  torchrun --nproc-per-node 2 fully_shard.py"
    )
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # ── 2. Create model on meta device (zero memory) ────────────────
    # quantized_model_init sets the flag for FP8 weight initialization,
    # but with device="meta" no actual memory is allocated yet.
    with te.quantized_model_init(enabled=True):
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

    # Verify all parameters are on meta device (no GPU memory used).
    for name, param in model.named_parameters():
        assert param.device == torch.device("meta"), f"{name} is not on meta device"
    dist_print("Model created on meta device (zero GPU memory).")

    # ── 3. FSDP2 sharding ────────────────────────────────────────────
    # Apply sharding to the meta-device model. FSDP2 wraps parameters
    # as DTensors but no GPU memory is allocated yet.
    mesh = DeviceMesh("cuda", list(range(world_size)))
    for child in model.children():
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)
    dist_print("FSDP2 sharding applied to meta-device model.")

    # ── 4. Materialize parameters on GPU ──────────────────────────────
    # reset_parameters() on each TE module materializes the local shard
    # on CUDA, applies weight initialization, and quantizes to FP8.
    for module in model.modules():
        if isinstance(module, TransformerEngineBaseModule):
            module.reset_parameters()

    # Post-materialization verification.
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} is not a DTensor after sharding"
    qt_count = sum(
        1
        for _, p in model.named_parameters()
        if isinstance(p, DTensor) and isinstance(p._local_tensor, QuantizedTensor)
    )
    assert qt_count > 0, "No QuantizedTensor local tensors after materialization"
    dist_print(
        f"Parameters materialized: {qt_count} FP8 (QuantizedTensor) weight params "
        "wrapped in DTensors."
    )

    # ── 5. Optimizer ─────────────────────────────────────────────────
    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )
    dist_print("Using FusedAdam with master_weights=True.")

    # ── 6. Training loop ─────────────────────────────────────────────
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

    # ── 7. Post-training assertions ──────────────────────────────────
    dist_print("\nVerifying invariants ...")

    qt_after = 0
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} lost DTensor wrapping"
        if isinstance(param._local_tensor, QuantizedTensor):
            qt_after += 1
    assert qt_after > 0, "No QuantizedTensor local tensors after training"
    dist_print(f"  {qt_after} params still have QuantizedTensor local tensors.")

    # Optimizer states: master weights and moments should be float32.
    for param in model.parameters():
        state = optimizer.state[param]
        if "master_param" in state:
            assert (
                state["master_param"].dtype == torch.float32
            ), f"Master weight dtype {state['master_param'].dtype}, expected float32"
        assert state["exp_avg"].dtype == torch.float32, "exp_avg should be float32"
        assert state["exp_avg_sq"].dtype == torch.float32, "exp_avg_sq should be float32"

    dist_print("All assertions passed!")
    dist_print("  - Linear weight parameters: QuantizedTensor (FP8) wrapped in DTensor")
    dist_print("  - Optimizer master weights: float32")
    dist_print("  - Optimizer states (exp_avg, exp_avg_sq): float32")

    # ── 8. Distributed checkpoint: save and load ─────────────────────
    # torch.distributed.checkpoint (DCP) saves sharded state — each rank
    # writes only its local shard.  This preserves FP8 compute weights
    # and the full optimizer state (master weights, moments, step count).
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    # Use a fixed path so all ranks agree on the checkpoint location.
    checkpoint_dir = "/tmp/te_fsdp2_example_checkpoint"
    dist_print(f"\nSaving distributed checkpoint to {checkpoint_dir} ...")

    # Save sharded checkpoint. DCP handles DTensor shards natively —
    # each rank writes only its local shard to the filesystem.
    dcp.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        checkpoint_id=checkpoint_dir,
    )
    dist_print("  Checkpoint saved (FP8 weights + optimizer state).")

    # Load checkpoint back. Provide empty state dict containers with the
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

    full_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)

    full_model_state = get_model_state_dict(model, options=full_opts)
    full_opt_state = get_optimizer_state_dict(model, optimizer, options=full_opts)

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        fp32_state = {}
        opt_param_states = full_opt_state.get("state", {})

        for key, value in full_model_state.items():
            if key in opt_param_states and "master_param" in opt_param_states[key]:
                # Prefer optimizer's FP32 master weight (maintained throughout training).
                fp32_state[key] = opt_param_states[key]["master_param"].float()
            elif isinstance(value, QuantizedTensor):
                # Fallback: dequantize FP8 → FP32 (e.g. if master_weights was off).
                fp32_state[key] = value.dequantize().float()
            else:
                # Non-FP8 params (e.g. LayerNorm weights): cast to FP32.
                fp32_state[key] = value.float()

        save_path = "/tmp/te_fsdp2_example_model_fp32.safetensors"
        save_file(fp32_state, save_path)
        dist_print(f"\nSaved FP32 model ({len(fp32_state)} params) to {save_path}")

        # Quick verification: all saved tensors are float32.
        from safetensors.torch import load_file

        loaded = load_file(save_path)
        for k, v in loaded.items():
            assert v.dtype == torch.float32, f"{k}: expected float32, got {v.dtype}"
        dist_print(f"  Verified: all {len(loaded)} tensors are float32.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
