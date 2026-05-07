# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Quantized model initialization with FusedAdam and gradient accumulation fusion.

Demonstrates three Transformer Engine features working together:

1. ``quantized_model_init`` -- Initialize a model with low-precision (FP8)
   parameters, avoiding the memory cost of storing both high-precision and
   quantized copies of every weight.

2. ``FusedAdam`` with master weights -- Maintain FP32 master copies of the
   weights inside the optimizer so that the training update retains full
   precision despite the model parameters being FP8.

3. Gradient accumulation fusion -- Use ``fuse_wgrad_accumulation=True``
   together with per-parameter ``main_grad`` buffers so that weight
   gradients are accumulated directly in FP32 via Tensor Cores, avoiding a
   separate FP8-to-FP32 cast kernel.

Usage::

    python main.py
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

# ── Configuration ──────────────────────────────────────────────────────
HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
SEQ_LEN = 32
BATCH_SIZE = 2
NUM_STEPS = 5
DTYPE = torch.bfloat16


def main():
    # ── 1. Create model with quantized parameters ─────────────────────
    #
    # Inside quantized_model_init, TransformerEngine modules store only the
    # FP8 quantized copy of each parameter (a Float8Tensor), eliminating the
    # memory overhead of a high-precision shadow copy.
    with te.quantized_model_init(enabled=True):
        model = te.TransformerLayer(
            HIDDEN_SIZE,
            FFN_HIDDEN_SIZE,
            NUM_ATTENTION_HEADS,
            fuse_wgrad_accumulation=True,
            fuse_qkv_params=True,  # required for fuse_wgrad_accumulation
            params_dtype=DTYPE,
            hidden_dropout=0.0,  # disable dropout for this synthetic example
            attention_dropout=0.0,
        )

    # Verify that linear-layer weight parameters are quantized.
    # Biases and LayerNorm parameters are *not* quantized.
    quantized_count = 0
    for name, param in model.named_parameters():
        if isinstance(param, QuantizedTensor):
            quantized_count += 1
    assert quantized_count > 0, "No QuantizedTensor parameters found"
    print(f"Found {quantized_count} QuantizedTensor (FP8) weight parameters.")

    # ── 2. Allocate main_grad buffers (FP32) ──────────────────────────
    #
    # fuse_wgrad_accumulation causes weight-gradient GEMMs to write directly
    # into ``param.main_grad`` in FP32 (via Tensor Core accumulation).
    # Non-weight parameters (e.g. LayerNorm) still receive gradients through
    # the normal ``param.grad`` path.
    for param in model.parameters():
        param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device=param.device)

    # ── 3. Optimizer with FP32 master weights ─────────────────────────
    #
    # use_decoupled_grad=True tells FusedAdam to read gradients from
    # ``param.decoupled_grad`` instead of ``param.grad``.  This avoids
    # the dtype-mismatch error that would occur when assigning FP32
    # gradients to bfloat16 parameters via ``.grad``.
    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
        use_decoupled_grad=True,
    )

    # ── 4. Training loop ──────────────────────────────────────────────
    #
    # Use a fixed synthetic dataset so that loss decreases over steps.
    x = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=DTYPE, device="cuda")
    target = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=DTYPE, device="cuda")

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        for param in model.parameters():
            param.main_grad.zero_()

        # Forward pass inside autocast to enable FP8 compute.
        with te.autocast(enabled=True):
            output = model(x)

        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Consolidate gradients into main_grad.
        #   * Weight params with fuse_wgrad_accumulation: backward already
        #     accumulated the gradient directly into main_grad (FP32).
        #   * Other params (e.g. LayerNorm): autograd set param.grad.
        for param in model.parameters():
            if param.grad is not None:
                param.main_grad.copy_(param.grad)
                param.grad = None

        # Expose main_grad as decoupled_grad so FusedAdam can read it.
        for param in model.parameters():
            param.decoupled_grad = param.main_grad

        optimizer.step()
        print(f"  Step {step}: loss = {loss.item():.6f}")

    # ── 5. Post-training assertions ───────────────────────────────────
    print("\nVerifying invariants ...")

    # Optimizer states.
    for param in model.parameters():
        state = optimizer.state[param]
        if "master_param" in state:
            master = state["master_param"]
            assert (
                master.dtype == torch.float32
            ), f"Master weight dtype {master.dtype}, expected float32"
        assert state["exp_avg"].dtype == torch.float32, "exp_avg should be float32"
        assert state["exp_avg_sq"].dtype == torch.float32, "exp_avg_sq should be float32"

    # main_grad buffers.
    for param in model.parameters():
        assert param.main_grad.dtype == torch.float32, "main_grad should be float32"

    print("All assertions passed!")
    print("  - Linear weight parameters: QuantizedTensor (FP8)")
    print("  - Optimizer master weights: float32")
    print("  - Optimizer states (exp_avg, exp_avg_sq): float32")
    print("  - Gradient accumulation buffers (main_grad): float32")


if __name__ == "__main__":
    main()
