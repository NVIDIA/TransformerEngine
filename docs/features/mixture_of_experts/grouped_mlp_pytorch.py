# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_GROUPED_MLP_PYTORCH
import transformer_engine.pytorch as te

# Build the expert MLP from the operation-based API: two grouped linear layers
# with a scaled GLU activation in between. FC1 produces 2 * ffn_hidden_size
# features (gate and value) for the GLU.
expert_mlp = te.ops.Sequential(
    te.ops.GroupedLinear(num_experts, hidden_size, 2 * ffn_hidden_size),
    te.ops.ScaledSwiGLU(),  # or ScaledClampedQGeGLU; ScaledSReLU for the unary variant
    te.ops.GroupedLinear(num_experts, ffn_hidden_size, hidden_size),
)

# When this sequence runs under a block-scaled recipe (MXFP8 or NVFP4) on a
# Blackwell (SM100) GPU with NVTE_CUTEDSL_FUSED_GROUPED_MLP=1, the operation
# fuser transparently replaces the three ops with a single fused grouped-MLP
# kernel (GroupedMLP_CuTeGEMMGLU). No code change is needed to opt in.
# END_GROUPED_MLP_PYTORCH
