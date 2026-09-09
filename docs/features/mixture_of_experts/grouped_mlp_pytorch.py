# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_GROUPED_MLP_PYTORCH
import transformer_engine.pytorch as te

# FC1 produces gate and value features for the GLU.
expert_mlp = te.ops.Sequential(
    te.ops.GroupedLinear(num_experts, hidden_size, 2 * ffn_hidden_size),
    te.ops.ScaledSwiGLU(),  # or ScaledClampedQGeGLU; ScaledSReLU for the unary variant
    te.ops.GroupedLinear(num_experts, ffn_hidden_size, hidden_size),
)

# The fuser selects GroupedMLP_CuTeGEMMGLU for supported configurations.
# FC1 fuses the activation; FC2 runs as a separate grouped GEMM.
# END_GROUPED_MLP_PYTORCH
