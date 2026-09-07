# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_GROUPED_LINEAR_PYTORCH
import torch
import transformer_engine.pytorch as te

# x:             [sum(m_splits), hidden_size], expert-contiguous tokens
# m_splits:      list[int] of length num_experts; m_splits[i] is the number
#                of routed tokens for expert i
# torch_experts: list[torch.nn.Linear] of length num_experts, one per expert
#                (used only by the baseline loop below)
x_by_expert = torch.split(x, m_splits, dim=0)

# Baseline: one Linear call per expert.
loop_out = torch.cat(
    [expert(x_i) for expert, x_i in zip(torch_experts, x_by_expert)],
    dim=0,
)

# Transformer Engine: one grouped linear call.
grouped_linear = te.GroupedLinear(
    num_experts,
    hidden_size,
    ffn_hidden_size,
    bias=True,
    params_dtype=torch.bfloat16,
).cuda()
grouped_out = grouped_linear(x, m_splits)
# END_GROUPED_LINEAR_PYTORCH
