# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_ROUTER_JAX
from transformer_engine.jax.router import fused_topk_with_score_function

# logits: [num_tokens, num_experts], produced by the gating (router) projection.
#
# Select the top-k experts for each token. The score function and the top-k
# selection run in a single fused kernel. Most arguments have defaults, so a
# basic call only needs the logits, topk and score_function.
probs, routing_map = fused_topk_with_score_function(
    logits,
    topk=2,
    score_function="softmax",  # "softmax" or "sigmoid"
)

# probs:       [num_tokens, num_experts], non-zero only at the selected experts.
#              Pass to token_combine as merging_probs.
# routing_map: [num_tokens, num_experts] bool mask. Cast to int32 for token_dispatch.
# END_ROUTER_JAX


# START_ROUTER_AUX_JAX
from transformer_engine.jax.router import fused_moe_aux_loss

# The load-balancing auxiliary loss uses the dense scores over all experts. In
# JAX the same router function returns them when compute_aux_scores=True (the
# bias / grouping / scaling arguments are ignored in this mode).
scores, routing_map = fused_topk_with_score_function(
    logits,
    topk=2,
    score_function="softmax",
    compute_aux_scores=True,
)
tokens_per_expert = routing_map.sum(axis=0)  # [num_experts]

aux_loss = fused_moe_aux_loss(
    scores,
    tokens_per_expert,
    topk=2,
    coeff=1e-2,  # loss weight; add aux_loss to the training loss
)
# END_ROUTER_AUX_JAX
