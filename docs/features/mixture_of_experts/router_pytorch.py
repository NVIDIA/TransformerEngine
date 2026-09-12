# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_ROUTER_PYTORCH
from transformer_engine.pytorch.router import fused_topk_with_score_function

# logits: [num_tokens, num_experts], produced by the gating (router) projection.
#
# Select the top-k experts for each token and return their routing weights. The
# score function and the top-k selection run in a single fused kernel (all math
# is done in fp32 internally for numerical stability).
probs, routing_map = fused_topk_with_score_function(
    logits,
    topk=2,
    use_pre_softmax=False,  # softmax after top-k; True selects softmax-then-top-k
    num_groups=None,  # set with group_topk to enable grouped (device-limited) routing
    group_topk=None,
    scaling_factor=None,  # optional scalar multiplied into the returned probs
    score_function="softmax",  # "softmax", "sigmoid" or "sqrtsoftplus"
    expert_bias=None,  # [num_experts] selection bias, only with score_function="sigmoid"
)

# probs:       [num_tokens, num_experts], non-zero only at the selected experts.
#              Pass directly to moe_unpermute as merging_probs.
# routing_map: [num_tokens, num_experts] bool mask, True at the selected experts.
#              Cast to int32 and pass to moe_permute.
# END_ROUTER_PYTORCH


# START_ROUTER_AUX_PYTORCH
from transformer_engine.pytorch.router import (
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
)

# The load-balancing auxiliary loss is computed from the *dense* scores over all
# experts (not from the sparse top-k probs above), so its gradient reaches every
# expert's logit. fused_compute_score_for_moe_aux_loss returns those dense scores
# together with the same routing map.
routing_map, scores = fused_compute_score_for_moe_aux_loss(
    logits,
    topk=2,
    score_function="softmax",
)
tokens_per_expert = routing_map.sum(dim=0)  # [num_experts]

aux_loss = fused_moe_aux_loss(
    scores,
    tokens_per_expert,
    total_num_tokens=logits.shape[0],
    num_experts=logits.shape[1],
    topk=2,
    coeff=1e-2,  # loss weight; add aux_loss to the training loss
)
# END_ROUTER_AUX_PYTORCH
