# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused Router operations for JAX/XLA."""

from functools import partial
from typing import Tuple, Optional

import jax

from . import cpp_extensions as tex


def fused_topk_with_score_function(
    logits: jax.Array,
    expert_bias: Optional[jax.Array] = None,
    topk: int = 2,
    use_pre_softmax: bool = False,
    num_groups: int = 1,
    group_topk: int = 1,
    scaling_factor: float = 1.0,
    score_function: str = "sigmoid",
) -> Tuple[jax.Array, jax.Array, jax.Array]:

    score_function_int = tex.map_score_function(score_function)

    if expert_bias is not None and score_function != "sigmoid":
        raise ValueError("expert_bias is only supported with sigmoid score function")

    if score_function == "sigmoid":
        use_pre_softmax = False  # Pre-softmax only applies to softmax

    outputs = _fused_topk_with_score_function(
        logits,
        expert_bias,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function_int,
    )

    return outputs


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7))
def _fused_topk_with_score_function(
    logits,
    expert_bias,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
):
    outputs, _ = _fused_topk_fwd_rule(
        logits,
        expert_bias,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
    )
    return outputs


def _fused_topk_fwd_rule(
    logits,
    expert_bias,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
):
    probs, routing_map, intermediate_output = tex.fused_topk_with_score_function_fwd(
        logits,
        expert_bias,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
    )
    return (probs, routing_map, intermediate_output), (routing_map, intermediate_output)


def _fused_topk_bwd_rule(
    topk, use_pre_softmax, num_groups, group_topk, scaling_factor, score_function, ctx, grads
):
    del num_groups, group_topk
    routing_map, intermediate_output = ctx
    grad_probs, _, _ = grads

    grad_logits = tex.fused_topk_with_score_function_bwd(
        routing_map,
        intermediate_output,
        grad_probs,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
    )

    return (grad_logits, None)


_fused_topk_with_score_function.defvjp(_fused_topk_fwd_rule, _fused_topk_bwd_rule)
