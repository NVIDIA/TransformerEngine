# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for fused MoE router"""
from enum import IntEnum
from functools import partial

import jax.numpy as jnp
from jax import dtypes, ffi

from .base import BasePrimitive, register_primitive

__all__ = [
    "ScoreFunction",
    "fused_topk_with_score_function_fwd",
    "fused_topk_with_score_function_bwd",
    "fused_moe_aux_loss_fwd",
    "fused_moe_aux_loss_bwd",
]


class ScoreFunction(IntEnum):
    SIGMOID = 0
    SOFTMAX = 1


# =========================================== ==================================
# Fused Top-K with Score Function - Forward
# =============================================================================

class FusedTopkWithScoreFunctionFwdPrimitive(BasePrimitive):
    """
    Fused Top-K with Score Function Forward Primitive.
    Computes score_function(logits) -> top-k -> probs, routing_map.
    When compute_aux_scores=1, instead computes clean scores for aux loss.
    """

    name = "te_fused_topk_with_score_function_forward_ffi"
    multiple_results = True
    impl_static_args = (2, 3, 4, 5, 6, 7, 8)  # topk, use_pre_softmax, num_groups, group_topk, scaling_factor, score_function, compute_aux_scores
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        logits_aval,
        expert_bias_aval,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        compute_aux_scores,
    ):
        """Abstract evaluation: describe output shapes and dtypes."""
        del expert_bias_aval, topk, use_pre_softmax, num_groups, group_topk
        del scaling_factor, score_function, compute_aux_scores
        i_dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        i_shape = logits_aval.shape
        probs_aval = logits_aval.update(shape=i_shape, dtype=i_dtype)
        routing_map_aval = logits_aval.update(shape=i_shape, dtype=jnp.bool_)
        intermediate_aval = logits_aval.update(shape=i_shape, dtype=i_dtype)
        return probs_aval, routing_map_aval, intermediate_aval

    @staticmethod
    def lowering(
        ctx,
        logits,
        expert_bias,
        *,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        compute_aux_scores,
    ):
        return ffi.ffi_lowering(FusedTopkWithScoreFunctionFwdPrimitive.name)(
            ctx,
            logits,
            expert_bias,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            compute_aux_scores=compute_aux_scores,
        )

    @staticmethod
    def impl(
        logits,
        expert_bias,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        compute_aux_scores,
    ):
        assert FusedTopkWithScoreFunctionFwdPrimitive.inner_primitive is not None
        return FusedTopkWithScoreFunctionFwdPrimitive.inner_primitive.bind(
            logits,
            expert_bias,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            compute_aux_scores=compute_aux_scores,
        )

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        compute_aux_scores,
    ):
        assert FusedTopkWithScoreFunctionFwdPrimitive.outer_primitive is not None
        logits, expert_bias = batched_args
        logits_bdim, _ = batch_dims
        return (
            FusedTopkWithScoreFunctionFwdPrimitive.outer_primitive.bind(
                logits,
                expert_bias,
                topk=topk,
                use_pre_softmax=use_pre_softmax,
                num_groups=num_groups,
                group_topk=group_topk,
                scaling_factor=scaling_factor,
                score_function=score_function,
                compute_aux_scores=compute_aux_scores,
            ),
            (logits_bdim, logits_bdim, logits_bdim),
        )

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "num_tokens num_experts, num_experts -> num_tokens num_experts, num_tokens num_experts, num_tokens num_experts"


register_primitive(FusedTopkWithScoreFunctionFwdPrimitive)


# =============================================================================
# Fused Top-K with Score Function - Backward
# =============================================================================


class FusedTopkWithScoreFunctionBwdPrimitive(BasePrimitive):
    """
    Fused Top-K with Score Function Backward Primitive.
    When compute_aux_scores=1, runs the score-for-aux-loss backward instead.
    """

    name = "te_fused_topk_with_score_function_backward_ffi"
    multiple_results = False
    impl_static_args = (3, 4, 5, 6, 7)  # topk, use_pre_softmax, scaling_factor, score_function, compute_aux_scores
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        routing_map_aval,
        intermediate_aval,
        grad_probs_aval,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        compute_aux_scores,
    ):
        del topk, use_pre_softmax, scaling_factor, score_function
        del compute_aux_scores, routing_map_aval
        return intermediate_aval.update(
            shape=intermediate_aval.shape,
            dtype=dtypes.canonicalize_dtype(grad_probs_aval.dtype),
        )

    @staticmethod
    def lowering(
        ctx,
        routing_map,
        intermediate,
        grad_probs,
        *,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        compute_aux_scores,
    ):
        return ffi.ffi_lowering(FusedTopkWithScoreFunctionBwdPrimitive.name)(
            ctx,
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=scaling_factor,
            score_function=score_function,
            compute_aux_scores=compute_aux_scores,
        )

    @staticmethod
    def impl(
        routing_map,
        intermediate,
        grad_probs,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        compute_aux_scores,
    ):
        assert FusedTopkWithScoreFunctionBwdPrimitive.inner_primitive is not None
        return FusedTopkWithScoreFunctionBwdPrimitive.inner_primitive.bind(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=scaling_factor,
            score_function=score_function,
            compute_aux_scores=compute_aux_scores,
        )

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        compute_aux_scores,
    ):
        assert FusedTopkWithScoreFunctionBwdPrimitive.outer_primitive is not None
        routing_map, intermediate, grad_probs = batched_args
        _, _, grad_probs_bdim = batch_dims
        return (
            FusedTopkWithScoreFunctionBwdPrimitive.outer_primitive.bind(
                routing_map,
                intermediate,
                grad_probs,
                topk=topk,
                use_pre_softmax=use_pre_softmax,
                scaling_factor=scaling_factor,
                score_function=score_function,
                compute_aux_scores=compute_aux_scores,
            ),
            grad_probs_bdim,
        )

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "num_tokens num_experts, num_tokens num_experts, num_tokens num_experts -> num_tokens num_experts"


register_primitive(FusedTopkWithScoreFunctionBwdPrimitive)


# =============================================================================
# Fused MoE Aux Loss - Forward
# =============================================================================


class FusedMoEAuxLossFwdPrimitive(BasePrimitive):
    """
    Fused MoE Aux Loss Forward Primitive.
    """

    name = "te_fused_moe_aux_loss_forward_ffi"
    multiple_results = True
    impl_static_args = (2, 3, 4, 5)  # total_num_tokens, num_experts, topk, coeff
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(probs_aval, tokens_per_expert_aval, total_num_tokens, num_experts, topk, coeff):
        del total_num_tokens, num_experts, topk, coeff, tokens_per_expert_aval
        i_dtype = dtypes.canonicalize_dtype(probs_aval.dtype)
        aux_loss_aval = probs_aval.update(shape=(1,), dtype=i_dtype)
        const_buf_aval = probs_aval.update(shape=(1,), dtype=jnp.float32)
        return aux_loss_aval, const_buf_aval

    @staticmethod
    def lowering(ctx, probs, tokens_per_expert, *, total_num_tokens, num_experts, topk, coeff):
        return ffi.ffi_lowering(FusedMoEAuxLossFwdPrimitive.name)(
            ctx,
            probs,
            tokens_per_expert,
            total_num_tokens=total_num_tokens,
            num_experts=num_experts,
            topk=topk,
            coeff=coeff,
        )

    @staticmethod
    def impl(probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff):
        assert FusedMoEAuxLossFwdPrimitive.inner_primitive is not None
        return FusedMoEAuxLossFwdPrimitive.inner_primitive.bind(
            probs,
            tokens_per_expert,
            total_num_tokens=total_num_tokens,
            num_experts=num_experts,
            topk=topk,
            coeff=coeff,
        )

    @staticmethod
    def batcher(
        batched_args, batch_dims, *, total_num_tokens, num_experts, topk, coeff
    ):
        assert FusedMoEAuxLossFwdPrimitive.outer_primitive is not None
        probs, tokens_per_expert = batched_args
        probs_bdim, _ = batch_dims
        return (
            FusedMoEAuxLossFwdPrimitive.outer_primitive.bind(
                probs,
                tokens_per_expert,
                total_num_tokens=total_num_tokens,
                num_experts=num_experts,
                topk=topk,
                coeff=coeff,
            ),
            (probs_bdim, probs_bdim),
        )

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "num_tokens num_experts, num_experts -> aux_loss_one, const_buf_one"


register_primitive(FusedMoEAuxLossFwdPrimitive)


# =============================================================================
# Fused MoE Aux Loss - Backward
# =============================================================================


class FusedMoEAuxLossBwdPrimitive(BasePrimitive):
    """
    Fused MoE Aux Loss Backward Primitive.
    """

    name = "te_fused_moe_aux_loss_backward_ffi"
    multiple_results = False
    impl_static_args = (3, 4)  # num_rows, num_cols
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(const_buf_aval, tokens_per_expert_aval, grad_aux_loss_aval, num_rows, num_cols):
        del const_buf_aval, tokens_per_expert_aval
        out_dtype = dtypes.canonicalize_dtype(grad_aux_loss_aval.dtype)
        return grad_aux_loss_aval.update(
            shape=(num_rows, num_cols),
            dtype=out_dtype,
        )

    @staticmethod
    def lowering(ctx, const_buf, tokens_per_expert, grad_aux_loss, *, num_rows, num_cols):
        return ffi.ffi_lowering(FusedMoEAuxLossBwdPrimitive.name)(
            ctx,
            const_buf,
            tokens_per_expert,
            grad_aux_loss,
            num_rows=num_rows,
            num_cols=num_cols,
        )

    @staticmethod
    def impl(const_buf, tokens_per_expert, grad_aux_loss, num_rows, num_cols):
        assert FusedMoEAuxLossBwdPrimitive.inner_primitive is not None
        return FusedMoEAuxLossBwdPrimitive.inner_primitive.bind(
            const_buf, tokens_per_expert, grad_aux_loss, num_rows=num_rows, num_cols=num_cols
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, num_rows, num_cols):
        assert FusedMoEAuxLossBwdPrimitive.outer_primitive is not None
        const_buf, tokens_per_expert, grad_aux_loss = batched_args
        _, _, grad_bdim = batch_dims
        return (
            FusedMoEAuxLossBwdPrimitive.outer_primitive.bind(
                const_buf, tokens_per_expert, grad_aux_loss, num_rows=num_rows, num_cols=num_cols
            ),
            grad_bdim,
        )

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "const_buf_one, num_experts, grad_one -> num_tokens num_experts"


register_primitive(FusedMoEAuxLossBwdPrimitive)


# =============================================================================
# Public API functions
# =============================================================================


def fused_topk_with_score_function_fwd(
    logits: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    scaling_factor: float,
    score_function,
    expert_bias: jnp.ndarray,
    compute_aux_scores: bool = False,
):
    """
    Fused top-k with score function forward pass.

    When compute_aux_scores=True, runs the clean score-for-aux-loss kernel
    instead of the full top-k kernel (expert_bias, use_pre_softmax, num_groups,
    group_topk, and scaling_factor are ignored).

    Parameters
    ----------
    logits : jnp.ndarray
        [num_tokens, num_experts] logits from gating GEMM.
    topk : int
        Number of top experts to select.
    use_pre_softmax : bool
        If True, apply softmax before top-k.
    num_groups : int
        Number of groups for grouped top-k (1 to disable).
    group_topk : int
        Top-k at group level (1 to disable).
    scaling_factor : float
        Scaling factor for output probs.
    score_function : ScoreFunction
        ScoreFunction.SOFTMAX or ScoreFunction.SIGMOID.
    expert_bias : jnp.ndarray
        Expert bias (only used with sigmoid). Pass empty array if unused.
    compute_aux_scores : bool
        If True, compute clean scores for aux loss instead of full top-k.

    Returns
    -------
    probs_or_scores, routing_map, intermediate_output
    """
    return FusedTopkWithScoreFunctionFwdPrimitive.outer_primitive.bind(
        logits,
        expert_bias,
        topk=int(topk),
        use_pre_softmax=int(use_pre_softmax),
        num_groups=int(num_groups),
        group_topk=int(group_topk),
        scaling_factor=float(scaling_factor),
        score_function=int(score_function),
        compute_aux_scores=int(compute_aux_scores),
    )


def fused_topk_with_score_function_bwd(
    routing_map: jnp.ndarray,
    intermediate_output: jnp.ndarray,
    grad_probs: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    score_function,
    compute_aux_scores: bool = False,
):
    """
    Fused top-k with score function backward pass.

    When compute_aux_scores=True, routing_map is ignored and the
    score-for-aux-loss backward kernel is used instead.
    """
    return FusedTopkWithScoreFunctionBwdPrimitive.outer_primitive.bind(
        routing_map,
        intermediate_output,
        grad_probs,
        topk=int(topk),
        use_pre_softmax=int(use_pre_softmax),
        scaling_factor=float(scaling_factor),
        score_function=int(score_function),
        compute_aux_scores=int(compute_aux_scores),
    )


def fused_moe_aux_loss_fwd(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    total_num_tokens: int,
    num_experts: int,
    topk: int,
    coeff: float,
):
    """
    Fused MoE aux loss forward pass.

    Returns
    -------
    aux_loss, const_buf
    """
    return FusedMoEAuxLossFwdPrimitive.outer_primitive.bind(
        probs,
        tokens_per_expert,
        total_num_tokens=int(total_num_tokens),
        num_experts=int(num_experts),
        topk=int(topk),
        coeff=float(coeff),
    )


def fused_moe_aux_loss_bwd(
    const_buf: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    grad_aux_loss: jnp.ndarray,
    num_rows: int,
    num_cols: int,
):
    """
    Fused MoE aux loss backward pass.
    """
    return FusedMoEAuxLossBwdPrimitive.outer_primitive.bind(
        const_buf,
        tokens_per_expert,
        grad_aux_loss,
        num_rows=int(num_rows),
        num_cols=int(num_cols),
    )
