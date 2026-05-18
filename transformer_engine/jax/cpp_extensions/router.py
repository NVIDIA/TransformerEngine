# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for fused MoE router"""
from enum import IntEnum

import jax.numpy as jnp
from jax import dtypes, ffi
from jax.sharding import NamedSharding, PartitionSpec
from transformer_engine_jax import JAXX_Routing_Map_Format, JAXX_Score_Function

from .base import BasePrimitive, register_primitive
from .misc import get_padded_spec

__all__ = [
    "ScoreFunction",
    "RoutingMapFormat",
    "fused_topk_with_score_function_fwd",
    "fused_topk_with_score_function_bwd",
    "fused_moe_aux_loss_fwd",
    "fused_moe_aux_loss_bwd",
]


class ScoreFunction(IntEnum):
    """Score function enum for fused MoE router kernels, synced with C++ JAXX_Score_Function."""

    SIGMOID = int(JAXX_Score_Function.SIGMOID)
    SOFTMAX = int(JAXX_Score_Function.SOFTMAX)


class RoutingMapFormat(IntEnum):
    """Routing-map output layout, synced with C++ JAXX_Routing_Map_Format / NVTERoutingMapFormat.

    BYTEMAP   — bool/uint8 tensor of shape [num_tokens, num_experts].
    BITMAP_U8 — uint8 tensor of shape [num_tokens, ceil(num_experts/8)]; bit
                (e % 8) of byte (e / 8) of row t is 1 iff token t routes to
                expert e (LSB-first packing along the expert axis). This is the
                layout NCCL EP dispatch is planned to consume directly.
    """

    BYTEMAP = int(JAXX_Routing_Map_Format.BYTEMAP)
    BITMAP_U8 = int(JAXX_Routing_Map_Format.BITMAP_U8)


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
    impl_static_args = (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    )  # topk, use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,
    #   compute_aux_scores, routing_map_format
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
        routing_map_format,
    ):
        """Abstract evaluation: describe output shapes and dtypes."""
        del expert_bias_aval, topk, use_pre_softmax, num_groups, group_topk
        del scaling_factor, score_function, compute_aux_scores
        i_dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        i_shape = logits_aval.shape
        probs_aval = logits_aval.update(shape=i_shape, dtype=i_dtype)
        # routing_map shape/dtype depends on the format. In BITMAP_U8 mode the
        # expert axis is bit-packed LSB-first into uint8 bytes, so the trailing
        # dim becomes ceil(num_experts/8).
        if int(routing_map_format) == int(RoutingMapFormat.BITMAP_U8):
            packed_experts = (i_shape[-1] + 7) // 8
            routing_map_shape = (*i_shape[:-1], packed_experts)
            routing_map_aval = logits_aval.update(shape=routing_map_shape, dtype=jnp.uint8)
        else:
            routing_map_aval = logits_aval.update(shape=i_shape, dtype=jnp.bool_)
        # The CUDA kernel always uses float32 (CompType) for intermediate
        # computations (softmax/sigmoid values saved for backward).
        intermediate_aval = logits_aval.update(shape=i_shape, dtype=jnp.float32)
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
        routing_map_format,
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
            routing_map_format=routing_map_format,
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
        routing_map_format,
    ):
        if FusedTopkWithScoreFunctionFwdPrimitive.inner_primitive is None:
            raise RuntimeError(
                "FusedTopkWithScoreFunctionFwdPrimitive.inner_primitive has not been registered"
            )
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
            routing_map_format=routing_map_format,
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
        routing_map_format,
    ):
        if FusedTopkWithScoreFunctionFwdPrimitive.outer_primitive is None:
            raise RuntimeError(
                "FusedTopkWithScoreFunctionFwdPrimitive.outer_primitive has not been registered"
            )
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
                routing_map_format=routing_map_format,
            ),
            (logits_bdim, logits_bdim, logits_bdim),
        )

    @staticmethod
    def partition(
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        compute_aux_scores,
        routing_map_format,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        logits_spec = get_padded_spec(arg_infos[0])
        out_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec))
        # For bitmap mode the trailing dim is ceil(E/8) instead of E. We keep the
        # routing_map sharded the same way logits is along all non-trailing dims
        # and replicate the (now packed) expert axis to avoid sharding mid-byte.
        if int(routing_map_format) == int(RoutingMapFormat.BITMAP_U8):
            routing_spec = (*logits_spec[:-1], None) if len(logits_spec) >= 1 else logits_spec
        else:
            routing_spec = logits_spec
        routing_sharding = NamedSharding(mesh, PartitionSpec(*routing_spec))
        intermediate_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec))
        out_shardings = [out_sharding, routing_sharding, intermediate_sharding]
        arg_shardings = (arg_infos[0].sharding, arg_infos[1].sharding)

        def sharded_impl(logits, expert_bias):
            return FusedTopkWithScoreFunctionFwdPrimitive.impl(
                logits,
                expert_bias,
                topk,
                use_pre_softmax,
                num_groups,
                group_topk,
                scaling_factor,
                score_function,
                compute_aux_scores,
                routing_map_format,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args, **kwargs):
        # Static args arrive in impl_static_args order: routing_map_format is the
        # last (8th) static arg. Be defensive about positional-vs-kwarg passing
        # across JAX versions.
        routing_map_format = kwargs.get("routing_map_format")
        if routing_map_format is None and len(args) >= 8:
            routing_map_format = args[7]
        # routing_map's expert axis is the same as logits in BYTEMAP mode; in
        # BITMAP_U8 mode it's a packed-byte axis distinct from num_experts.
        if routing_map_format is not None and int(routing_map_format) == int(
            RoutingMapFormat.BITMAP_U8
        ):
            return (
                "num_tokens num_experts, bias_dim ->"
                " num_tokens num_experts, num_tokens packed_experts, num_tokens num_experts"
            )
        return (
            "num_tokens num_experts, bias_dim -> num_tokens num_experts, num_tokens num_experts,"
            " num_tokens num_experts"
        )


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
    impl_static_args = (
        3,
        4,
        5,
        6,
        7,
        8,
    )  # topk, use_pre_softmax, scaling_factor, score_function, compute_aux_scores, routing_map_format
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
        routing_map_format,
    ):
        del topk, use_pre_softmax, scaling_factor, score_function
        del compute_aux_scores, routing_map_aval, routing_map_format
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
        routing_map_format,
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
            routing_map_format=routing_map_format,
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
        routing_map_format,
    ):
        if FusedTopkWithScoreFunctionBwdPrimitive.inner_primitive is None:
            raise RuntimeError(
                "FusedTopkWithScoreFunctionBwdPrimitive.inner_primitive has not been registered"
            )
        return FusedTopkWithScoreFunctionBwdPrimitive.inner_primitive.bind(
            routing_map,
            intermediate,
            grad_probs,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=scaling_factor,
            score_function=score_function,
            compute_aux_scores=compute_aux_scores,
            routing_map_format=routing_map_format,
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
        routing_map_format,
    ):
        if FusedTopkWithScoreFunctionBwdPrimitive.outer_primitive is None:
            raise RuntimeError(
                "FusedTopkWithScoreFunctionBwdPrimitive.outer_primitive has not been registered"
            )
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
                routing_map_format=routing_map_format,
            ),
            grad_probs_bdim,
        )

    @staticmethod
    def partition(
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        compute_aux_scores,
        routing_map_format,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, routing_map_format
        grad_spec = get_padded_spec(arg_infos[2])
        out_sharding = NamedSharding(mesh, PartitionSpec(*grad_spec))
        arg_shardings = (arg_infos[0].sharding, arg_infos[1].sharding, arg_infos[2].sharding)

        def sharded_impl(routing_map, intermediate, grad_probs):
            return FusedTopkWithScoreFunctionBwdPrimitive.impl(
                routing_map,
                intermediate,
                grad_probs,
                topk,
                use_pre_softmax,
                scaling_factor,
                score_function,
                compute_aux_scores,
                routing_map_format,
            )

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args, **kwargs):
        # routing_map_format is the 6th static arg (impl_static_args index 5).
        routing_map_format = kwargs.get("routing_map_format")
        if routing_map_format is None and len(args) >= 6:
            routing_map_format = args[5]
        if routing_map_format is not None and int(routing_map_format) == int(
            RoutingMapFormat.BITMAP_U8
        ):
            return (
                "num_tokens packed_experts, num_tokens num_experts, num_tokens num_experts ->"
                " num_tokens num_experts"
            )
        return (
            "num_tokens num_experts, num_tokens num_experts, num_tokens num_experts -> num_tokens"
            " num_experts"
        )


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
    impl_static_args = (2, 3)  # topk, coeff
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(probs_aval, tokens_per_expert_aval, topk, coeff):
        del topk, coeff, tokens_per_expert_aval
        i_dtype = dtypes.canonicalize_dtype(probs_aval.dtype)
        aux_loss_aval = probs_aval.update(shape=(), dtype=i_dtype)
        const_buf_aval = probs_aval.update(shape=(2,), dtype=jnp.float32)
        return aux_loss_aval, const_buf_aval

    @staticmethod
    def lowering(ctx, probs, tokens_per_expert, *, topk, coeff):
        return ffi.ffi_lowering(FusedMoEAuxLossFwdPrimitive.name)(
            ctx,
            probs,
            tokens_per_expert,
            topk=topk,
            coeff=coeff,
        )

    @staticmethod
    def impl(probs, tokens_per_expert, topk, coeff):
        if FusedMoEAuxLossFwdPrimitive.inner_primitive is None:
            raise RuntimeError(
                "FusedMoEAuxLossFwdPrimitive.inner_primitive has not been registered"
            )
        return FusedMoEAuxLossFwdPrimitive.inner_primitive.bind(
            probs,
            tokens_per_expert,
            topk=topk,
            coeff=coeff,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, topk, coeff):
        if FusedMoEAuxLossFwdPrimitive.outer_primitive is None:
            raise RuntimeError(
                "FusedMoEAuxLossFwdPrimitive.outer_primitive has not been registered"
            )
        probs, tokens_per_expert = batched_args
        probs_bdim, _ = batch_dims
        return (
            FusedMoEAuxLossFwdPrimitive.outer_primitive.bind(
                probs,
                tokens_per_expert,
                topk=topk,
                coeff=coeff,
            ),
            (probs_bdim, probs_bdim),
        )

    @staticmethod
    def partition(topk, coeff, mesh, arg_infos, result_infos):
        del result_infos
        aux_loss_sharding = NamedSharding(mesh, PartitionSpec())
        const_buf_sharding = NamedSharding(mesh, PartitionSpec(None))
        out_shardings = [aux_loss_sharding, const_buf_sharding]
        arg_shardings = (arg_infos[0].sharding, arg_infos[1].sharding)

        def sharded_impl(probs, tokens_per_expert):
            return FusedMoEAuxLossFwdPrimitive.impl(
                probs,
                tokens_per_expert,
                topk,
                coeff,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "num_tokens num_experts, num_experts -> , const_buf_one"


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
    impl_static_args = (3,)  # num_tokens
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(const_buf_aval, tokens_per_expert_aval, grad_aux_loss_aval, num_tokens):
        del const_buf_aval
        num_experts = tokens_per_expert_aval.shape[0]
        out_dtype = dtypes.canonicalize_dtype(grad_aux_loss_aval.dtype)
        return grad_aux_loss_aval.update(
            shape=(num_tokens, num_experts),
            dtype=out_dtype,
        )

    @staticmethod
    def lowering(ctx, const_buf, tokens_per_expert, grad_aux_loss, *, num_tokens):
        del num_tokens
        return ffi.ffi_lowering(FusedMoEAuxLossBwdPrimitive.name)(
            ctx,
            const_buf,
            tokens_per_expert,
            grad_aux_loss,
        )

    @staticmethod
    def impl(const_buf, tokens_per_expert, grad_aux_loss, num_tokens):
        if FusedMoEAuxLossBwdPrimitive.inner_primitive is None:
            raise RuntimeError(
                "FusedMoEAuxLossBwdPrimitive.inner_primitive has not been registered"
            )
        return FusedMoEAuxLossBwdPrimitive.inner_primitive.bind(
            const_buf,
            tokens_per_expert,
            grad_aux_loss,
            num_tokens=num_tokens,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, num_tokens):
        if FusedMoEAuxLossBwdPrimitive.outer_primitive is None:
            raise RuntimeError(
                "FusedMoEAuxLossBwdPrimitive.outer_primitive has not been registered"
            )
        const_buf, tokens_per_expert, grad_aux_loss = batched_args
        _, _, grad_bdim = batch_dims
        return (
            FusedMoEAuxLossBwdPrimitive.outer_primitive.bind(
                const_buf,
                tokens_per_expert,
                grad_aux_loss,
                num_tokens=num_tokens,
            ),
            grad_bdim,
        )

    @staticmethod
    def partition(
        num_tokens,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        out_sharding = NamedSharding(mesh, PartitionSpec(None, None))
        arg_shardings = (
            arg_infos[0].sharding,
            arg_infos[1].sharding,
            arg_infos[2].sharding,
        )

        def sharded_impl(const_buf, tokens_per_expert, grad_aux_loss):
            return FusedMoEAuxLossBwdPrimitive.impl(
                const_buf,
                tokens_per_expert,
                grad_aux_loss,
                num_tokens,
            )

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        # num_tokens only appears in the output (not in any input) because the
        # backward reconstructs the full [num_tokens, num_experts] grad_probs from
        # scalar inputs.  Shardy will leave num_tokens unsharded, which matches the
        # replicated PartitionSpec(None, None) in partition().
        return "const_buf_one, num_experts, grad_one -> i num_experts"


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
    routing_map_format: int = int(RoutingMapFormat.BYTEMAP),
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
    routing_map_format : int
        RoutingMapFormat.BYTEMAP (default, bool[T, E]) or RoutingMapFormat.BITMAP_U8
        (uint8[T, ceil(E/8)], LSB-first along the expert axis).

    Returns
    -------
    probs_or_scores, routing_map, saved_scores
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
        routing_map_format=int(routing_map_format),
    )


def fused_topk_with_score_function_bwd(
    routing_map: jnp.ndarray,
    saved_scores: jnp.ndarray,
    grad_probs: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    score_function,
    compute_aux_scores: bool = False,
    routing_map_format: int = int(RoutingMapFormat.BYTEMAP),
):
    """
    Fused top-k with score function backward pass.

    When compute_aux_scores=True, routing_map is ignored and the
    score-for-aux-loss backward kernel is used instead.

    routing_map_format must match the layout produced by the matching forward
    call (BYTEMAP or BITMAP_U8). The CUDA kernel branches per-lane on this flag
    when loading bits into shmem.
    """
    return FusedTopkWithScoreFunctionBwdPrimitive.outer_primitive.bind(
        routing_map,
        saved_scores,
        grad_probs,
        topk=int(topk),
        use_pre_softmax=int(use_pre_softmax),
        scaling_factor=float(scaling_factor),
        score_function=int(score_function),
        compute_aux_scores=int(compute_aux_scores),
        routing_map_format=int(routing_map_format),
    )


def fused_moe_aux_loss_fwd(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
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
        topk=int(topk),
        coeff=float(coeff),
    )


def fused_moe_aux_loss_bwd(
    const_buf: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    grad_aux_loss: jnp.ndarray,
    num_tokens: int,
):
    """
    Fused MoE aux loss backward pass.
    """
    return FusedMoEAuxLossBwdPrimitive.outer_primitive.bind(
        const_buf,
        tokens_per_expert,
        grad_aux_loss,
        num_tokens=int(num_tokens),
    )
