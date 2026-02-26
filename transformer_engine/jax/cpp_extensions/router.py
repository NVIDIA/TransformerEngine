# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for fused MoE router"""
import warnings
from functools import partial

import jax.numpy as jnp
from jax import dtypes, ffi
from jax.sharding import PartitionSpec, NamedSharding

from .base import BasePrimitive, register_primitive
from .misc import get_padded_spec

__all__ = [
    "fused_topk_with_score_function_fwd",
    "fused_topk_with_score_function_bwd",
    "fused_score_for_moe_aux_loss_fwd",
    "fused_score_for_moe_aux_loss_bwd",
    "fused_moe_aux_loss_fwd",
    "fused_moe_aux_loss_bwd",
]

SCORE_FUNCTION_MAP = {"sigmoid": 0, "softmax": 1}


# =========================================== ==================================
# Fused Top-K with Score Function - Forward
# =============================================================================

class FusedTopkWithScoreFunctionFwdPrimitive(BasePrimitive):
    """
    Fused Top-K with Score Function Forward Primitive.
    Computes score_function(logits) -> top-k -> probs, routing_map.
    """

    name = "te_fused_topk_with_score_function_forward_ffi"
    multiple_results = True
    impl_static_args = (2, 3, 4, 5, 6, 7)  # topk, use_pre_softmax, num_groups, group_topk, scaling_factor, score_function
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
    ):
        """Abstract evaluation: describe output shapes and dtypes."""
        del expert_bias_aval, topk, use_pre_softmax, num_groups, group_topk, scaling_factor, score_function
        i_dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        i_shape = logits_aval.shape
        assert len(i_shape) == 2, f"logits must be 2D [num_tokens, num_experts], got {i_shape}"
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
            ),
            (logits_bdim, logits_bdim, logits_bdim),
        )

    @staticmethod
    def infer_sharding_from_operands(
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        mesh,
        arg_infos,
        result_infos,
    ):
        del (
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            result_infos,
        )
        logits_spec = get_padded_spec(arg_infos[0])
        if logits_spec[-1] is not None:
            warnings.warn(
                f"Sharding the expert dimension is not supported in "
                f"{FusedTopkWithScoreFunctionFwdPrimitive.name}! "
                "Forcing XLA to not shard the expert dim, which might introduce extra "
                "collective ops and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec[:-1], None))
        return [out_sharding, out_sharding, out_sharding]

    @staticmethod
    def partition(
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        logits_spec = get_padded_spec(arg_infos[0])
        if logits_spec[-1] is not None:
            warnings.warn(
                f"Sharding the expert dimension is not supported in "
                f"{FusedTopkWithScoreFunctionFwdPrimitive.name}! "
                "Forcing XLA to not shard the expert dim, which might introduce extra "
                "collective ops and hurt performance."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec[:-1], None))
        logits_sharding = out_sharding
        bias_sharding = NamedSharding(mesh, PartitionSpec(None))
        arg_shardings = (logits_sharding, bias_sharding)
        impl = partial(
            FusedTopkWithScoreFunctionFwdPrimitive.impl,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
        )
        return mesh, impl, [out_sharding, out_sharding, out_sharding], arg_shardings

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
    """

    name = "te_fused_topk_with_score_function_backward_ffi"
    multiple_results = False
    impl_static_args = (3, 4, 5, 6)  # topk, use_pre_softmax, scaling_factor, score_function
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
    ):
        del topk, use_pre_softmax, scaling_factor, score_function, routing_map_aval
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
            ),
            grad_probs_bdim,
        )

    @staticmethod
    def infer_sharding_from_operands(
        topk, use_pre_softmax, scaling_factor, score_function, mesh, arg_infos, result_infos
    ):
        del topk, use_pre_softmax, scaling_factor, score_function, result_infos
        grad_spec = get_padded_spec(arg_infos[2])
        if grad_spec[-1] is not None:
            warnings.warn(
                f"Sharding the expert dimension is not supported in "
                f"{FusedTopkWithScoreFunctionBwdPrimitive.name}! "
                "Forcing XLA to not shard the expert dim."
            )
        return NamedSharding(mesh, PartitionSpec(*grad_spec[:-1], None))

    @staticmethod
    def partition(
        topk, use_pre_softmax, scaling_factor, score_function, mesh, arg_infos, result_infos
    ):
        del result_infos
        grad_spec = get_padded_spec(arg_infos[2])
        if grad_spec[-1] is not None:
            warnings.warn(
                f"Sharding the expert dimension is not supported in "
                f"{FusedTopkWithScoreFunctionBwdPrimitive.name}! "
                "Forcing XLA to not shard the expert dim."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*grad_spec[:-1], None))
        arg_shardings = tuple(
            NamedSharding(mesh, PartitionSpec(*get_padded_spec(a)[:-1], None))
            for a in arg_infos
        )
        impl = partial(
            FusedTopkWithScoreFunctionBwdPrimitive.impl,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=scaling_factor,
            score_function=score_function,
        )
        return mesh, impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "num_tokens num_experts, num_tokens num_experts, num_tokens num_experts -> num_tokens num_experts"


register_primitive(FusedTopkWithScoreFunctionBwdPrimitive)


# =============================================================================
# Fused Score for MoE Aux Loss - Forward
# =============================================================================


class FusedScoreForMoEAuxLossFwdPrimitive(BasePrimitive):
    """
    Fused Score for MoE Aux Loss Forward Primitive.
    """

    name = "te_fused_score_for_moe_aux_loss_forward_ffi"
    multiple_results = True
    impl_static_args = (1, 2)  # topk, score_function
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(logits_aval, topk, score_function):
        del topk, score_function
        i_dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        i_shape = logits_aval.shape
        scores_aval = logits_aval.update(shape=i_shape, dtype=i_dtype)
        routing_map_aval = logits_aval.update(shape=i_shape, dtype=jnp.bool_)
        intermediate_aval = logits_aval.update(shape=i_shape, dtype=i_dtype)
        return scores_aval, routing_map_aval, intermediate_aval

    @staticmethod
    def lowering(ctx, logits, *, topk, score_function):
        return ffi.ffi_lowering(FusedScoreForMoEAuxLossFwdPrimitive.name)(
            ctx, logits, topk=topk, score_function=score_function
        )

    @staticmethod
    def impl(logits, topk, score_function):
        assert FusedScoreForMoEAuxLossFwdPrimitive.inner_primitive is not None
        return FusedScoreForMoEAuxLossFwdPrimitive.inner_primitive.bind(
            logits, topk=topk, score_function=score_function
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, topk, score_function):
        assert FusedScoreForMoEAuxLossFwdPrimitive.outer_primitive is not None
        (logits,) = batched_args
        (logits_bdim,) = batch_dims
        return (
            FusedScoreForMoEAuxLossFwdPrimitive.outer_primitive.bind(
                logits, topk=topk, score_function=score_function
            ),
            (logits_bdim, logits_bdim, logits_bdim),
        )

    @staticmethod
    def infer_sharding_from_operands(topk, score_function, mesh, arg_infos, result_infos):
        del topk, score_function, result_infos
        logits_spec = get_padded_spec(arg_infos[0])
        if logits_spec[-1] is not None:
            warnings.warn(
                f"Sharding the expert dimension is not supported in "
                f"{FusedScoreForMoEAuxLossFwdPrimitive.name}! "
                "Forcing XLA to not shard the expert dim."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec[:-1], None))
        return [out_sharding, out_sharding, out_sharding]

    @staticmethod
    def partition(topk, score_function, mesh, arg_infos, result_infos):
        del result_infos
        logits_spec = get_padded_spec(arg_infos[0])
        if logits_spec[-1] is not None:
            warnings.warn(
                f"Sharding the expert dimension is not supported in "
                f"{FusedScoreForMoEAuxLossFwdPrimitive.name}! "
                "Forcing XLA to not shard the expert dim."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*logits_spec[:-1], None))
        arg_shardings = (out_sharding,)
        impl = partial(
            FusedScoreForMoEAuxLossFwdPrimitive.impl, topk=topk, score_function=score_function
        )
        return mesh, impl, [out_sharding, out_sharding, out_sharding], arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "num_tokens num_experts -> num_tokens num_experts, num_tokens num_experts, num_tokens num_experts"


register_primitive(FusedScoreForMoEAuxLossFwdPrimitive)


# =============================================================================
# Fused Score for MoE Aux Loss - Backward
# =============================================================================


class FusedScoreForMoEAuxLossBwdPrimitive(BasePrimitive):
    """
    Fused Score for MoE Aux Loss Backward Primitive.
    """

    name = "te_fused_score_for_moe_aux_loss_backward_ffi"
    multiple_results = False
    impl_static_args = (2, 3)  # topk, score_function
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(intermediate_aval, grad_scores_aval, topk, score_function):
        del topk, score_function, intermediate_aval
        return grad_scores_aval.update(
            shape=grad_scores_aval.shape,
            dtype=dtypes.canonicalize_dtype(grad_scores_aval.dtype),
        )

    @staticmethod
    def lowering(ctx, intermediate, grad_scores, *, topk, score_function):
        return ffi.ffi_lowering(FusedScoreForMoEAuxLossBwdPrimitive.name)(
            ctx, intermediate, grad_scores, topk=topk, score_function=score_function
        )

    @staticmethod
    def impl(intermediate, grad_scores, topk, score_function):
        assert FusedScoreForMoEAuxLossBwdPrimitive.inner_primitive is not None
        return FusedScoreForMoEAuxLossBwdPrimitive.inner_primitive.bind(
            intermediate, grad_scores, topk=topk, score_function=score_function
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, topk, score_function):
        assert FusedScoreForMoEAuxLossBwdPrimitive.outer_primitive is not None
        intermediate, grad_scores = batched_args
        _, grad_scores_bdim = batch_dims
        return (
            FusedScoreForMoEAuxLossBwdPrimitive.outer_primitive.bind(
                intermediate, grad_scores, topk=topk, score_function=score_function
            ),
            grad_scores_bdim,
        )

    @staticmethod
    def infer_sharding_from_operands(topk, score_function, mesh, arg_infos, result_infos):
        del topk, score_function, result_infos
        spec = get_padded_spec(arg_infos[1])
        if spec[-1] is not None:
            warnings.warn(
                f"Sharding the expert dimension is not supported in "
                f"{FusedScoreForMoEAuxLossBwdPrimitive.name}! "
                "Forcing XLA to not shard the expert dim."
            )
        return NamedSharding(mesh, PartitionSpec(*spec[:-1], None))

    @staticmethod
    def partition(topk, score_function, mesh, arg_infos, result_infos):
        del result_infos
        spec = get_padded_spec(arg_infos[1])
        if spec[-1] is not None:
            warnings.warn(
                f"Sharding the expert dimension is not supported in "
                f"{FusedScoreForMoEAuxLossBwdPrimitive.name}! "
                "Forcing XLA to not shard the expert dim."
            )
        out_sharding = NamedSharding(mesh, PartitionSpec(*spec[:-1], None))
        arg_shardings = tuple(
            NamedSharding(mesh, PartitionSpec(*get_padded_spec(a)[:-1], None))
            for a in arg_infos
        )
        impl = partial(
            FusedScoreForMoEAuxLossBwdPrimitive.impl, topk=topk, score_function=score_function
        )
        return mesh, impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "num_tokens num_experts, num_tokens num_experts -> num_tokens num_experts"


register_primitive(FusedScoreForMoEAuxLossBwdPrimitive)


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
    def infer_sharding_from_operands(
        total_num_tokens, num_experts, topk, coeff, mesh, arg_infos, result_infos
    ):
        del total_num_tokens, num_experts, topk, coeff, arg_infos, result_infos
        replicated = NamedSharding(mesh, PartitionSpec())
        return [replicated, replicated]

    @staticmethod
    def partition(
        total_num_tokens, num_experts, topk, coeff, mesh, arg_infos, result_infos
    ):
        del result_infos, arg_infos
        replicated = NamedSharding(mesh, PartitionSpec())
        arg_shardings = (replicated, replicated)
        impl = partial(
            FusedMoEAuxLossFwdPrimitive.impl,
            total_num_tokens=total_num_tokens,
            num_experts=num_experts,
            topk=topk,
            coeff=coeff,
        )
        return mesh, impl, [replicated, replicated], arg_shardings

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
    def infer_sharding_from_operands(num_rows, num_cols, mesh, arg_infos, result_infos):
        del num_rows, num_cols, result_infos, arg_infos
        # Output is [num_rows, num_cols]; cannot infer token sharding from
        # scalar/1D inputs, so replicate by default.
        return NamedSharding(mesh, PartitionSpec(None, None))

    @staticmethod
    def partition(num_rows, num_cols, mesh, arg_infos, result_infos):
        del result_infos, arg_infos
        # All inputs are scalars or 1D vectors — replicate them.
        # Output is [num_rows, num_cols] — replicate (no token sharding info
        # available from scalar inputs).
        replicated = NamedSharding(mesh, PartitionSpec())
        out_sharding = NamedSharding(mesh, PartitionSpec(None, None))
        arg_shardings = (replicated, replicated, replicated)
        impl = partial(FusedMoEAuxLossBwdPrimitive.impl, num_rows=num_rows, num_cols=num_cols)
        return mesh, impl, out_sharding, arg_shardings

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
    score_function: str,
    expert_bias: jnp.ndarray,
):
    """
    Fused top-k with score function forward pass.

    Parameters
    ----------
    logits : jnp.ndarray
        [num_tokens, num_experts] logits from gating GEMM.
    topk : int
        Number of top experts to select.
    use_pre_softmax : bool
        If True, apply softmax before top-k.
    num_groups : int
        Number of groups for grouped top-k (-1 to disable).
    group_topk : int
        Top-k at group level (-1 to disable).
    scaling_factor : float
        Scaling factor for output probs.
    score_function : str
        "softmax" or "sigmoid".
    expert_bias : jnp.ndarray
        Expert bias (only used with sigmoid). Pass empty array if unused.

    Returns
    -------
    probs, routing_map, intermediate_output
    """
    score_fn_int = SCORE_FUNCTION_MAP[score_function]
    return FusedTopkWithScoreFunctionFwdPrimitive.outer_primitive.bind(
        logits,
        expert_bias,
        topk=int(topk),
        use_pre_softmax=int(use_pre_softmax),
        num_groups=int(num_groups),
        group_topk=int(group_topk),
        scaling_factor=float(scaling_factor),
        score_function=int(score_fn_int),
    )


def fused_topk_with_score_function_bwd(
    routing_map: jnp.ndarray,
    intermediate_output: jnp.ndarray,
    grad_probs: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    score_function: str,
):
    """
    Fused top-k with score function backward pass.
    """
    score_fn_int = SCORE_FUNCTION_MAP[score_function]
    return FusedTopkWithScoreFunctionBwdPrimitive.outer_primitive.bind(
        routing_map,
        intermediate_output,
        grad_probs,
        topk=int(topk),
        use_pre_softmax=int(use_pre_softmax),
        scaling_factor=float(scaling_factor),
        score_function=int(score_fn_int),
    )


def fused_score_for_moe_aux_loss_fwd(
    logits: jnp.ndarray,
    topk: int,
    score_function: str,
):
    """
    Fused compute scores for MoE aux loss forward pass.

    Returns
    -------
    scores, routing_map, intermediate_output
    """
    score_fn_int = SCORE_FUNCTION_MAP[score_function]
    return FusedScoreForMoEAuxLossFwdPrimitive.outer_primitive.bind(
        logits,
        topk=int(topk),
        score_function=int(score_fn_int),
    )


def fused_score_for_moe_aux_loss_bwd(
    intermediate_output: jnp.ndarray,
    grad_scores: jnp.ndarray,
    topk: int,
    score_function: str,
):
    """
    Fused compute scores for MoE aux loss backward pass.
    """
    score_fn_int = SCORE_FUNCTION_MAP[score_function]
    return FusedScoreForMoEAuxLossBwdPrimitive.outer_primitive.bind(
        intermediate_output,
        grad_scores,
        topk=int(topk),
        score_function=int(score_fn_int),
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
