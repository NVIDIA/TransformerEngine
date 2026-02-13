# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for fused router"""
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import dtypes, ffi
from jax.interpreters.mlir import ir
from jax.experimental.custom_partitioning import SdyShardingRule

from .base import BasePrimitive, register_primitive


__all__ = [
    "fused_topk_with_score_function_fwd",
    "fused_topk_with_score_function_bwd",
    "map_score_function",
]


def map_score_function(score_function: str) -> int:
    score_function_map = {"sigmoid": 0, "softmax": 1}
    assert (
        score_function in score_function_map
    ), f"score_function must be 'sigmoid' or 'softmax', got {score_function}"
    return score_function_map[score_function]


class FusedTopkWithScoreFunctionFwdPrimitive(BasePrimitive):
    """
    Fused TopK with Score Function Forward Primitive
    """

    name = "te_fused_topk_with_score_function_forward_ffi"
    multiple_results = True  # Returns (probs, routing_map, intermediate_output)
    impl_static_args = (
        2,
        3,
        4,
        5,
        6,
        7,
    )  # topk, use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        logits_aval,
        expert_bias_aval,
        *,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
    ):
        """
        te_fused_topk_with_score_function_forward abstract
        """
        dtype = dtypes.canonicalize_dtype(logits_aval.dtype)
        assert len(logits_aval.shape) == 3  # (batch, seqlen, num_experts)

        probs_aval = logits_aval.update(shape=logits_aval.shape, dtype=dtype)
        routing_map_aval = logits_aval.update(shape=logits_aval.shape, dtype=jnp.bool_)
        intermediate_output_aval = logits_aval.update(shape=logits_aval.shape, dtype=dtype)

        return probs_aval, routing_map_aval, intermediate_output_aval

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
        """
        te_fused_topk_with_score_function_forward lowering rules
        """
        logits_type = ir.RankedTensorType(logits.type)
        logits_shape = logits_type.shape
        assert len(logits_shape) == 3  # (batch, seqlen, num_experts)
        (batch, seqlen, num_experts) = logits_shape

        return ffi.ffi_lowering(FusedTopkWithScoreFunctionFwdPrimitive.name)(
            ctx,
            logits,
            expert_bias,
            num_tokens=batch * seqlen,
            num_experts=num_experts,
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
        (probs, routing_map, intermediate_output) = (
            FusedTopkWithScoreFunctionFwdPrimitive.inner_primitive.bind(
                logits,
                expert_bias,
                topk=topk,
                use_pre_softmax=use_pre_softmax,
                num_groups=num_groups,
                group_topk=group_topk,
                scaling_factor=scaling_factor,
                score_function=score_function,
            )
        )
        return probs, routing_map, intermediate_output

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
    ):
        raise NotImplementedError(
            "Batcher not implemented for FusedTopkWithScoreFunctionFwdPrimitive"
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
        return (arg_infos[0].sharding, arg_infos[0].sharding, arg_infos[0].sharding)

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
        out_shardings = (arg_infos[0].sharding, arg_infos[0].sharding, arg_infos[0].sharding)
        arg_shardings = (arg_infos[0].sharding, arg_infos[1].sharding)
        impl = partial(
            FusedTopkWithScoreFunctionFwdPrimitive.impl,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
        )
        return mesh, impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        mesh,
        operand_types,
        result_types,
    ):
        del (
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            mesh,
            result_types,
        )

        prefix = "Router_"
        logits_spec = (prefix + "batch", prefix + "seqlen", prefix + "experts")
        expert_bias_spec = (prefix + "experts",)

        output_spec = (prefix + "batch", prefix + "seqlen", prefix + "experts")

        return SdyShardingRule(
            (logits_spec, expert_bias_spec),
            (output_spec, output_spec, output_spec),
        )


register_primitive(FusedTopkWithScoreFunctionFwdPrimitive)


class FusedTopkWithScoreFunctionBwdPrimitive(BasePrimitive):

    name = "te_fused_topk_with_score_function_backward_ffi"
    multiple_results = False
    impl_static_args = (3, 4, 5, 6)  # topk, use_pre_softmax, scaling_factor, score_function,
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        routing_map_aval,
        intermediate_output_aval,
        grad_probs_aval,
        *,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
    ):
        """
        te_fused_topk_with_score_function_backward abstract
        """
        dtype = dtypes.canonicalize_dtype(grad_probs_aval.dtype)

        grad_logits_aval = grad_probs_aval.update(shape=intermediate_output_aval.shape, dtype=dtype)
        return grad_logits_aval

    @staticmethod
    def lowering(
        ctx,
        routing_map,
        intermediate_output,
        grad_probs,
        *,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
    ):
        """
        te_fused_topk_with_score_function_backward lowering rules
        """
        intermediate_output_type = ir.RankedTensorType(intermediate_output.type)
        intermediate_output_shape = intermediate_output_type.shape
        assert len(intermediate_output_shape) == 3  # (batch, seqlen, num_experts)
        (batch, seqlen, num_experts) = intermediate_output_shape

        return ffi.ffi_lowering(FusedTopkWithScoreFunctionBwdPrimitive.name)(
            ctx,
            routing_map,
            intermediate_output,
            grad_probs,
            num_tokens=batch * seqlen,
            num_experts=num_experts,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=scaling_factor,
            score_function=score_function,
        )

    @staticmethod
    def impl(
        routing_map,
        intermediate_output,
        grad_probs,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
    ):
        assert FusedTopkWithScoreFunctionBwdPrimitive.inner_primitive is not None
        return FusedTopkWithScoreFunctionBwdPrimitive.inner_primitive.bind(
            routing_map,
            intermediate_output,
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
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
    ):
        raise NotImplementedError(
            "Batcher not implemented for FusedTopkWithScoreFunctionBwdPrimitive"
        )

    @staticmethod
    def infer_sharding_from_operands(
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        mesh,
        arg_infos,
        result_infos,
    ):
        del (
            topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
            result_infos,
        )
        # grad_logits sharding follows grad_probs sharding
        return arg_infos[2].sharding

    @staticmethod
    def partition(
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        grad_probs_sharding = arg_infos[2].sharding
        out_shardings = grad_probs_sharding
        arg_shardings = (arg_infos[0].sharding, arg_infos[1].sharding, grad_probs_sharding)
        impl = partial(
            FusedTopkWithScoreFunctionBwdPrimitive.impl,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=scaling_factor,
            score_function=score_function,
        )
        return mesh, impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
        mesh,
        operand_types,
        result_types,
    ):
        del (
            topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
            mesh,
            result_types,
        )

        prefix = "RouterBwd_"

        input_spec = (prefix + "batch", prefix + "seqlen", prefix + "experts")

        output_spec = (prefix + "batch", prefix + "seqlen", prefix + "experts")

        return SdyShardingRule(
            (input_spec, input_spec, input_spec),
            (output_spec,),
        )


register_primitive(FusedTopkWithScoreFunctionBwdPrimitive)


def fused_topk_with_score_function_fwd(
    logits: jnp.ndarray,
    expert_bias: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    scaling_factor: float,
    score_function: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if expert_bias is None:
        expert_bias = jnp.zeros((0,), dtype=logits.dtype)

    return FusedTopkWithScoreFunctionFwdPrimitive.outer_primitive.bind(
        logits,
        expert_bias,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
    )


def fused_topk_with_score_function_bwd(
    routing_map: jnp.ndarray,
    intermediate_output: jnp.ndarray,
    grad_probs: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    score_function: int,
) -> jnp.ndarray:

    return FusedTopkWithScoreFunctionBwdPrimitive.outer_primitive.bind(
        routing_map,
        intermediate_output,
        grad_probs,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        scaling_factor=scaling_factor,
        score_function=score_function,
    )
