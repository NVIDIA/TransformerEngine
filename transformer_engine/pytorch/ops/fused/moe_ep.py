# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Reference-backed BF16 expert-parallel MoE fusion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional

import torch

from ...ep import get_ep_group
from ...ep_reference import MoeEpReference
from ...quantization import Recipe
from ...tensor import Quantizer
from ..basic import Combine, Dispatch, GroupedLinear, ScaledSwiGLU
from ..fuser import register_forward_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext


def _weight_list(op: GroupedLinear) -> list[torch.Tensor]:
    """Return per-expert weights in their registered order."""
    return [getattr(op, f"weight{idx}") for idx in range(op.num_groups)]


def _reference_weights(op: GroupedLinear) -> torch.Tensor:
    """Pack ``(out, in)`` expert weights into reference ``(E, in, out)`` layout."""
    return torch.stack([weight.transpose(0, 1) for weight in _weight_list(op)])


def _grouped_linear_supported(op: GroupedLinear) -> bool:
    weights = _weight_list(op) if not op.single_grouped_weight else []
    return (
        not op.use_bias
        and not op._scale_bias
        and not op.single_grouped_weight
        and not op.single_grouped_bias
        and not op._accumulate_into_main_grad
        and not op._is_distributed_weight()
        and not op.wgrad_store.delay_wgrad_compute()
        and bool(weights)
        and all(weight.dtype is torch.bfloat16 for weight in weights)
    )


def _routing_extras_internal(
    dispatch: Dispatch,
    fc1: GroupedLinear,
    activation: ScaledSwiGLU,
    fc2: GroupedLinear,
) -> bool:
    """Whether the dispatch routing extras stay inside the fusion.

    The fused op keeps tokens-per-expert and the received routing weights
    internal to :class:`MoeEpReference`, so it can only replace the sequence
    when those two outputs feed exactly these ops and are not returned to the
    caller.
    """
    tokens_per_expert, routing_weights = dispatch._extra_output_channels
    if tokens_per_expert is None or routing_weights is None:
        return False
    if any(dispatch._extra_output_to_caller):
        return False
    return (
        fc1._extra_input_channels[0] == tokens_per_expert
        and fc2._extra_input_channels[0] == tokens_per_expert
        and activation._extra_input_channels[0] == routing_weights
    )


def _routing_extras_internal(
    dispatch: Dispatch,
    fc1: GroupedLinear,
    activation: ScaledSwiGLU,
    fc2: GroupedLinear,
) -> bool:
    """Whether the dispatch routing extras stay inside the fusion.

    The fused op keeps tokens-per-expert and the received routing weights
    internal to :class:`MoeEpReference`, so it can only replace the sequence
    when those two outputs feed exactly these ops and are not returned to the
    caller.
    """
    tokens_per_expert, routing_weights = dispatch._extra_output_channels
    if tokens_per_expert is None or routing_weights is None:
        return False
    if any(dispatch._extra_output_to_caller):
        return False
    return (
        fc1._extra_input_channels[0] == tokens_per_expert
        and fc2._extra_input_channels[0] == tokens_per_expert
        and activation._extra_input_channels[0] == routing_weights
    )


def _matches(window: Sequence[FusibleOperation], recipe: Optional[Recipe]) -> bool:
    if recipe is not None or len(window) != 5:
        return False
    dispatch, fc1, activation, fc2, combine = window
    if not (
        isinstance(dispatch, Dispatch)
        and isinstance(fc1, GroupedLinear)
        and isinstance(activation, ScaledSwiGLU)
        and isinstance(fc2, GroupedLinear)
        and isinstance(combine, Combine)
    ):
        return False
    if dispatch.buffer is not combine.buffer:
        return False
    buffer = dispatch.buffer
    if not buffer.eager or buffer.payload_dtype is not torch.bfloat16:
        return False
    if not (_grouped_linear_supported(fc1) and _grouped_linear_supported(fc2)):
        return False
    if activation.activation_recompute_in_mlp or activation.glu_interleave_size is not None:
        return False
    if not _routing_extras_internal(dispatch, fc1, activation, fc2):
        return False
    return (
        fc1.num_groups == buffer.num_local_experts
        and fc2.num_groups == buffer.num_local_experts
        and fc1.in_features == buffer.hidden_dim
        and fc2.out_features == buffer.hidden_dim
        and fc1.out_features == 2 * fc2.in_features
    )


class FusedMoeEp(FusedOperation):
    """Joint BF16 fusion implemented with :class:`MoeEpReference`."""

    def __init__(
        self,
        *,
        dispatch: Dispatch,
        fc1: GroupedLinear,
        activation: ScaledSwiGLU,
        fc2: GroupedLinear,
        combine: Combine,
    ) -> None:
        super().__init__([dispatch, fc1, activation, fc2, combine])
        ep_group = get_ep_group()
        ep_size = 1 if ep_group is None else ep_group.size()
        self._reference = MoeEpReference(
            num_experts=dispatch.buffer.num_local_experts * ep_size,
            hidden_size=dispatch.buffer.hidden_dim,
            intermediate_size=fc2.in_features,
            top_k=dispatch.buffer.top_k,
            ep_group=ep_group,
            max_tokens_per_rank=dispatch.buffer.max_tokens_per_rank,
            compute_dtype=torch.bfloat16,
            apply_topk_in_fc1=True,
            generate_c=True,
        )

    @property
    def dispatch(self) -> Dispatch:
        return self.basic_ops[0]

    @property
    def fc1(self) -> GroupedLinear:
        return self.basic_ops[1]

    @property
    def fc2(self) -> GroupedLinear:
        return self.basic_ops[3]

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Sequence[Sequence[Optional[torch.Tensor]]]]:
        del prev_op_grad_output_quantizer, next_op_input_quantizer
        if input_.dtype is not torch.bfloat16:
            raise NotImplementedError(f"FusedMoeEp requires BF16 input, got {input_.dtype}.")
        if any(kwargs for kwargs in basic_op_kwargs):
            raise NotImplementedError("FusedMoeEp does not support per-operation output buffers.")

        topk_idx, topk_weights = basic_op_extra_inputs[0]
        if topk_weights.dtype is not torch.float32:
            raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}.")
        fc1_weight = _reference_weights(self.fc1)
        fc2_weight = _reference_weights(self.fc2)
        output, fc1_c, route_metadata = self._reference(
            input_,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights,
        )

        if any(ctx.requires_grad for ctx in basic_op_ctxs):
            basic_op_ctxs[0].save_for_backward(
                input_,
                fc1_weight,
                fc2_weight,
                topk_idx,
                topk_weights,
                fc1_c,
                route_metadata,
            )

        # Dispatch extras are channel-bound with output_to_caller=False and are
        # only consumed by ops inside this fusion, so they need not be materialized.
        return output, [
            (None, None),
            (),
            (),
            (),
            (),
        ]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[Optional[torch.Tensor], ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        del basic_op_grad_extra_outputs
        (
            input_,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights,
            fc1_c,
            route_metadata,
        ) = basic_op_ctxs[0].saved_tensors
        grad_input, grad_fc1, grad_fc2, grad_topk_weights = self._reference.backward(
            grad_output,
            input_,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights,
            fc1_c,
            route_metadata,
        )

        fc1_param_grads = [
            grad_fc1[idx].transpose(0, 1) if weight.requires_grad else None
            for idx, weight in enumerate(_weight_list(self.fc1))
        ]
        fc2_param_grads = [
            grad_fc2[idx].transpose(0, 1) if weight.requires_grad else None
            for idx, weight in enumerate(_weight_list(self.fc2))
        ]
        return (
            grad_input,
            [(), fc1_param_grads, (), fc2_param_grads, ()],
            [(None, grad_topk_weights), (None,), (None,), (None,), ()],
        )


def fuse_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused: Any,
) -> list[FusibleOperation]:
    """Fuse supported five-op BF16 EP MoE sequences."""
    del unused
    out: list[FusibleOperation] = []
    idx = 0
    while idx < len(ops):
        window = ops[idx : idx + 5]
        if _matches(window, recipe):
            dispatch, fc1, activation, fc2, combine = window
            out.append(
                FusedMoeEp(
                    dispatch=dispatch,
                    fc1=fc1,
                    activation=activation,
                    fc2=fc2,
                    combine=combine,
                )
            )
            idx += 5
        else:
            out.append(ops[idx])
            idx += 1
    return out


register_forward_backward_fusion(fuse_ops, prepend=True)


__all__ = ["FusedMoeEp"]
