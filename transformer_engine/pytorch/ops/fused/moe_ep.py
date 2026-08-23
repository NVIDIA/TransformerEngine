# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MegaMoE-backed expert-parallel MoE fusion (cudnn.moe_ep.MoeEp)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional

import torch
import transformer_engine_torch as tex

from ...constants import MXFP8_BLOCK_SCALING_SIZE
from ...ep import get_ep_group
from ...quantization import Recipe
from ...tensor import GroupedTensor, MXFP8Quantizer, Quantizer
from .._common import (
    is_quantized_tensor,
    maybe_dequantize,
    quantize_mxfp8_for_ep,
)
from ..basic import Combine, Dispatch, GroupedLinear, ScaledSwiGLU
from ..fuser import register_forward_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext


def _pack_as_cudnn_moe_tensor(
    data: torch.Tensor,
    scale: Optional[torch.Tensor],
    block_scaled_cls: type,
):
    """Represent payload and scales using the public cuDNN MoE tensor type."""
    if scale is None:
        return data
    return block_scaled_cls(
        data=data,
        scale=scale,
        format="mxfp8",
        logical_shape=tuple(data.shape),
        axis=1,
    )


def _pack_cudnn_activation(
    input_: torch.Tensor,
    quantizer: Optional[MXFP8Quantizer],
    block_scaled_cls: type,
):
    """Pack the dispatch input in the public cuDNN MoE activation layout."""
    if quantizer is None:
        return input_
    quantized, scale = quantize_mxfp8_for_ep(input_, quantizer)
    return _pack_as_cudnn_moe_tensor(
        quantized._rowwise_data.view(torch.float8_e4m3fn),
        scale.view(torch.float8_e8m0fnu),
        block_scaled_cls,
    )


def _pack_cudnn_weights(
    op: GroupedLinear,
    *,
    block_scaled_cls: Optional[type] = None,
):
    """Pack a GroupedLinear ``(E, out, in)`` weight as cuDNN ``(E, in, out)``.

    The permutation is intentionally not made contiguous. MegaMoE internally
    permutes back to ``(E, out, in)`` before requesting contiguous storage, so
    retaining this view lets quantized-model-init weights reuse their original
    packed buffer. Dense parameters are quantized with GroupedLinear's weight
    quantizer, matching its normal MXFP8 forward path.
    """
    weight = op.weight
    num_groups = op.num_groups
    out_features = op.out_features
    in_features = op.in_features
    weight_quantizer = op.get_quantizer("forward", 1)
    if weight_quantizer is None and weight.quantizer is None:
        return weight.rowwise_data.view(
            num_groups,
            out_features,
            in_features,
        ).permute(0, 2, 1)

    if weight_quantizer is not None and weight.quantizer is None:
        weight_quantizer.set_usage(rowwise=True)
        weight = tex.group_quantize(
            weight.rowwise_data.view(weight.logical_shape),
            weight_quantizer,
            op.num_groups,
            None,
        )
    scale_cols = in_features // MXFP8_BLOCK_SCALING_SIZE
    data = (
        weight.rowwise_data.view(num_groups, out_features, in_features)
        .view(torch.float8_e4m3fn)
        .permute(0, 2, 1)
    )
    scale = (
        weight.scale_inv.view(num_groups, out_features, scale_cols)
        .view(torch.float8_e8m0fnu)
        .permute(0, 2, 1)
    )
    if block_scaled_cls is None:
        from cudnn.moe_ep import BlockScaledTensor as block_scaled_cls
    return _pack_as_cudnn_moe_tensor(
        data,
        scale,
        block_scaled_cls,
    )


def _grouped_weight_grad(op: GroupedLinear, grad: torch.Tensor) -> list[Optional[torch.Tensor]]:
    """Convert a MegaMoE ``(E, in, out)`` wgrad to one TE ``(E, out, in)`` grad."""
    weight = op.weight
    expected_shape = (op.num_groups, op.in_features, op.out_features)
    if tuple(grad.shape) != expected_shape:
        raise RuntimeError(
            f"MegaMoE weight gradient must have shape {expected_shape}, got {tuple(grad.shape)}"
        )
    if not weight.requires_grad:
        return [None]

    # copy_ performs the float32-to-parameter-dtype conversion while writing
    # directly into the contiguous layout expected by the grouped parameter.
    param_grad = torch.empty(
        (op.num_groups, op.out_features, op.in_features),
        dtype=weight.dtype,
        device=grad.device,
    )
    param_grad.copy_(grad.transpose(1, 2))
    return [param_grad]


def _grouped_linear_supported(op: GroupedLinear) -> bool:
    weight = op.weight if op.single_grouped_weight else None
    weight_ok = (
        isinstance(weight, GroupedTensor)
        and weight.dtype is torch.bfloat16
        and weight.rowwise_data is not None
    )
    if weight_ok and weight.quantizer is not None:
        recipe = weight.quantizer._get_compatible_recipe()
        weight_ok = (
            recipe is not None
            and recipe.mxfp8()
            and not weight._with_gemm_swizzled_scales
            and weight.rowwise_data is not None
            and weight.scale_inv is not None
        )

    return (
        not op.use_bias
        and not op._scale_bias
        and op.single_grouped_weight
        and not op.single_grouped_bias
        and not op._accumulate_into_main_grad
        and not op._is_distributed_weight()
        and not op.wgrad_store.delay_wgrad_compute()
        and weight_ok
    )


def _import_cudnn_moe_ep():
    """Return ``cudnn.moe_ep.MoeEp`` or ``None`` if the package is missing."""
    try:
        from cudnn.moe_ep import MoeEp
    except ImportError:
        return None
    return MoeEp


def _routing_extras_internal(
    dispatch: Dispatch,
    fc1: GroupedLinear,
    activation: ScaledSwiGLU,
    fc2: GroupedLinear,
    combine: Combine,
) -> bool:
    """Whether the dispatch routing extras stay inside the fusion.

    The fused op keeps tokens-per-expert and the received routing weights
    internal to :class:`cudnn.moe_ep.MoeEp`, so it can only replace the
    sequence when those two outputs feed exactly these ops and are not
    returned to the caller.
    """
    tokens_per_expert, routing_weights, ep_handle, routing_indices = (
        dispatch._extra_output_channels
    )
    if (
        tokens_per_expert is None
        or routing_weights is None
        or ep_handle is None
        or routing_indices is None
    ):
        return False
    if any(dispatch._extra_output_to_caller):
        return False
    return (
        fc1._extra_input_channels[0] == tokens_per_expert
        and fc2._extra_input_channels[0] == tokens_per_expert
        and activation._extra_input_channels[0] == routing_weights
        and combine._extra_input_channels[0] == ep_handle
        and combine._extra_input_channels[1] == tokens_per_expert
        and combine._extra_input_channels[2] == routing_indices
    )


def _megamoe_supported(buffer, fc1: GroupedLinear, fc2: GroupedLinear) -> bool:
    """Static MegaMoE capability gates that can be checked before first launch."""
    if _import_cudnn_moe_ep() is None:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return False
    if buffer.max_tokens_per_rank is None or buffer.max_tokens_per_rank <= 0:
        return False
    if buffer.hidden_dim % 128 != 0 or fc2.in_features % 256 != 0:
        return False
    if buffer.top_k > 32:
        return False
    ep_group = get_ep_group()
    ep_size = 1 if ep_group is None else ep_group.size()
    if ep_size > 16:
        return False
    return True


def _matches(window: Sequence[FusibleOperation], recipe: Optional[Recipe]) -> bool:
    if len(window) != 5:
        return False
    if recipe is None or not recipe.mxfp8():
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
    buffer = dispatch.buffer
    if not buffer.eager or buffer.payload_dtype is not torch.bfloat16:
        return False
    if not (_grouped_linear_supported(fc1) and _grouped_linear_supported(fc2)):
        return False
    if activation.activation_recompute_in_mlp or activation.glu_interleave_size is not None:
        return False
    if not _routing_extras_internal(dispatch, fc1, activation, fc2, combine):
        return False
    if not _megamoe_supported(buffer, fc1, fc2):
        return False
    return (
        fc1.num_groups == buffer.num_local_experts
        and fc2.num_groups == buffer.num_local_experts
        and fc1.in_features == buffer.hidden_dim
        and fc2.out_features == buffer.hidden_dim
        and fc1.out_features == 2 * fc2.in_features
    )


class FusedMoeEp(FusedOperation):
    """Joint EP MoE fusion implemented with :class:`cudnn.moe_ep.MoeEp`."""

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
        moe_ep_cls = _import_cudnn_moe_ep()
        if moe_ep_cls is None:
            raise ImportError(
                "FusedMoeEp requires cudnn.moe_ep.MoeEp. Install the in-tree "
                "cuDNN frontend with: pip install --force-reinstall "
                "'./cudnn_frontend[moe_ep]'"
            )
        from cudnn.moe_ep import BlockScaledTensor

        ep_group = get_ep_group()
        ep_size = 1 if ep_group is None else ep_group.size()
        self._block_scaled_cls = BlockScaledTensor
        self._moe = moe_ep_cls(
            num_experts=dispatch.buffer.num_local_experts * ep_size,
            hidden_size=dispatch.buffer.hidden_dim,
            intermediate_size=fc2.in_features,
            top_k=dispatch.buffer.top_k,
            ep_group=ep_group,
            max_tokens_per_rank=dispatch.buffer.max_tokens_per_rank,
            apply_topk_in_fc1=True,
            generate_c=True,
            combine_format="bf16",
            output_format="bf16",
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
        if input_.dtype is not torch.bfloat16:
            raise NotImplementedError(f"FusedMoeEp requires BF16 input, got {input_.dtype}.")
        if any(kwargs for kwargs in basic_op_kwargs):
            raise NotImplementedError("FusedMoeEp does not support per-operation output buffers.")

        topk_idx, topk_weights = basic_op_extra_inputs[0]
        if topk_weights.dtype is not torch.float32:
            raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}.")
        input_quantizer = self.dispatch.get_quantizer("forward", 0)
        activation = _pack_cudnn_activation(
            input_,
            input_quantizer,
            self._block_scaled_cls,
        )
        fc1_weight = _pack_cudnn_weights(
            self.fc1, block_scaled_cls=self._block_scaled_cls
        )
        fc2_weight = _pack_cudnn_weights(
            self.fc2, block_scaled_cls=self._block_scaled_cls
        )
        output, fc1_c, route_metadata = self._moe(
            activation,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights,
        )

        if any(ctx.requires_grad for ctx in basic_op_ctxs):
            input_data, input_scale = activation.data, activation.scale
            fc1_data, fc1_scale = fc1_weight.data, fc1_weight.scale
            fc2_data, fc2_scale = fc2_weight.data, fc2_weight.scale
            basic_op_ctxs[0].save_for_backward(
                input_data,
                input_scale,
                fc1_data,
                fc1_scale,
                fc2_data,
                fc2_scale,
                topk_idx,
                topk_weights,
                fc1_c,
                route_metadata,
            )
            basic_op_ctxs[0].input_dtype = input_.dtype
            basic_op_ctxs[0].prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        # Dispatch extras are channel-bound with output_to_caller=False and are
        # only consumed by ops inside this fusion, so they need not be materialized.
        if next_op_input_quantizer is not None and not is_quantized_tensor(output):
            output = next_op_input_quantizer(output)
        return output, [
            (None, None, None, None),
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
            input_data,
            input_scale,
            fc1_data,
            fc1_scale,
            fc2_data,
            fc2_scale,
            topk_idx,
            topk_weights,
            fc1_c,
            route_metadata,
        ) = basic_op_ctxs[0].saved_tensors
        input_ = _pack_as_cudnn_moe_tensor(
            input_data,
            input_scale,
            self._block_scaled_cls,
        )
        fc1_weight = _pack_as_cudnn_moe_tensor(fc1_data, fc1_scale, self._block_scaled_cls)
        fc2_weight = _pack_as_cudnn_moe_tensor(fc2_data, fc2_scale, self._block_scaled_cls)
        grad_output = maybe_dequantize(
            grad_output,
            basic_op_ctxs[0].input_dtype,
        )
        grad_input, grad_fc1, grad_fc2, grad_topk_weights = self._moe.backward(
            grad_output,
            input_,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights,
            fc1_c,
            route_metadata,
        )

        # MegaMoE returns float32 grads in (E, in, out); each GroupedLinear has
        # one packed (E, out, in) parameter.
        fc1_param_grads = _grouped_weight_grad(self.fc1, grad_fc1)
        fc2_param_grads = _grouped_weight_grad(self.fc2, grad_fc2)
        grad_input = grad_input.to(dtype=basic_op_ctxs[0].input_dtype)
        grad_input_quantizer = basic_op_ctxs[0].prev_op_grad_output_quantizer
        if grad_input_quantizer is not None:
            grad_input = grad_input_quantizer(grad_input)
        return (
            grad_input,
            [(), fc1_param_grads, (), fc2_param_grads, ()],
            [(None, grad_topk_weights.float()), (None,), (None,), (None,), (None, None)],
        )


def fuse_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused: Any,
) -> list[FusibleOperation]:
    """Fuse supported five-op EP MoE sequences into MegaMoE.

    Unfused Sequential (NCCL dispatch/combine + GroupedLinear + ScaledSwiGLU)
    is the default. MegaMoE is claimed only when ``_matches`` succeeds.
    """
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
