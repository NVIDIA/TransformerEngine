# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MegaMoE-backed expert-parallel MoE fusion (cudnn.moe_ep.MoeEp)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional

import torch

from ...constants import MXFP8_BLOCK_SCALING_SIZE
from ...ep import get_ep_group
from ...quantization import Recipe
from ...tensor import Quantizer
from ...tensor.mxfp8_tensor import MXFP8Tensor
from ..basic import Combine, Dispatch, GroupedLinear, ScaledSwiGLU
from ..fuser import register_forward_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext


def _weight_list(op: GroupedLinear) -> list[torch.Tensor]:
    """Return per-expert weights in their registered order."""
    return [getattr(op, f"weight{idx}") for idx in range(op.num_groups)]


def _mxfp8_weight_k_major(weight: MXFP8Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose a TE ``(out, in)`` MXFP8 weight to MegaMoE ``(in, out)``.

    Rowwise block scales travel with the ``in`` axis (K after the transpose).
    GEMM-swizzled CuBLAS layouts are not a MegaMoE input; GroupedLinear stores
    compact unswizzled scales under quantized model init.
    """
    if weight._with_gemm_swizzled_scales:
        raise NotImplementedError(
            "FusedMoeEp requires unswizzled MXFP8 weight scales, got a GEMM-swizzled tensor"
        )
    if weight._rowwise_data is None or weight._rowwise_scale_inv is None:
        raise ValueError("MXFP8 weight is missing rowwise data or scales")
    out_features, in_features = weight.size()
    if in_features % MXFP8_BLOCK_SCALING_SIZE != 0:
        raise ValueError(
            f"MXFP8 weight K={in_features} is not divisible by {MXFP8_BLOCK_SCALING_SIZE}"
        )
    data = weight._rowwise_data.view(torch.float8_e4m3fn).transpose(0, 1).contiguous()
    scale = (
        weight._rowwise_scale_inv[:out_features, : in_features // MXFP8_BLOCK_SCALING_SIZE]
        .view(torch.float8_e8m0fnu)
        .transpose(0, 1)
        .contiguous()
    )
    return data, scale


def _pack_grouped_linear_weights(op: GroupedLinear, *, block_scaled_cls: Optional[type] = None):
    """Pack TE ``(out, in)`` expert weights into MegaMoE ``(E, in, out)``.

    Dense BF16 stays a dense tensor. MXFP8 stays MXFP8: payloads and compact
    scales are restacked without dequantizing. ``block_scaled_cls`` selects the
    ``BlockScaledTensor`` type (cuDNN MegaMoE vs the PyTorch reference).
    """
    weights = _weight_list(op)
    if not weights:
        raise ValueError("GroupedLinear has no per-expert weights to pack")
    if all(isinstance(weight, MXFP8Tensor) for weight in weights):
        if block_scaled_cls is None:
            from cudnn.moe_ep import BlockScaledTensor as block_scaled_cls
        data = []
        scale = []
        for weight in weights:
            expert_data, expert_scale = _mxfp8_weight_k_major(weight)
            data.append(expert_data)
            scale.append(expert_scale)
        packed_data = torch.stack(data)
        return block_scaled_cls(
            data=packed_data,
            scale=torch.stack(scale),
            format="mxfp8",
            logical_shape=tuple(packed_data.shape),
            axis=1,
        )
    if any(isinstance(weight, MXFP8Tensor) for weight in weights):
        raise TypeError("cannot mix MXFP8 and dense expert weights")
    return torch.stack([weight.transpose(0, 1).contiguous() for weight in weights])


def _flatten_moe_weight(weight) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split a MegaMoE weight into tensors that ``save_for_backward`` can hold."""
    if isinstance(weight, torch.Tensor):
        return weight, None
    return weight.data, weight.scale


def _restore_moe_weight(data: torch.Tensor, scale: Optional[torch.Tensor], block_scaled_cls: type):
    """Rebuild the MegaMoE weight saved by :func:`_flatten_moe_weight`."""
    if scale is None:
        return data
    return block_scaled_cls(
        data=data,
        scale=scale,
        format="mxfp8",
        logical_shape=tuple(data.shape),
        axis=1,
    )


def _grouped_linear_supported(op: GroupedLinear) -> bool:
    weights = _weight_list(op) if not op.single_grouped_weight else []

    def _weight_ok(weight: torch.Tensor) -> bool:
        if weight.dtype is not torch.bfloat16:
            return False
        if isinstance(weight, MXFP8Tensor):
            return True
        return not hasattr(weight, "dequantize")

    return (
        not op.use_bias
        and not op._scale_bias
        and not op.single_grouped_weight
        and not op.single_grouped_bias
        and not op._accumulate_into_main_grad
        and not op._is_distributed_weight()
        and not op.wgrad_store.delay_wgrad_compute()
        and bool(weights)
        and all(_weight_ok(weight) for weight in weights)
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
) -> bool:
    """Whether the dispatch routing extras stay inside the fusion.

    The fused op keeps tokens-per-expert and the received routing weights
    internal to :class:`cudnn.moe_ep.MoeEp`, so it can only replace the
    sequence when those two outputs feed exactly these ops and are not
    returned to the caller.
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


def _megamoe_supported(buffer, fc1: GroupedLinear, fc2: GroupedLinear) -> bool:
    """Static MegaMoE capability gates that can be checked before first launch."""
    if _import_cudnn_moe_ep() is None:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return False
    if buffer.max_tokens_per_rank is None or buffer.max_tokens_per_rank <= 0:
        return False
    if buffer.hidden_dim % 128 != 0 or fc2.in_features % 128 != 0:
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
    if recipe is not None and not recipe.mxfp8():
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
        with torch.no_grad():
            fc1_weight = _pack_grouped_linear_weights(
                self.fc1, block_scaled_cls=self._block_scaled_cls
            )
            fc2_weight = _pack_grouped_linear_weights(
                self.fc2, block_scaled_cls=self._block_scaled_cls
            )
        output, fc1_c, route_metadata = self._moe(
            input_,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights,
        )

        if any(ctx.requires_grad for ctx in basic_op_ctxs):
            fc1_data, fc1_scale = _flatten_moe_weight(fc1_weight)
            fc2_data, fc2_scale = _flatten_moe_weight(fc2_weight)
            basic_op_ctxs[0].save_for_backward(
                input_,
                fc1_data,
                fc1_scale,
                fc2_data,
                fc2_scale,
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
            fc1_data,
            fc1_scale,
            fc2_data,
            fc2_scale,
            topk_idx,
            topk_weights,
            fc1_c,
            route_metadata,
        ) = basic_op_ctxs[0].saved_tensors
        fc1_weight = _restore_moe_weight(fc1_data, fc1_scale, self._block_scaled_cls)
        fc2_weight = _restore_moe_weight(fc2_data, fc2_scale, self._block_scaled_cls)
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

        # MegaMoE returns float32 grads in (E, in, out); TE parameters are (out, in).
        fc1_param_grads = [
            (grad_fc1[idx].transpose(0, 1).to(dtype=weight.dtype) if weight.requires_grad else None)
            for idx, weight in enumerate(_weight_list(self.fc1))
        ]
        fc2_param_grads = [
            (grad_fc2[idx].transpose(0, 1).to(dtype=weight.dtype) if weight.requires_grad else None)
            for idx, weight in enumerate(_weight_list(self.fc2))
        ]
        return (
            grad_input.to(dtype=input_.dtype),
            [(), fc1_param_grads, (), fc2_param_grads, ()],
            [(None, grad_topk_weights.float()), (None,), (None,), (None,), ()],
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
