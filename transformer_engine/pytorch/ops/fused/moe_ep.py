# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MegaMoE-backed expert-parallel MoE fusion (cudnn.moe_ep.MoeEp)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
from typing import Any, Optional

import torch
import transformer_engine_torch as tex

from ...constants import MXFP8_BLOCK_SCALING_SIZE
from ...quantization import Recipe
from ...tensor import GroupedTensor, Quantizer
from .._common import (
    get_accumulate_flag_in_param,
    get_dummy_wgrads_for_params,
    get_main_grad_from_param,
    is_quantized_tensor,
    maybe_dequantize,
    view_main_grad_as_grouped_buffer,
)
from ..basic import GroupedLinear, MoeCombine, MoeDispatch, ScaledSwiGLU
from ..fuser import register_forward_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext


def _cudnn_megamoe_supported() -> bool:
    """Whether cuDNN FE provides the fixed-resource training API."""
    try:
        import cudnn
        import cudnn.moe_ep as cudnn_moe_ep
    except (AttributeError, ImportError):
        return False
    return all(
        hasattr(module, name)
        for module, name in (
            (cudnn, "grouped_gemm_wgrad_wrapper_sm100"),
            (cudnn_moe_ep, "MoeEp"),
            (cudnn_moe_ep, "MoeEpTrainingWeights"),
            (cudnn_moe_ep, "MoeEpTrainingWgradOperands"),
        )
    )


def _get_megamoe_combine_format() -> str:
    """Return the MegaMoE combine wire format selected by the environment."""
    enabled = int(os.environ.get("NVTE_MEGAMOE_MXFP8_COMBINE", "0"))
    return "mxfp8" if enabled > 0 else "bf16"


def _get_megamoe_training_slot_count() -> int:
    """Return the fixed number of concurrent MegaMoE training flights."""
    value = os.environ.get("NVTE_MEGAMOE_TRAINING_SLOT_COUNT", "8")
    try:
        slot_count = int(value)
    except ValueError as exc:
        raise ValueError(
            f"NVTE_MEGAMOE_TRAINING_SLOT_COUNT must be a positive integer, got {value!r}"
        ) from exc
    if slot_count <= 0:
        raise ValueError(
            f"NVTE_MEGAMOE_TRAINING_SLOT_COUNT must be a positive integer, got {value!r}"
        )
    return slot_count


def _pack_as_cudnn_moe_tensor(
    data: torch.Tensor,
    scale: Optional[torch.Tensor],
    block_scaled_cls: type,
):
    """Represent data and scales using the public cuDNN MoE tensor type."""
    if scale is None:
        return data
    return block_scaled_cls(
        data=data,
        scale=scale,
        format="mxfp8",
        logical_shape=tuple(data.shape),
        axis=1,
    )


def _pack_cudnn_weights(
    op: GroupedLinear,
    *,
    block_scaled_cls: Optional[type] = None,
):
    """Pack TE rowwise/columnwise weight storage for cuDNN MoE.

    The rowwise binding is a zero-copy ``(E, in, out)`` K-major view over TE's
    rowwise ``(E, out, in)`` storage. The columnwise binding is a zero-copy
    ``(E, out, in)`` view backed by TE's columnwise storage.
    """
    weight = op.weight
    num_groups = op.num_groups
    out_features = op.out_features
    in_features = op.in_features
    weight_quantizer = op.get_quantizer("forward", 1)
    if weight_quantizer is None and weight.quantizer is None:
        return (
            weight.rowwise_data.view(
                num_groups,
                out_features,
                in_features,
            ).permute(0, 2, 1),
            None,
        )

    if weight_quantizer is not None and weight.quantizer is None:
        weight_quantizer.set_usage(rowwise=True, columnwise=True)
        weight = tex.group_quantize(
            weight.rowwise_data.view(weight.logical_shape),
            weight_quantizer,
            op.num_groups,
            None,
        )
    data = (
        weight.rowwise_data.view(num_groups, out_features, in_features)
        .view(torch.float8_e4m3fn)
        .permute(0, 2, 1)
    )
    scale = (
        weight.scale_inv.view(
            num_groups,
            out_features,
            in_features // MXFP8_BLOCK_SCALING_SIZE,
        )
        .view(torch.float8_e8m0fnu)
        .permute(0, 2, 1)
    )
    if block_scaled_cls is None:
        from cudnn.moe_ep import BlockScaledTensor as block_scaled_cls
    if weight.columnwise_data is None or weight.columnwise_scale_inv is None:
        raise ValueError("FusedMoeEp training requires columnwise MXFP8 weight storage")

    columnwise_data = weight.columnwise_data.view(
        num_groups,
        out_features,
        in_features,
    ).view(torch.float8_e4m3fn)
    columnwise_scale = weight.columnwise_scale_inv.view(
        num_groups,
        out_features // MXFP8_BLOCK_SCALING_SIZE,
        in_features,
    ).view(torch.float8_e8m0fnu)
    return (
        _pack_as_cudnn_moe_tensor(data, scale, block_scaled_cls),
        _pack_as_cudnn_moe_tensor(
            columnwise_data,
            columnwise_scale,
            block_scaled_cls,
        ),
    )


def _launch_grouped_wgrad_from_operands(
    layer_operands: list[torch.Tensor],
    unused: None,
    output: torch.Tensor | GroupedTensor,
    *,
    offsets: torch.Tensor,
    accumulate: bool,
    descriptor_workspace: torch.Tensor,
) -> None:
    """Compute one TE-layout grouped wgrad directly from MegaMoE's operands."""
    del unused
    from cudnn import grouped_gemm_wgrad_wrapper_sm100

    x_transpose, x_scale, dy, dy_scale = layer_operands
    output_data = (
        output.rowwise_data.view(output.shape) if isinstance(output, GroupedTensor) else output
    )
    # MegaMoE exports X^T and dY. Present the same dY^T @ X convention used by
    # grouped MLP. The transposes are views, and cuDNN writes directly to TE's
    # contiguous (expert, out, in) gradient buffer.
    # The public wrapper selects the Rubin specialization on SM107.
    grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=dy.transpose(0, 1),
        b_tensor=x_transpose.transpose(0, 1),
        sfa_tensor=dy_scale,
        sfb_tensor=x_scale,
        offsets_tensor=offsets,
        output_mode="dense",
        wgrad_tensor=output_data,
        wgrad_dtype=output_data.dtype,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
        accumulate_on_output=accumulate,
        input_order="tensor2d",
        descriptor_workspace=descriptor_workspace,
    )


def _compute_grouped_weight_grad(
    op: GroupedLinear,
    operands,
    prefix: str,
    descriptor_workspace: torch.Tensor,
) -> list[Optional[torch.Tensor]]:
    """Compute one dense weight gradient with cuDNN's grouped-WGrad API."""
    weight = op.weight
    if not weight.requires_grad:
        return [None]

    weight_shape = (op.out_features, op.in_features)
    output_shape = (op.num_groups, *weight_shape)
    accumulate = False
    if op._accumulate_into_main_grad:
        output_data = get_main_grad_from_param(
            weight,
            op_label=f"FusedMoeEp {prefix.upper()}",
        )
        output_data = view_main_grad_as_grouped_buffer(
            output_data,
            op.num_groups,
            weight_shape,
            label=f"FusedMoeEp {prefix.upper()} weight",
        )
        accumulate = get_accumulate_flag_in_param(weight)
    else:
        output_data = torch.empty(
            output_shape,
            dtype=weight.dtype,
            device=weight.device,
        )

    layer_operands = [
        getattr(operands, f"{prefix}_a"),
        getattr(operands, f"{prefix}_sfa"),
        getattr(operands, f"{prefix}_b"),
        getattr(operands, f"{prefix}_sfb"),
    ]
    _launch_grouped_wgrad_from_operands(
        layer_operands,
        None,
        output_data,
        offsets=operands.expert_offsets,
        accumulate=accumulate,
        descriptor_workspace=descriptor_workspace,
    )

    if op._accumulate_into_main_grad:
        return get_dummy_wgrads_for_params([weight])
    return [output_data]


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
            and weight.columnwise_data is not None
            and weight.columnwise_scale_inv is not None
        )
    else:
        weight_ok = False

    return (
        not op.use_bias
        and not op._scale_bias
        and op.single_grouped_weight
        and not op.single_grouped_bias
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
    dispatch: MoeDispatch,
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


def _megamoe_supported(config, fc2: GroupedLinear) -> bool:
    """Static MegaMoE capability gates that can be checked before first launch."""
    if not _cudnn_megamoe_supported():
        return False
    if _import_cudnn_moe_ep() is None:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return False
    if config.max_tokens_per_rank <= 0:
        return False
    if config.recv_capacity_per_rank is None or config.recv_capacity_per_rank <= 0:
        return False
    if config.hidden_dim % 128 != 0 or fc2.in_features % 256 != 0:
        return False
    if config.top_k > 32:
        return False
    return True


def is_moe_fusion_supported(
    window: Sequence[FusibleOperation],
    recipe: Optional[Recipe],
) -> bool:
    """Whether a five-op Sequential window supports FusedMoeEp."""
    if len(window) != 5:
        return False
    if recipe is not None and not recipe.mxfp8():
        return False
    dispatch, fc1, activation, fc2, combine = window
    if not (
        isinstance(dispatch, MoeDispatch)
        and isinstance(fc1, GroupedLinear)
        and isinstance(activation, ScaledSwiGLU)
        and isinstance(fc2, GroupedLinear)
        and isinstance(combine, MoeCombine)
    ):
        return False
    config = dispatch.config
    if combine.config != config or config.payload_dtype is not torch.bfloat16:
        return False
    if not (_grouped_linear_supported(fc1) and _grouped_linear_supported(fc2)):
        return False
    if activation.activation_recompute_in_mlp or activation.glu_interleave_size != 32:
        return False
    if not _routing_extras_internal(dispatch, fc1, activation, fc2):
        return False
    if not _megamoe_supported(config, fc2):
        return False
    return (
        fc1.num_groups == config.num_local_experts
        and fc2.num_groups == config.num_local_experts
        and fc1.in_features == config.hidden_dim
        and fc2.out_features == config.hidden_dim
        and fc1.out_features == 2 * fc2.in_features
    )


class FusedMoeEp(FusedOperation):
    """Joint EP MoE fusion implemented with :class:`cudnn.moe_ep.MoeEp`."""

    def __init__(
        self,
        *,
        dispatch: MoeDispatch,
        fc1: GroupedLinear,
        activation: ScaledSwiGLU,
        fc2: GroupedLinear,
        combine: MoeCombine,
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

        config = dispatch.config
        ep_group = config.ep_group
        ep_size = ep_group.size()
        combine_format = _get_megamoe_combine_format()
        self._block_scaled_cls = BlockScaledTensor
        self._moe = moe_ep_cls(
            num_experts=config.num_local_experts * ep_size,
            hidden_size=config.hidden_dim,
            intermediate_size=fc2.in_features,
            top_k=config.top_k,
            ep_group=ep_group,
            max_tokens_per_rank=config.max_tokens_per_rank,
            max_recv_size_per_rank=config.recv_capacity_per_rank,
            drop_on_overflow=config.drop_on_overflow,
            apply_topk_in_fc1=True,
            weight_interleave_size=activation.glu_interleave_size,
            token_padding_size=128,
            sf_padding_size=128,
            combine_format=combine_format,
            output_format="bf16",
        )
        self._training_resources = None
        self._training_slot_count = _get_megamoe_training_slot_count()
        self._free_training_slots = []
        self._active_training_slots = set()
        self._training_wgrad_workspaces = {}

    def _make_training_weights(self):
        """Bind cuDNN weight views to the GroupedLinear parameters' current storage."""
        from cudnn.moe_ep import MoeEpTrainingWeights

        fc1_rowwise, fc1_columnwise = _pack_cudnn_weights(
            self.fc1,
            block_scaled_cls=self._block_scaled_cls,
        )
        fc2_rowwise, fc2_columnwise = _pack_cudnn_weights(
            self.fc2,
            block_scaled_cls=self._block_scaled_cls,
        )
        return MoeEpTrainingWeights(
            forward_fc1=fc1_rowwise,
            forward_fc2=fc2_rowwise,
            backward_w2_transpose=fc2_columnwise,
            backward_w1_transpose=fc1_columnwise,
        )

    def _make_training_wgrad_workspaces(self, slots):
        """Allocate caller-owned descriptor workspaces for each slot and FC."""
        from cudnn import get_grouped_gemm_wgrad_workspace_size_sm100

        workspace_bytes = get_grouped_gemm_wgrad_workspace_size_sm100(
            self.fc1.num_groups,
            output_mode="dense",
            input_order="tensor2d",
        )
        return {
            slot: (
                torch.empty(workspace_bytes, dtype=torch.uint8, device=self.fc1.weight.device),
                torch.empty(workspace_bytes, dtype=torch.uint8, device=self.fc2.weight.device),
            )
            for slot in slots
        }

    def _begin_training_flight(self):
        """Reserve one fixed training slot for a forward/backward flight."""
        if self._training_resources is None:
            self._training_resources = self._moe.prepare_training_resources(
                self._make_training_weights(),
                slot_count=self._training_slot_count,
                lane_count=1,
            )
            self._free_training_slots.extend(self._training_resources.slots)
            self._training_wgrad_workspaces.update(
                self._make_training_wgrad_workspaces(self._training_resources.slots)
            )
        if not self._free_training_slots:
            raise RuntimeError(
                "FusedMoeEp has no free training slots; increase "
                "NVTE_MEGAMOE_TRAINING_SLOT_COUNT "
                f"(currently {self._training_slot_count}) "
                "or complete backward for an outstanding microbatch"
            )
        if not self._active_training_slots:
            self._training_resources.refresh_weights()
        slot = self._free_training_slots.pop(0)
        self._active_training_slots.add(slot)
        return slot

    def _release_training_flight(self, slot) -> None:
        """Return a completed forward/backward flight's slot to the pool."""
        if slot not in self._active_training_slots:
            raise RuntimeError("FusedMoeEp attempted to release an inactive training slot")
        self._active_training_slots.remove(slot)
        self._free_training_slots.append(slot)

    @property
    def dispatch(self) -> MoeDispatch:
        """Return the underlying dispatch operation."""
        return self.basic_ops[0]

    @property
    def fc1(self) -> GroupedLinear:
        """Return the first grouped linear operation."""
        return self.basic_ops[1]

    @property
    def fc2(self) -> GroupedLinear:
        """Return the second grouped linear operation."""
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
        if (
            not isinstance(input_, torch.Tensor)
            or is_quantized_tensor(input_)
            or input_.dtype is not torch.bfloat16
        ):
            raise TypeError(
                f"FusedMoeEp input must be a plain BF16 tensor, got {type(input_).__name__}."
            )

        topk_idx, topk_weights = basic_op_extra_inputs[0]
        if topk_weights.dtype is not torch.float32:
            raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}.")
        activation = input_
        if not activation.is_contiguous():
            activation = activation.contiguous()
        if topk_idx.dtype is not torch.int32:
            topk_idx = topk_idx.to(dtype=torch.int32)
        if not topk_idx.is_contiguous():
            topk_idx = topk_idx.contiguous()
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()

        slot = self._begin_training_flight()
        try:
            output = self._training_resources.forward(
                slot,
                self._training_resources.lanes[0],
                activation,
                topk_idx,
                topk_weights,
            )
        except Exception:
            self._release_training_flight(slot)
            raise

        if any(ctx.requires_grad for ctx in basic_op_ctxs):
            basic_op_ctxs[0].moe_ep_training_slot = slot
            basic_op_ctxs[0].prev_op_grad_output_quantizer = prev_op_grad_output_quantizer
        else:
            try:
                self._training_resources.finalize_overflow(
                    (slot,),
                    self._training_resources.lanes[0],
                )
            finally:
                self._release_training_flight(slot)

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
        grad_output = maybe_dequantize(
            grad_output,
            torch.bfloat16,
        )
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        slot = basic_op_ctxs[0].moe_ep_training_slot
        try:
            grad_input, grad_topk_weights, wgrad_operands = self._training_resources.backward(
                slot,
                self._training_resources.lanes[0],
                grad_output,
            )
            fc1_workspace, fc2_workspace = self._training_wgrad_workspaces[slot]
            fc1_param_grads = _compute_grouped_weight_grad(
                self.fc1,
                wgrad_operands,
                "fc1",
                fc1_workspace,
            )
            fc2_param_grads = _compute_grouped_weight_grad(
                self.fc2,
                wgrad_operands,
                "fc2",
                fc2_workspace,
            )
            self._training_resources.finalize_overflow(
                (slot,),
                self._training_resources.lanes[0],
            )
        finally:
            self._release_training_flight(slot)
        return (
            grad_input,
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
    is the default. MegaMoE is claimed only when
    :func:`is_moe_fusion_supported` succeeds.
    """
    del unused
    out: list[FusibleOperation] = []
    idx = 0
    while idx < len(ops):
        window = ops[idx : idx + 5]
        if is_moe_fusion_supported(window, recipe):
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


__all__ = ["FusedMoeEp", "is_moe_fusion_supported"]
