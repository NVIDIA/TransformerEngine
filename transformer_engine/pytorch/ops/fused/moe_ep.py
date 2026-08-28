# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MegaMoE-backed expert-parallel MoE fusion (cudnn.moe_ep.MoeEp)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import functools
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
import os
from typing import Any, Optional

from packaging.version import Version as PkgVersion
import torch
import transformer_engine_torch as tex

from ...constants import DType, MXFP8_BLOCK_SCALING_SIZE
from ...ep import get_ep_group, quantize_for_ep
from ...quantization import Recipe
from ...tensor import GroupedTensor, MXFP8Quantizer, Quantizer
from ...tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
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
    """Whether the installed cuDNN frontend includes the public MegaMoE API."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion("1.28.0")
    except PackageNotFoundError:
        return False


def _get_megamoe_combine_format() -> str:
    """Return the MegaMoE combine wire format selected by the environment."""
    enabled = int(os.environ.get("NVTE_MEGAMOE_MXFP8_COMBINE", "0"))
    return "mxfp8" if enabled > 0 else "bf16"


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
    input_: torch.Tensor | MXFP8TensorStorage,
    quantizer: Optional[MXFP8Quantizer],
    block_scaled_cls: type,
):
    """Pack the dispatch input in the public cuDNN MoE activation layout."""
    if quantizer is None and not isinstance(input_, MXFP8TensorStorage):
        return input_
    quantized, scale = quantize_for_ep(input_, quantizer)
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


def _launch_grouped_wgrad_from_operands(
    layer_operands: list[torch.Tensor],
    unused: None,
    output: torch.Tensor | GroupedTensor,
    *,
    offsets: torch.Tensor,
    accumulate: bool,
) -> None:
    """Compute one TE-layout grouped wgrad directly from MegaMoE's operands."""
    del unused
    from cudnn.gemm.cutedsl.grouped.wgrad import grouped_gemm_wgrad_wrapper_sm100

    a_tensor, sfa_tensor, b_tensor, sfb_tensor = layer_operands
    output_data = (
        output.rowwise_data.view(output.shape) if isinstance(output, GroupedTensor) else output
    )
    # MegaMoE exports dW in (in, out) layout. Swapping operands computes
    # B.T @ A.T directly into TE's contiguous (out, in) parameter layout.
    grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=b_tensor.transpose(0, 1),
        b_tensor=a_tensor.transpose(0, 1),
        sfa_tensor=sfb_tensor,
        sfb_tensor=sfa_tensor,
        offsets_tensor=offsets,
        output_mode="dense",
        wgrad_tensor=output_data,
        wgrad_dtype=output_data.dtype,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=MXFP8_BLOCK_SCALING_SIZE,
        accumulate_on_output=accumulate,
    )


def _compute_grouped_weight_grad(
    op: GroupedLinear,
    operands,
    prefix: str,
) -> list[Optional[torch.Tensor]]:
    """Launch or defer one operand-mode wgrad with GroupedLinear semantics."""
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
    launch = functools.partial(
        _launch_grouped_wgrad_from_operands,
        offsets=operands.expert_offsets,
        accumulate=accumulate,
    )
    delay_wgrad = op.wgrad_store is not None and op.wgrad_store.delay_wgrad_compute()
    if delay_wgrad:
        grouped_output = GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=op.num_groups,
            tensor_shape=weight_shape,
            rowwise_data=output_data,
            dtype=output_data.dtype,
        )
        op.wgrad_store.put([layer_operands, None, grouped_output], launch)
    else:
        launch(layer_operands, None, output_data)

    if op._accumulate_into_main_grad:
        return get_dummy_wgrads_for_params([weight])
    if delay_wgrad:
        return [None]
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
        )

    return (
        not op.use_bias
        and not op._scale_bias
        and op.single_grouped_weight
        and not op.single_grouped_bias
        and not op._is_distributed_weight()
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


def _megamoe_supported(config, fc1: GroupedLinear, fc2: GroupedLinear) -> bool:
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


def _matches(window: Sequence[FusibleOperation], recipe: Optional[Recipe]) -> bool:
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
    if activation.activation_recompute_in_mlp or activation.glu_interleave_size is not None:
        return False
    if not _routing_extras_internal(dispatch, fc1, activation, fc2):
        return False
    if not _megamoe_supported(config, fc1, fc2):
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

        ep_group = get_ep_group()
        ep_size = 1 if ep_group is None else ep_group.size()
        config = dispatch.config
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
            generate_c=True,
            backward_wgrad_mode="operands",
            token_padding_size=256,
            sf_padding_size=128,
            combine_format=combine_format,
            output_format="bf16",
        )

    @property
    def dispatch(self) -> MoeDispatch:
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
        input_: torch.Tensor | MXFP8TensorStorage,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Sequence[Sequence[Optional[torch.Tensor]]]]:
        is_mxfp8 = isinstance(input_, MXFP8TensorStorage)
        if is_quantized_tensor(input_) and not is_mxfp8:
            raise TypeError(
                f"FusedMoeEp supports BF16 and MXFP8 inputs, got {type(input_).__name__}."
            )
        input_dtype = input_.dtype if isinstance(input_, torch.Tensor) else input_._dtype
        if input_dtype is not torch.bfloat16:
            raise TypeError(
                "FusedMoeEp input must be BF16 or an MXFP8 tensor representing BF16 values, "
                f"got logical dtype {input_dtype}."
            )
        if is_mxfp8 and input_._fp8_dtype != DType.kFloat8E4M3:
            raise NotImplementedError(
                f"FusedMoeEp supports E4M3 MXFP8 only, got {input_._fp8_dtype}."
            )

        topk_idx, topk_weights = basic_op_extra_inputs[0]
        if topk_weights.dtype is not torch.float32:
            raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}.")
        input_quantizer = self.dispatch.get_quantizer("forward", 0)
        activation = _pack_cudnn_activation(
            input_,
            input_quantizer,
            self._block_scaled_cls,
        )
        fc1_weight = _pack_cudnn_weights(self.fc1, block_scaled_cls=self._block_scaled_cls)
        fc2_weight = _pack_cudnn_weights(self.fc2, block_scaled_cls=self._block_scaled_cls)
        output, fc1_c, route_metadata, wgrad_forward_stash = self._moe(
            activation,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights,
        )

        if any(ctx.requires_grad for ctx in basic_op_ctxs):
            fc1_data = fc1_weight if isinstance(fc1_weight, torch.Tensor) else fc1_weight.data
            fc1_scale = None if isinstance(fc1_weight, torch.Tensor) else fc1_weight.scale
            fc2_data = fc2_weight if isinstance(fc2_weight, torch.Tensor) else fc2_weight.data
            fc2_scale = None if isinstance(fc2_weight, torch.Tensor) else fc2_weight.scale
            basic_op_ctxs[0].save_for_backward(
                fc1_data,
                fc1_scale,
                fc2_data,
                fc2_scale,
                topk_idx,
                topk_weights,
                fc1_c,
                route_metadata,
                wgrad_forward_stash.fc1_a,
                wgrad_forward_stash.fc1_sfa,
                wgrad_forward_stash.expert_offsets,
                wgrad_forward_stash.valid_route_counts,
                wgrad_forward_stash.route_metadata,
            )
            basic_op_ctxs[0].input_dtype = input_dtype
            basic_op_ctxs[0].prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        # Dispatch extras are channel-bound with output_to_caller=False and are
        # only consumed by ops inside this fusion, so they need not be materialized.
        if next_op_input_quantizer is not None and not is_quantized_tensor(output):
            output = next_op_input_quantizer(output)
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
            fc1_data,
            fc1_scale,
            fc2_data,
            fc2_scale,
            topk_idx,
            topk_weights,
            fc1_c,
            route_metadata,
            wgrad_fc1_a,
            wgrad_fc1_sfa,
            wgrad_expert_offsets,
            wgrad_valid_route_counts,
            wgrad_route_metadata,
        ) = basic_op_ctxs[0].saved_tensors
        fc1_weight = _pack_as_cudnn_moe_tensor(fc1_data, fc1_scale, self._block_scaled_cls)
        fc2_weight = _pack_as_cudnn_moe_tensor(fc2_data, fc2_scale, self._block_scaled_cls)
        grad_output = maybe_dequantize(
            grad_output,
            basic_op_ctxs[0].input_dtype,
        )
        from cudnn.moe_ep import MoeEpWgradForwardStash

        wgrad_forward_stash = MoeEpWgradForwardStash(
            fc1_a=wgrad_fc1_a,
            fc1_sfa=wgrad_fc1_sfa,
            expert_offsets=wgrad_expert_offsets,
            valid_route_counts=wgrad_valid_route_counts,
            route_metadata=wgrad_route_metadata,
        )
        grad_input, grad_topk_weights, wgrad_operands = self._moe.backward(
            grad_output,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights,
            fc1_c,
            route_metadata,
            wgrad_forward_stash=wgrad_forward_stash,
        )

        fc1_param_grads = _compute_grouped_weight_grad(self.fc1, wgrad_operands, "fc1")
        fc2_param_grads = _compute_grouped_weight_grad(self.fc2, wgrad_operands, "fc2")
        grad_input = grad_input.to(dtype=basic_op_ctxs[0].input_dtype)
        grad_input_quantizer = basic_op_ctxs[0].prev_op_grad_output_quantizer
        if grad_input_quantizer is not None:
            grad_input = grad_input_quantizer(grad_input)
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
