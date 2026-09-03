# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MegaMoE-backed expert-parallel MoE fusion (cudnn.moe_ep.MoeEp)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import os
from typing import Any, Optional

import torch
import transformer_engine_torch as tex

from ...constants import MXFP8_BLOCK_SCALING_SIZE
from ...ep import EpConfig
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


@dataclass
class _MoeEpResource:
    """One cached cuDNN operator and its prepared symmetric buffers."""

    moe: Any
    requirements: Any = None
    lane: Any = None
    symmetric_buffers: Any = None
    device: Optional[torch.device] = None


class _MoeEpResourceManager:
    """Share one cuDNN MoeEp instance per compatible fused-MoE configuration."""

    def __init__(self) -> None:
        self._resources: dict[tuple[Any, ...], _MoeEpResource] = {}

    def get(
        self,
        config: EpConfig,
        device: torch.device,
        intermediate_size: int,
        glu_interleave_size: int,
    ) -> _MoeEpResource:
        """Get or construct the cuDNN operator for one complete configuration."""
        device = torch.device(device)
        combine_format = _get_megamoe_combine_format()
        key = self._resource_key(
            config,
            device,
            intermediate_size,
            glu_interleave_size,
            combine_format,
        )
        resource = self._resources.get(key)
        if resource is None:
            resource = _MoeEpResource(
                moe=self._construct_moe(
                    config,
                    intermediate_size,
                    glu_interleave_size,
                    combine_format,
                ),
            )
            self._resources[key] = resource
        self._prepare(resource, device)
        return resource

    @staticmethod
    def _construct_moe(
        config: EpConfig,
        intermediate_size: int,
        glu_interleave_size: int,
        combine_format: str,
    ):
        """Construct one cuDNN MoeEp instance from its complete static configuration."""
        moe_ep_cls = _import_cudnn_moe_ep()
        if moe_ep_cls is None:
            raise ImportError(
                "FusedMoeEp requires cudnn.moe_ep.MoeEp. Install the in-tree "
                "cuDNN frontend with: pip install --force-reinstall "
                "'./cudnn_frontend[moe_ep]'"
            )
        from cudnn.moe_ep import MoeEpTuningConfig

        forward_group_hint = 768 if combine_format == "mxfp8" else 1024
        return moe_ep_cls(
            num_experts=config.num_local_experts * config.ep_group.size(),
            hidden_size=config.hidden_dim,
            intermediate_size=intermediate_size,
            top_k=config.top_k,
            ep_group=config.ep_group,
            max_tokens_per_rank=config.max_tokens_per_rank,
            max_recv_size_per_rank=config.recv_capacity_per_rank,
            drop_on_overflow=config.drop_on_overflow,
            apply_topk_in_fc1=True,
            weight_interleave_size=glu_interleave_size,
            token_padding_size=128,
            sf_padding_size=128,
            combine_format=combine_format,
            output_format="bf16",
            forward_tuning=MoeEpTuningConfig(
                token_back_mode="standalone_warps",
                epi_flag_batch=(2, 2),
                token_in_flag_batch=8,
                group_hint=forward_group_hint,
                reduce_topk_in_kernel=False,
            ),
            backward_tuning=MoeEpTuningConfig(
                token_back_mode="epi_warps",
                epi_flag_batch=(2, 2),
                token_in_flag_batch=8,
                group_hint=512,
                reduce_topk_in_kernel=False,
            ),
        )

    @staticmethod
    def _resource_key(
        config: EpConfig,
        device: torch.device,
        intermediate_size: int,
        glu_interleave_size: int,
        combine_format: str,
    ) -> tuple[Any, ...]:
        """Return every value that affects a shared cuDNN MoeEp instance."""
        return (
            id(config.ep_group),
            device.type,
            device.index,
            config.top_k,
            config.hidden_dim,
            intermediate_size,
            config.num_local_experts,
            config.max_tokens_per_rank,
            config.recv_capacity_per_rank,
            config.alignment,
            config.payload_dtype,
            config.zero_copy,
            config.drop_on_overflow,
            glu_interleave_size,
            combine_format,
        )

    def _prepare(
        self,
        resource: _MoeEpResource,
        device: torch.device,
    ) -> tuple[Any, Any, Any]:
        """Prepare and cache one resource's lane and symmetric buffer views."""
        device = torch.device(device)
        if resource.requirements is None:
            resource.requirements = resource.moe.prepare_training(
                lane_count=1,
                device=device,
            )
            resource.lane = resource.moe.training_lanes[0]
            resource.symmetric_buffers = resource.moe.training_symmetric_buffers(resource.lane)
            resource.device = device
        elif resource.device != device:
            raise ValueError(
                f"shared MoeEp is prepared on {resource.device}, but was requested on {device}"
            )
        return resource.requirements, resource.lane, resource.symmetric_buffers

    def cleanup(self) -> None:
        """Close every cached MoeEp instance and clear the registry."""
        first_error = None
        for resource in tuple(self._resources.values()):
            try:
                if resource.moe is not None:
                    resource.moe.close()
            except Exception as error:  # pylint: disable=broad-exception-caught
                if first_error is None:
                    first_error = error
            finally:
                resource.requirements = None
                resource.lane = None
                resource.symmetric_buffers = None
                resource.device = None
                resource.moe = None
        self._resources.clear()
        if first_error is not None:
            raise first_error


_MOE_EP_RESOURCE_MANAGER = _MoeEpResourceManager()


def _cudnn_megamoe_supported() -> bool:
    """Whether cuDNN FE provides the stateless training API."""
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
            (cudnn_moe_ep, "MoeEpTuningConfig"),
            (cudnn_moe_ep.MoeEp, "training_symmetric_buffers"),
            (cudnn_moe_ep, "MoeEpNativeForwardWeights"),
            (cudnn_moe_ep, "MoeEpNativeBackwardWeights"),
            (cudnn_moe_ep, "MoeEpTrainingForwardOutputs"),
            (cudnn_moe_ep, "MoeEpTrainingBackwardOutputs"),
            (cudnn_moe_ep, "MoeEpTrainingWgradOperands"),
        )
    )


def finalize_moe_ep_resources() -> None:
    """Close all cached cuDNN/NVSHMEM resources before EP group teardown."""
    # CUDA graph callables form reference cycles. Collect unreachable graph
    # executables before tearing down communication resources they captured.
    import gc
    gc.collect()
    _MOE_EP_RESOURCE_MANAGER.cleanup()


def _get_megamoe_combine_format() -> str:
    """Return the MegaMoE combine wire format selected by the environment."""
    enabled = int(os.environ.get("NVTE_MEGAMOE_MXFP8_COMBINE", "0"))
    return "mxfp8" if enabled > 0 else "bf16"


def _allocate_training_buffer(requirements, name: str, device: torch.device) -> torch.Tensor:
    """Allocate one caller-owned buffer from cuDNN's named contract."""
    shape, stride, dtype, _alignment = requirements[name]
    return torch.empty_strided(shape, stride, dtype=dtype, device=device)


def _quantize_into_cudnn_symmetric_buffer(
    input_: torch.Tensor,
    data_buffer: torch.Tensor,
    scale_buffer: torch.Tensor,
    block_scaled_cls: type,
):
    """Quantize a plain tensor directly into cuDNN's symmetric MXFP8 storage."""
    from ...tensor.mxfp8_tensor import MXFP8Quantizer
    from ...tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

    token_count, hidden_size = input_.shape
    scale_rows = (token_count + 127) // 128 * 128
    quantizer = MXFP8Quantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    data = data_buffer[:token_count]
    padded_scale = scale_buffer[:scale_rows]
    output = MXFP8TensorStorage(
        data.view(torch.uint8),
        padded_scale.view(torch.uint8),
        None,
        None,
        tex.DType.kFloat8E4M3,
        quantizer,
        False,
        fake_dtype=input_.dtype,
    )
    tex.quantize(input_, quantizer, output, None)
    return block_scaled_cls(
        data=data,
        scale=padded_scale[:token_count, : hidden_size // MXFP8_BLOCK_SCALING_SIZE],
        format="mxfp8",
        logical_shape=tuple(input_.shape),
        axis=1,
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
        from cudnn.moe_ep import BlockScaledTensor
        self._block_scaled_cls = BlockScaledTensor
        self._resource = None

    def _prepare_training(self, device: torch.device) -> None:
        """Acquire and prepare the shared cuDNN runtime on the execution device."""
        if self._resource is not None:
            return
        resource = _MOE_EP_RESOURCE_MANAGER.get(
            self.dispatch.config,
            device,
            self.fc2.in_features,
            self.basic_ops[2].glu_interleave_size,
        )
        self._resource = resource

    def _make_native_training_weights(self):
        """Expose TE payloads with freshly swizzled caller-owned scale buffers."""
        from cudnn.moe_ep import (
            MoeEpNativeBackwardWeights,
            MoeEpNativeForwardWeights,
            MoeEpNativeWeight,
            MoeEpNativeWeightLayout,
        )

        def swizzle(weight: GroupedTensor):
            native = weight.copy()
            tex.grouped_swizzle_for_gemm(native, rowwise=True, columnwise=True)
            return native

        fc1 = swizzle(self.fc1.weight)
        fc2 = swizzle(self.fc2.weight)
        num_experts = self.fc1.num_groups

        def data(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
            return tensor.view(*shape).view(torch.float8_e4m3fn)

        def scale(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(torch.float8_e8m0fnu).reshape(num_experts, -1)

        fc1_rowwise = data(
            fc1.rowwise_data,
            (num_experts, self.fc1.out_features, self.fc1.in_features),
        ).permute(0, 2, 1)
        fc2_rowwise = data(
            fc2.rowwise_data,
            (num_experts, self.fc2.out_features, self.fc2.in_features),
        ).permute(0, 2, 1)
        fc1_columnwise = data(
            fc1.columnwise_data,
            (num_experts, self.fc1.out_features, self.fc1.in_features),
        )
        fc2_columnwise = data(
            fc2.columnwise_data,
            (num_experts, self.fc2.out_features, self.fc2.in_features),
        )
        forward = MoeEpNativeForwardWeights(
            fc1=MoeEpNativeWeight(
                fc1_rowwise,
                scale(fc1.scale_inv),
                MoeEpNativeWeightLayout.FORWARD_FC1_GATE_UP_INTERLEAVED_32_V1,
            ),
            fc2=MoeEpNativeWeight(
                fc2_rowwise,
                scale(fc2.scale_inv),
                MoeEpNativeWeightLayout.FORWARD_FC2_K_MAJOR_V1,
            ),
        )
        backward = MoeEpNativeBackwardWeights(
            w2_transpose=MoeEpNativeWeight(
                fc2_columnwise,
                scale(fc2.columnwise_scale_inv),
                MoeEpNativeWeightLayout.BACKWARD_W2_TRANSPOSE_V1,
            ),
            w1_transpose=MoeEpNativeWeight(
                fc1_columnwise,
                scale(fc1.columnwise_scale_inv),
                MoeEpNativeWeightLayout.BACKWARD_W1_TRANSPOSE_GATE_UP_INTERLEAVED_32_V1,
            ),
        )
        return forward, backward

    def _make_training_forward_outputs(self, device: torch.device):
        """Allocate forward output and backward-state destinations."""
        from cudnn.moe_ep import MoeEpTrainingForwardOutputs

        requirements = self._resource.requirements
        return MoeEpTrainingForwardOutputs(
            output=self._resource.symmetric_buffers["output"],
            fc1_preact=_allocate_training_buffer(requirements, "fc1_preact", device),
            fc1_a=_allocate_training_buffer(requirements, "fc1_a", device),
            fc1_sfa=_allocate_training_buffer(requirements, "fc1_sfa", device),
            valid_route_counts=_allocate_training_buffer(
                requirements,
                "valid_route_counts",
                device,
            ),
            expert_offsets=_allocate_training_buffer(requirements, "expert_offsets", device),
        )

    def _make_training_backward_outputs(self, device: torch.device):
        """Allocate gradient and grouped-WGrad operand destinations."""
        from cudnn.moe_ep import MoeEpTrainingBackwardOutputs

        requirements = self._resource.requirements
        return MoeEpTrainingBackwardOutputs(
            grad_activation=self._resource.symmetric_buffers["grad_activation"],
            dprob=self._resource.symmetric_buffers["dprob"],
            fc1_b=_allocate_training_buffer(requirements, "fc1_b", device),
            fc1_sfb=_allocate_training_buffer(requirements, "fc1_sfb", device),
            fc2_a=_allocate_training_buffer(requirements, "fc2_a", device),
            fc2_sfa=_allocate_training_buffer(requirements, "fc2_sfa", device),
            fc2_b=_allocate_training_buffer(requirements, "fc2_b", device),
            fc2_sfb=_allocate_training_buffer(requirements, "fc2_sfb", device),
        )

    def _make_training_wgrad_workspaces(self):
        """Allocate caller-owned descriptor workspaces for both FC gradients."""
        from cudnn import get_grouped_gemm_wgrad_workspace_size_sm100

        workspace_bytes = get_grouped_gemm_wgrad_workspace_size_sm100(
            self.fc1.num_groups,
            output_mode="dense",
            input_order="tensor2d",
        )
        return (
            torch.empty(workspace_bytes, dtype=torch.uint8, device=self.fc1.weight.device),
            torch.empty(workspace_bytes, dtype=torch.uint8, device=self.fc2.weight.device),
        )

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

        self._prepare_training(activation.device)
        activation = _quantize_into_cudnn_symmetric_buffer(
            activation,
            self._resource.symmetric_buffers["forward_input"],
            self._resource.symmetric_buffers["forward_input_scale"],
            self._block_scaled_cls,
        )
        forward_weights, backward_weights = self._make_native_training_weights()
        forward_out = self._make_training_forward_outputs(activation.device)
        output = self._resource.moe.training_forward(
            self._resource.lane,
            activation,
            topk_idx,
            topk_weights,
            weights=forward_weights,
            out=forward_out,
        )

        if any(ctx.requires_grad for ctx in basic_op_ctxs):
            basic_op_ctxs[0].save_for_backward(
                topk_idx,
                topk_weights,
                forward_out.fc1_preact,
                forward_out.fc1_a,
                forward_out.fc1_sfa,
                forward_out.valid_route_counts,
                forward_out.expert_offsets,
                backward_weights.w2_transpose.payload,
                backward_weights.w2_transpose.scale,
                backward_weights.w1_transpose.payload,
                backward_weights.w1_transpose.scale,
            )

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
        grad_output = _quantize_into_cudnn_symmetric_buffer(
            grad_output,
            self._resource.symmetric_buffers["backward_input"],
            self._resource.symmetric_buffers["backward_input_scale"],
            self._block_scaled_cls,
        )
        (
            topk_idx,
            topk_weights,
            fc1_preact,
            fc1_a,
            fc1_sfa,
            valid_route_counts,
            expert_offsets,
            w2_payload,
            w2_scale,
            w1_payload,
            w1_scale,
        ) = basic_op_ctxs[0].saved_tensors
        from cudnn.moe_ep import (
            MoeEpNativeBackwardWeights,
            MoeEpNativeWeight,
            MoeEpNativeWeightLayout,
        )

        backward_weights = MoeEpNativeBackwardWeights(
            w2_transpose=MoeEpNativeWeight(
                w2_payload,
                w2_scale,
                MoeEpNativeWeightLayout.BACKWARD_W2_TRANSPOSE_V1,
            ),
            w1_transpose=MoeEpNativeWeight(
                w1_payload,
                w1_scale,
                MoeEpNativeWeightLayout.BACKWARD_W1_TRANSPOSE_GATE_UP_INTERLEAVED_32_V1,
            ),
        )
        backward_out = self._make_training_backward_outputs(grad_output.device)
        grad_input, grad_topk_weights, wgrad_operands = self._resource.moe.training_backward(
            self._resource.lane,
            grad_output,
            topk_idx,
            topk_weights,
            weights=backward_weights,
            fc1_preact=fc1_preact,
            fc1_a=fc1_a,
            fc1_sfa=fc1_sfa,
            valid_route_counts=valid_route_counts,
            expert_offsets=expert_offsets,
            out=backward_out,
        )
        fc1_workspace, fc2_workspace = self._make_training_wgrad_workspaces()
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


__all__ = [
    "FusedMoeEp",
    "finalize_moe_ep_resources",
    "is_moe_fusion_supported",
]
