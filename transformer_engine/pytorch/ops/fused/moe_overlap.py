# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused MoE layer with EP dispatch and combine overlapped with the grouped GEMMs."""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Optional

import torch

import transformer_engine_torch as tex
from ....common.recipe import Format, MXFP8BlockScaling
from ...constants import MXFP8_BLOCK_SCALING_SIZE
from ...cpu_offload import (
    is_cpu_offload_enabled,
    mark_activation_offload,
    mark_not_offload,
    start_offload,
)
from ...ep import (
    EpConfig,
    _dispatch_window,
    _make_grouped_mxfp8,
    _moe_overlap_available,
    get_ep_drop_on_overflow,
    quantize_for_ep,
)
from ...distributed_weight import is_distributed_weight
from ...quantization import Recipe
from ...tensor.grouped_tensor import GroupedTensor
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from ...utils import clear_tensor_data, mark_grouped_tensor
from ..basic import (
    GroupedLinear,
    MoeCombine,
    MoeDispatch,
    ScaledClampedQGeGLU,
    ScaledSiTUGLU,
    ScaledSwiGLU,
)
from ..basic.dispatch import _validate_dispatch_input, _validate_routing_inputs
from ..fuser import register_forward_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import maybe_dequantize, validate_ep_buffer
from .grouped_mlp import (
    GroupedMLP_CuTeGEMMGLU,
    GroupedMLP_CuTeGEMMGLUBf16,
    _bf16_packed_weight,
    _bf16_wgrad_fn_factory,
    _compute_grad_params,
    _deterministic_algorithms_required,
    validate_grouped_mlp_dims,
)

# Recv rows are multiples of this tile, as are hidden_dim and the MXFP8 ffn_hidden_size.
_TILE = 256
# Each distinct recv-row count compiles a new GEMM, so eager rounds the recv rows up to a multiple
# of this to bound how many are compiled.
_EAGER_ROW_BUCKET = 1024
_SF_VEC = MXFP8_BLOCK_SCALING_SIZE
# Width of the alternating FC1 gate and up blocks the GEMMs take.
_GLU_BLOCK = 32


def _overlap_workspace(buffer, *, ffn, mxfp8, with_wgrad, rows):
    """Persistent uint8 scratch shared by a group's overlapped GEMM calls. Sized once per
    (ffn, mxfp8, with_wgrad, rows) and kept on the buffer, so a captured graph reuses the same
    buffer. Returns None in eager mode, where the recv rows are known only inside the call and the
    binding sizes and allocates the scratch there."""
    if buffer.eager:
        return None
    cache = buffer.overlap_workspaces
    key = (int(ffn), bool(mxfp8), bool(with_wgrad), int(rows))
    workspace = cache.get(key)
    if workspace is None:
        nbytes = tex.ep_overlap_workspace_size(
            buffer.hidden_dim, int(ffn), buffer.num_local_experts, int(rows),
            bool(with_wgrad), bool(mxfp8), buffer.top_k, buffer.alignment, buffer.topk_idx_dtype,
        )  # fmt: skip
        workspace = torch.empty(nbytes, dtype=torch.uint8, device=buffer.device)
        mark_not_offload(workspace)
        cache[key] = workspace
    return workspace


class FusedMoE(FusedOperation):
    """MoeDispatch, GroupedLinear, ScaledSwiGLU, GroupedLinear and MoeCombine as one op.

    Pipeline: dispatch -> FC1 -> SwiGLU -> FC2 -> combine, with the dispatch overlapped with FC1
    and the combine overlapped with FC2, forward and backward.

    Requirements:
      - Packed weights (``single_grouped_weight=True``) without bias: BF16 parameters, or MXFP8
        primary weights (``quantized_model_init``) under ``MXFP8BlockScaling``.
      - Contiguous forward inputs.
      - EP within one NVLink domain.
      - ``EpConfig`` ``alignment`` of 256, 128 or (without a recipe) 64, with the receive capacity
        a multiple of 256. Below 256 the unfused ops run where FusedMoE cannot, also skipping the
        grouped MLP fusion, and run slower.
      - Recipe of none or ``MXFP8BlockScaling``.

    Notes:
      - The GEMMs take a uint8 workspace for their flags and state, sized once per buffer and
        reused across calls; every FusedMoE sharing it must run on one CUDA stream.
      - With a fixed receive capacity the op can be captured in a CUDA graph after a warmup call;
        with eager receive sizing (no ``recv_capacity_per_rank``) the host is synchronized and it
        cannot be captured.
      - Without a recipe the routing weights are applied in the activation; under
        ``MXFP8BlockScaling`` the GEMMs and the EP transport run in MXFP8 (the ``EpConfig`` sets
        MXFP8 ``dispatch_fwd_quant_recipe`` and ``combine_bwd_quant_recipe``) and the routing
        weights are applied to the second GEMM's output. Either way their gradient goes to the
        dispatch's ``topk_weights``.
      - The weights are prepared for the GEMMs (quantized to MXFP8, or their FC1 rows reordered) at
        each forward, unless the GroupedLinear's ``op_kwargs`` set ``is_first_microbatch``: True
        keeps the prepared weights, False reuses them for the next microbatches of the same
        weights.
      - The GEMMs take the FC1 gate and up rows in alternating blocks of 32
        (``glu_interleave_size=32``). With ``glu_interleave_size=None`` the FC1 weight rows are
        reordered when the weights are prepared and the FC1 weight gradient is returned in the
        parameter's own layout.
    """

    @classmethod
    def is_supported(cls) -> bool:
        """Whether this fused operation can run on the current system."""
        if not _moe_overlap_available() or _gemm_interface_reason() is not None:
            return False
        if torch.cuda.get_device_capability() != (10, 7):
            return GroupedMLP_CuTeGEMMGLUBf16.is_supported()
        # SM 10.7 has no BF16 cuDNN GLU kernels; of those kernels this op uses only the wgrad.
        try:
            kernel = GroupedMLP_CuTeGEMMGLUBf16.grouped_gemm_wgrad_kernel()
        except ImportError:
            return False
        return getattr(kernel, "supports_bf16_canonical_layouts", False)

    def __init__(
        self,
        *,
        dispatch: MoeDispatch,
        fc1: GroupedLinear,
        activation: ScaledSwiGLU,
        fc2: GroupedLinear,
        combine: MoeCombine,
        recipe: Optional[Recipe] = None,
        _validated: bool = False,
    ) -> None:
        super().__init__((dispatch, fc1, activation, fc2, combine))
        if not _validated:
            reason = _unsupported_reason(dispatch, fc1, activation, fc2, combine, recipe=recipe)
            if reason is not None:
                raise RuntimeError(f"FusedMoE is not supported: {reason}.")
        self._mxfp8 = recipe is not None
        # The GEMMs take FC1 gate and up rows in alternating blocks of _GLU_BLOCK.
        self._reorder_glu = activation.glu_interleave_size is None
        # MXFP8 weight operands kept across microbatches, per GroupedLinear index.
        self._weight_cache: dict[int, _PreparedWeight] = {}
        # The dispatch and combine share one EpBuffer (see _unsupported_reason); FusedMoE reads it
        # from self.basic_ops[0].buffer at each forward.

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer,
        next_op_input_quantizer,
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Sequence[Sequence[torch.Tensor]]]:
        del prev_op_grad_output_quantizer, next_op_input_quantizer
        _, fc1_op, _, fc2_op, _ = self.basic_ops
        topk_idx, topk_weights = basic_op_extra_inputs[0]
        # Splits and scales come from the dispatch inside this op.
        for idx in (1, 2, 3):
            if any(t is not None for t in basic_op_extra_inputs[idx]):
                raise ValueError(
                    "FusedMoE takes the splits and routing scales from its dispatch; "
                    "do not pass them as extra inputs."
                )

        buffer = validate_ep_buffer("FusedMoE", self.basic_ops[0].config, self.basic_ops[0].buffer)
        input_shape = _validate_dispatch_input(input_, buffer)
        _validate_routing_inputs(topk_idx, topk_weights, device=buffer.device)
        if topk_idx.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}.")
        if torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") != torch.bfloat16:
            raise RuntimeError("FusedMoE computes in BF16; keep the autocast dtype BF16.")
        if not buffer.eager and buffer.recv_capacity_per_rank % _TILE != 0:
            raise ValueError(
                f"FusedMoE needs the recv rows ({buffer.recv_capacity_per_rank}) to be a"
                f" multiple of {_TILE}."
            )
        buffer.num_local_tokens = input_shape[0]
        buffer.topk_idx_dtype = topk_idx.dtype

        if self._mxfp8:
            return self._forward_mxfp8(
                basic_op_ctxs, buffer, input_, topk_idx, topk_weights, basic_op_kwargs
            )
        device = buffer.device
        num_groups = fc1_op.num_groups
        hidden = buffer.hidden_dim
        ffn = fc2_op.in_features
        w1 = self._weight_operands(1, False, basic_op_kwargs[1].get("is_first_microbatch"))
        w2 = _bf16_weight(fc2_op, num_groups)
        tokens = _require_contiguous(input_, "input")
        topk_idx = _require_contiguous(topk_idx, "topk_idx")
        topk_weights = _require_contiguous(topk_weights, "topk_weights")

        with_wgrad = any(ctx.requires_grad for ctx in basic_op_ctxs) and (
            fc1_op.weight.requires_grad or fc2_op.weight.requires_grad
        )
        tokens_per_expert = buffer.overlap_tokens_per_expert
        ws = None
        if buffer.eager:
            # The binding runs the prepare, reads the recv total on the host and allocates the
            # recv rows in whole buckets, so the number of compiled GEMMs stays bounded.
            recv_tokens, recv_weights, gemm_out, act_out, expert_token_offsets = (
                tex.ep_dispatch_gemm_fwd(
                    buffer.handle_mem, topk_idx, tokens, topk_weights, w1, None, None,
                    tokens_per_expert, None, None, buffer.top_k, buffer.alignment,
                    buffer.overlap_total_recv_tokens, _EAGER_ROW_BUCKET,
                )
            )  # fmt: skip
            rows = recv_tokens.shape[0]
        else:
            rows = buffer.recv_capacity_per_rank
            ws = _overlap_workspace(buffer, ffn=ffn, mxfp8=False, with_wgrad=with_wgrad, rows=rows)
            recv_tokens = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
            recv_weights = torch.empty(rows, dtype=torch.float32, device=device)
            gemm_out = torch.empty(rows, 2 * ffn, dtype=torch.bfloat16, device=device)
            act_out = torch.empty(rows, ffn, dtype=torch.bfloat16, device=device)
            # The dispatch writes the expert layout the GEMMs read.
            expert_token_offsets = tex.ep_dispatch_gemm_fwd(
                buffer.handle_mem, topk_idx, tokens, topk_weights, w1, recv_tokens, recv_weights,
                tokens_per_expert, gemm_out, act_out, buffer.top_k, buffer.alignment,
                buffer.overlap_total_recv_tokens, workspace=ws,
            )[4]  # fmt: skip
        expert_out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
        result = torch.empty(buffer.num_local_tokens, hidden, dtype=torch.bfloat16, device=device)
        tex.ep_combine_gemm_fwd(
            buffer.handle_mem, act_out, expert_token_offsets, w2, expert_out, result,
            buffer.top_k, buffer.alignment, buffer.topk_idx_dtype, workspace=ws,
        )  # fmt: skip
        recv_total = _recv_total_from_offsets(buffer, expert_token_offsets)
        self.basic_ops[0].update_recv_tokens_peak(buffer, recv_total)

        # The BF16 SwiGLU backward GEMM cannot write the SwiGLU output, so the FC2 wgrad reads it.
        self._save_for_backward(
            basic_op_ctxs, buffer, recv_total, (recv_tokens, gemm_out, act_out, recv_weights),
            (w1,) if self._reorder_glu else (), topk_weights,
        )  # fmt: skip
        return result, [(tokens_per_expert, recv_weights), (), (), (), ()]

    def _save_for_backward(
        self, basic_op_ctxs, buffer, recv_total, row_tensors, other_tensors, topk_weights
    ):
        """Save the tensors the backward reads, when any output needs a gradient."""
        _, fc1_op, _, fc2_op, _ = self.basic_ops
        fc1_ctx, fc2_ctx = basic_op_ctxs[1], basic_op_ctxs[3]
        if any(ctx.requires_grad for ctx in basic_op_ctxs):
            fc1_ctx.weight_requires_grad = fc1_op.weight.requires_grad
            fc2_ctx.weight_requires_grad = fc2_op.weight.requires_grad
            fc1_ctx.dtype = fc2_ctx.dtype = torch.bfloat16
            # Paged stashing keeps only the received rows of each row-sized tensor. The recv total
            # counts them, except in eager mode (a host total) and when rows past the capacity drop.
            mark_grouped_tensor(*row_tensors)
            if buffer.eager or get_ep_drop_on_overflow():
                num_tokens = buffer.overlap_tokens_per_expert.sum(
                    0, keepdim=True, dtype=torch.int32
                )
            else:
                num_tokens = recv_total
            for tensor in row_tensors:
                tensor.grouped_tensor_num_tokens = num_tokens
            # CPU offload takes the activations, as the unfused ops do, not the routing weights.
            # The weight copies stay on the GPU, as in TE modules, since they may be kept for
            # the next microbatches.
            if is_cpu_offload_enabled():
                start_offload(*row_tensors[:3])
                mark_activation_offload(*row_tensors[:3])
                mark_not_offload(*row_tensors[3:], *other_tensors)
            fc1_ctx.save_for_backward(*row_tensors, *other_tensors)
            fc1_ctx.buffer = buffer
            fc1_ctx.topk_weights_shape = topk_weights.shape

    def _forward_mxfp8(
        self, basic_op_ctxs, buffer, input_, topk_idx, topk_weights, basic_op_kwargs
    ):
        """Forward with MXFP8 GEMMs and EP transport."""
        fc1_op, fc2_op = self.basic_ops[1], self.basic_ops[3]
        device = buffer.device
        hidden = buffer.hidden_dim
        ffn = fc2_op.in_features
        backward = any(ctx.requires_grad for ctx in basic_op_ctxs)
        with_wgrad = backward and (fc1_op.weight.requires_grad or fc2_op.weight.requires_grad)
        w1, w2 = (
            self._weight_operands(idx, backward, basic_op_kwargs[idx].get("is_first_microbatch"))
            for idx in (1, 3)
        )
        tokens, tokens_scale, symm_input = _quantize_for_dispatch(
            _require_contiguous(input_, "input"), device
        )
        topk_idx = _require_contiguous(topk_idx, "topk_idx")
        topk_weights = _require_contiguous(topk_weights, "topk_weights")
        tokens_per_expert = buffer.overlap_tokens_per_expert
        ws = None
        if buffer.eager:
            (
                recv_tokens, recv_weights, gemm_out, act_out, expert_token_offsets, recv_scale,
                act_scale,
            ) = (
                tex.ep_dispatch_gemm_fwd(
                    buffer.handle_mem, topk_idx, tokens, topk_weights, w1.data, None, None,
                    tokens_per_expert, None, None, buffer.top_k, buffer.alignment,
                    buffer.overlap_total_recv_tokens, _EAGER_ROW_BUCKET,
                    tokens_scale_inv=tokens_scale, weight_scale_inv=w1.scale_inv, symm_input=symm_input,
                )
            )  # fmt: skip
            rows = recv_tokens.shape[0]
        else:
            rows = buffer.recv_capacity_per_rank
            ws = _overlap_workspace(buffer, ffn=ffn, mxfp8=True, with_wgrad=with_wgrad, rows=rows)
            fp8 = torch.float8_e4m3fn
            recv_tokens = torch.empty(rows, hidden, dtype=fp8, device=device)
            recv_scale = torch.empty(rows, hidden // _SF_VEC, dtype=torch.uint8, device=device)
            recv_weights = torch.empty(rows, dtype=torch.float32, device=device)
            gemm_out = torch.empty(rows, 2 * ffn, dtype=torch.bfloat16, device=device)
            act_out = torch.empty(rows, ffn, dtype=fp8, device=device)
            act_scale = torch.empty(rows * ffn // _SF_VEC, dtype=torch.uint8, device=device)
            expert_token_offsets = tex.ep_dispatch_gemm_fwd(
                buffer.handle_mem, topk_idx, tokens, topk_weights, w1.data, recv_tokens,
                recv_weights, tokens_per_expert, gemm_out, act_out, buffer.top_k,
                buffer.alignment, buffer.overlap_total_recv_tokens, tokens_scale_inv=tokens_scale,
                recv_scale_inv=recv_scale, weight_scale_inv=w1.scale_inv, act_out_scale_inv=act_scale,
                symm_input=symm_input, workspace=ws,
            )[4]  # fmt: skip
        expert_out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
        result = torch.empty(buffer.num_local_tokens, hidden, dtype=torch.bfloat16, device=device)
        # The routing weights scale the second GEMM's output rows.
        tex.ep_combine_gemm_fwd(
            buffer.handle_mem, act_out, expert_token_offsets, w2.data, expert_out, result,
            buffer.top_k, buffer.alignment, buffer.topk_idx_dtype, recv_topk_weights=recv_weights,
            gemm_in_scale_inv=act_scale, weight_scale_inv=w2.scale_inv, workspace=ws,
        )  # fmt: skip
        recv_total = _recv_total_from_offsets(buffer, expert_token_offsets)
        self.basic_ops[0].update_recv_tokens_peak(buffer, recv_total)
        # The FC2 dgrad recomputes the SwiGLU output for the FC2 wgrad, so it is not saved.
        self._save_for_backward(
            basic_op_ctxs, buffer, recv_total, (recv_tokens, recv_scale, gemm_out, recv_weights),
            (w1.columnwise_data, w1.columnwise_scale_inv, w2.columnwise_data, w2.columnwise_scale_inv)
            if backward else (), topk_weights,
        )  # fmt: skip
        return result, [(tokens_per_expert, recv_weights), (), (), (), ()]

    def _weight_operands(self, idx, columnwise, is_first_microbatch):
        """GEMM operands of the weight of the GroupedLinear at ``idx``: its ``_MxFp8Weight``, or
        the BF16 weight. As in TE modules, ``is_first_microbatch`` True prepares and keeps them,
        False reuses the kept ones unless the weight changed in place, and None prepares them
        without keeping them."""
        fc_op = self.basic_ops[idx]
        reorder = self._reorder_glu and idx == 1
        if not (self._mxfp8 or reorder):
            return _bf16_weight(fc_op, fc_op.num_groups)
        key = None if is_first_microbatch is None else _weight_key(fc_op.weight)
        kept = self._weight_cache.get(idx)
        if is_first_microbatch is False and key is not None and kept is not None:
            operands = kept.operands
            has_columnwise = not isinstance(operands, _MxFp8Weight) or (
                operands.columnwise_data is not None
            )
            if kept.key == key and (has_columnwise or not columnwise):
                return operands
        if self._mxfp8:
            operands = _mxfp8_weight(fc_op, fc_op.num_groups, columnwise, reorder_glu=reorder)
        else:
            operands = _interleave_glu_rows(_bf16_weight(fc_op, fc_op.num_groups))
        if key is None:
            self._weight_cache.pop(idx, None)
        else:
            self._weight_cache[idx] = _PreparedWeight(key, operands)
        return operands

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[Optional[torch.Tensor], ...]],
        **unused,  # pylint: disable=unused-argument
    ):
        # A later op that also reads the dispatch's routing weights delivers their gradient here;
        # fold it into the routing-weight gradient the dispatch backward scatters to topk_weights.
        grad_recv_weights_extra = basic_op_grad_extra_outputs[0][1]
        if self._mxfp8:
            return self._backward_mxfp8(basic_op_ctxs, grad_output, grad_recv_weights_extra)
        _, fc1_op, _, fc2_op, _ = self.basic_ops
        fc1_ctx, fc2_ctx = basic_op_ctxs[1], basic_op_ctxs[3]
        recv_tokens, gemm_out, act_out, recv_weights, *w1 = fc1_ctx.saved_tensors
        buffer = fc1_ctx.buffer
        device = buffer.device
        num_groups = fc1_op.num_groups
        hidden = buffer.hidden_dim
        ffn = fc2_op.in_features
        rows = gemm_out.shape[0]
        with_wgrad = fc1_ctx.weight_requires_grad or fc2_ctx.weight_requires_grad
        ws = _overlap_workspace(buffer, ffn=ffn, mxfp8=False, with_wgrad=with_wgrad, rows=rows)
        tokens_per_expert = buffer.overlap_tokens_per_expert
        w1 = w1[0] if w1 else _bf16_weight(fc1_op, num_groups)
        w2 = _bf16_weight(fc2_op, num_groups)
        grad_result = _bf16_grad_output(grad_output)
        num_tokens = grad_result.shape[0]

        wgrad_kernel_fn = GroupedMLP_CuTeGEMMGLUBf16.grouped_gemm_wgrad_kernel()
        fc2_shape = (fc2_op.out_features, fc2_op.in_features)
        fc1_shape = (fc1_op.out_features, fc1_op.in_features)

        def grad_params(fc_op, ctx, shape, grouped_x, grouped_dy, label, wgrad_args=None):
            # With wgrad_args, the backward call computes the weight gradient and reads no offsets.
            offsets = tokens_per_expert.cumsum(0, dtype=torch.int32) if wgrad_args is None else None
            factory = (
                _bf16_wgrad_fn_factory(
                    weight_shape=shape, offsets=offsets, wgrad_kernel_fn=wgrad_kernel_fn,
                    single_grouped_weight=True, num_groups=num_groups,
                )
                if wgrad_args is None
                else wgrad_args.gemm_fn_factory(num_groups, shape)
            )  # fmt: skip
            grads = _compute_grad_params(
                fc_op=fc_op, ctx=ctx, num_groups=num_groups, weight_shape=shape,
                grouped_x=grouped_x, grouped_dy=grouped_dy, dtype=torch.bfloat16, device=device,
                bias_grads=None, bias_grad_packed=None, label=label,
                cudnn_wgrad_kernel_fn=wgrad_kernel_fn, use_nvfp4=False,
                data_dtype=torch.bfloat16, scale_view_dtype=None, sf_vec_size=None,
                offsets=offsets, use_dense_single_group=False, wgrad_gemm_fn_factory=factory,
            )  # fmt: skip
            if wgrad_args is not None and ctx.weight_requires_grad:
                wgrad_args.require_recorded()
            return grads

        grad_expert_out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
        grad_gemm = torch.empty(rows, 2 * ffn, dtype=torch.bfloat16, device=device)
        grad_recv_weights = torch.empty(rows, dtype=torch.float32, device=device)
        fc2_grads = None
        fc2_wgrad = _CallWgrad()
        if fc2_ctx.weight_requires_grad:
            fc2_grads = grad_params(
                fc2_op, fc2_ctx, fc2_shape, act_out, grad_expert_out, "FC2", fc2_wgrad
            )
        tex.ep_combine_gemm_bwd(
            buffer.handle_mem, grad_result, gemm_out, recv_weights, tokens_per_expert, w2,
            grad_expert_out, grad_gemm, grad_recv_weights, buffer.top_k, buffer.alignment,
            buffer.topk_idx_dtype, workspace=ws, **fc2_wgrad,
        )  # fmt: skip
        if fc2_grads is None:
            fc2_grads = grad_params(fc2_op, fc2_ctx, fc2_shape, act_out, grad_expert_out, "FC2")

        # Neither the input nor the routing weights need a gradient: no dispatch backward.
        grad_tokens = grad_topk = fc1_grads = None
        if basic_op_ctxs[0].requires_grad:
            if grad_recv_weights_extra is not None:
                grad_recv_weights.add_(grad_recv_weights_extra.to(torch.float32))
            fc1_wgrad = _CallWgrad()
            # The weight gradient of reordered FC1 rows is computed after the reorder back.
            if not self._reorder_glu and fc1_ctx.weight_requires_grad:
                fc1_grads = grad_params(
                    fc1_op, fc1_ctx, fc1_shape, recv_tokens, grad_gemm, "FC1", fc1_wgrad
                )
            grad_recv = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
            grad_tokens = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=device)
            grad_topk = torch.empty(fc1_ctx.topk_weights_shape, dtype=torch.float32, device=device)
            tex.ep_dispatch_gemm_bwd(
                buffer.handle_mem, grad_gemm, grad_recv_weights, tokens_per_expert, w1,
                grad_recv, grad_tokens, grad_topk, buffer.top_k, buffer.alignment,
                buffer.topk_idx_dtype, workspace=ws, **fc1_wgrad,
            )  # fmt: skip

        if fc1_grads is None:
            if self._reorder_glu:
                grad_gemm = _deinterleave_glu_columns(grad_gemm)
            fc1_grads = grad_params(fc1_op, fc1_ctx, fc1_shape, recv_tokens, grad_gemm, "FC1")

        clear_tensor_data(recv_tokens, act_out)
        return self._finish_backward(grad_tokens, grad_topk, fc1_grads, fc2_grads)

    def _finish_backward(self, grad_tokens, grad_topk, fc1_grads, fc2_grads):
        """Return the gradients; the routing-weight gradient is part of the dispatch's
        ``topk_weights`` gradient."""
        return (
            grad_tokens,
            [(), fc1_grads, (), fc2_grads, ()],
            [(None, grad_topk), (None,), (None,), (None,), (None,)],
        )

    def _backward_mxfp8(self, basic_op_ctxs, grad_output, grad_recv_weights_extra=None):
        """Backward with MXFP8 GEMMs and EP transport."""
        _, fc1_op, _, fc2_op, _ = self.basic_ops
        fc1_ctx, fc2_ctx = basic_op_ctxs[1], basic_op_ctxs[3]
        (
            recv_tokens,
            recv_scale,
            gemm_out,
            recv_weights,
            w1_col,
            w1_col_scale,
            w2_col,
            w2_col_scale,
        ) = fc1_ctx.saved_tensors
        buffer = fc1_ctx.buffer
        device = buffer.device
        num_groups = fc1_op.num_groups
        hidden = buffer.hidden_dim
        ffn = fc2_op.in_features
        rows = gemm_out.shape[0]
        with_wgrad = fc1_ctx.weight_requires_grad or fc2_ctx.weight_requires_grad
        ws = _overlap_workspace(buffer, ffn=ffn, mxfp8=True, with_wgrad=with_wgrad, rows=rows)
        tokens_per_expert = buffer.overlap_tokens_per_expert
        # GroupedTensor group sizes are int64.
        first_dims = tokens_per_expert.long()
        grad_result = _bf16_grad_output(grad_output)
        num_tokens = grad_result.shape[0]
        grad_result, grad_result_scale, symm_input = _quantize_for_dispatch(grad_result, device)

        def fp8_rows(cols):
            data = torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device=device)
            return data, torch.empty(rows * cols // _SF_VEC, dtype=torch.uint8, device=device)

        grad_expert_out, grad_expert_out_scale = fp8_rows(hidden)
        grad_expert_out_scale = grad_expert_out_scale.view(rows, hidden // _SF_VEC)
        # The call also writes the received rows columnwise for the FC2 weight gradient.
        grad_expert_out_col, grad_expert_out_col_scale = fp8_rows(hidden)
        grad_gemm, grad_gemm_scale = fp8_rows(2 * ffn)
        grad_gemm_col, grad_gemm_col_scale = fp8_rows(2 * ffn)
        act_col, act_col_scale = fp8_rows(ffn)
        grad_recv_weights = torch.empty(rows, dtype=torch.float32, device=device)

        wgrad_kernel_fn = GroupedMLP_CuTeGEMMGLU.grouped_gemm_wgrad_kernel()

        def grouped(cols, data, scale, col_data, col_scale, in_call):
            # Expert slabs of a row-sized MXFP8 tensor with GEMM-swizzled scales. A weight
            # gradient in a backward call reads no tensor offsets.
            return GroupedTensor(
                shape=(rows, cols), dtype=torch.bfloat16, num_tensors=num_groups,
                quantizer=_ep_quantizer(), data=data, columnwise_data=col_data.view(-1),
                scale_inv=scale, columnwise_scale_inv=col_scale, first_dims=first_dims,
                tensor_offsets=None if in_call else tex.splits_to_offsets(first_dims, cols),
                with_gemm_swizzled_scales=True,
            )  # fmt: skip

        def grad_params(fc_op, ctx, grouped_x, grouped_dy, label, wgrad_args=None):
            # With wgrad_args, the backward call computes the weight gradient.
            shape = (fc_op.out_features, fc_op.in_features)
            grads = _compute_grad_params(
                fc_op=fc_op, ctx=ctx, num_groups=num_groups, weight_shape=shape,
                grouped_x=grouped_x, grouped_dy=grouped_dy, dtype=torch.bfloat16, device=device,
                bias_grads=None, bias_grad_packed=None, label=label,
                cudnn_wgrad_kernel_fn=wgrad_kernel_fn, use_nvfp4=False,
                data_dtype=torch.float8_e4m3fn, scale_view_dtype=torch.float8_e8m0fnu,
                sf_vec_size=_SF_VEC, use_dense_single_group=False,
                offsets=(
                    tokens_per_expert.cumsum(0, dtype=torch.int32) if wgrad_args is None else None
                ),
                wgrad_gemm_fn_factory=(
                    None
                    if wgrad_args is None
                    else wgrad_args.gemm_fn_factory(num_groups, shape)
                ),
            )  # fmt: skip
            if wgrad_args is not None and ctx.weight_requires_grad:
                wgrad_args.require_recorded()
            return grads

        def fc1_grad_params(wgrad_args=None):
            in_call = wgrad_args is not None
            if in_call:
                # The call writes the columnwise received rows from their rowwise copy.
                x_col, x_col_scale = fp8_rows(hidden)
                grouped_x = grouped(hidden, None, None, x_col, x_col_scale, True)
                wgrad_args.update(
                    wgrad_input_rowwise=recv_tokens, wgrad_input_rowwise_scale_inv=recv_scale
                )
            else:
                grouped_x = _columnwise_rows(recv_tokens, recv_scale, first_dims)
            grad_gemm_cols = (
                grouped(2 * ffn, None, None, _deinterleave_glu_columns(grad_gemm_col),
                        _deinterleave_glu_col_scales(grad_gemm_col_scale, tokens_per_expert,
                                                     2 * ffn), in_call)
                if self._reorder_glu
                else grouped(2 * ffn, grad_gemm.view(-1), grad_gemm_scale, grad_gemm_col,
                             grad_gemm_col_scale, in_call)
            )  # fmt: skip
            return grad_params(
                fc1_op, fc1_ctx, grouped_x, grad_gemm_cols, "FC1", wgrad_args
            )  # fmt: skip

        # The call computes the FC2 weight gradient too.
        fc2_wgrad = _CallWgrad()
        fc2_grads = grad_params(
            fc2_op, fc2_ctx, grouped(ffn, None, None, act_col, act_col_scale, True),
            grouped(hidden, None, None, grad_expert_out_col, grad_expert_out_col_scale, True),
            "FC2", fc2_wgrad,
        )  # fmt: skip
        tex.ep_combine_gemm_bwd(
            buffer.handle_mem, grad_result, gemm_out,
            recv_weights, tokens_per_expert, w2_col, grad_expert_out, grad_gemm,
            grad_recv_weights, buffer.top_k, buffer.alignment, buffer.topk_idx_dtype,
            grad_result_scale_inv=grad_result_scale,
            grad_expert_out_scale_inv=grad_expert_out_scale, weight_scale_inv=w2_col_scale,
            grad_gemm_out_scale_inv=grad_gemm_scale, grad_gemm_out_columnwise=grad_gemm_col,
            grad_gemm_out_columnwise_scale_inv=grad_gemm_col_scale, act_out=act_col,
            act_out_scale_inv=act_col_scale, grad_expert_out_columnwise=grad_expert_out_col,
            grad_expert_out_columnwise_scale_inv=grad_expert_out_col_scale, **fc2_wgrad,
            symm_input=symm_input, workspace=ws,
        )  # fmt: skip

        grad_tokens = grad_topk = fc1_grads = None
        if basic_op_ctxs[0].requires_grad:
            if grad_recv_weights_extra is not None:
                grad_recv_weights.add_(grad_recv_weights_extra.to(torch.float32))
            fc1_wgrad = _CallWgrad()
            # The weight gradient of reordered FC1 rows is computed after the reorder back.
            if not self._reorder_glu and fc1_ctx.weight_requires_grad:
                fc1_wgrad.update(
                    grad_gemm_out_columnwise=grad_gemm_col,
                    grad_gemm_out_columnwise_scale_inv=grad_gemm_col_scale,
                )
                fc1_grads = fc1_grad_params(fc1_wgrad)
            grad_recv = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
            grad_tokens = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=device)
            grad_topk = torch.empty(fc1_ctx.topk_weights_shape, dtype=torch.float32, device=device)
            tex.ep_dispatch_gemm_bwd(
                buffer.handle_mem, grad_gemm, grad_recv_weights, tokens_per_expert, w1_col,
                grad_recv, grad_tokens, grad_topk, buffer.top_k, buffer.alignment,
                buffer.topk_idx_dtype, grad_gemm_out_scale_inv=grad_gemm_scale,
                weight_scale_inv=w1_col_scale, workspace=ws, **fc1_wgrad,
            )  # fmt: skip

        if fc1_grads is None:
            fc1_grads = fc1_grad_params()

        clear_tensor_data(recv_tokens, recv_scale)
        return self._finish_backward(grad_tokens, grad_topk, fc1_grads, fc2_grads)


def _recv_total_from_offsets(buffer, expert_token_offsets) -> torch.Tensor:
    """The int32 [1] recv total: in capacity mode the last element of the expert token offsets.
    Eager and overflow-drop modes get it from the routing prepare instead."""
    if buffer.eager or get_ep_drop_on_overflow():
        return buffer.overlap_total_recv_tokens
    return expert_token_offsets[-1:]


def _require_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Forward inputs must already be contiguous; making them contiguous here would hide a copy."""
    if not tensor.is_contiguous():
        raise ValueError(f"FusedMoE requires a contiguous {name}, got strides {tensor.stride()}.")
    return tensor


def _bf16_grad_output(grad_output) -> torch.Tensor:
    """Contiguous BF16 gradient of the combine output."""
    if isinstance(grad_output, GroupedTensor):
        raise RuntimeError("FusedMoE does not accept a quantized grad output.")
    return maybe_dequantize(grad_output, torch.bfloat16).contiguous()


class _CallWgrad(dict):
    """Keyword arguments of a weight gradient computed inside a backward call, passed as
    ``**wgrad``. Its wgrad factory only records the operands; the call runs the kernel later, so
    distributed weights, which FusedMoE rejects, do not fit."""

    def gemm_fn_factory(self, num_groups: int, weight_shape) -> Callable:
        """Wgrad factory that records the operands of a weight of shape ``weight_shape``."""

        def factory(accumulate: bool) -> Callable:
            def gemm_fn(grouped_x, grouped_dy, wgrad_output):
                del grouped_dy
                if isinstance(grouped_x, GroupedTensor):
                    self.update(
                        wgrad_input=grouped_x.columnwise_data.view(torch.float8_e4m3fn).view(
                            -1, weight_shape[1]
                        ),
                        wgrad_input_scale_inv=grouped_x.columnwise_scale_inv,
                    )
                else:
                    self.update(wgrad_input=grouped_x)
                self.update(
                    weight_grad=wgrad_output.rowwise_data.view(num_groups, *weight_shape),
                    accumulate_weight_grad=accumulate,
                )

            return gemm_fn

        return factory

    def require_recorded(self) -> None:
        """Raise unless the GEMM function ran: the call would otherwise skip the gradient."""
        if "weight_grad" not in self:
            raise RuntimeError("FusedMoE: the weight-gradient GEMM function was not called.")


class _MxFp8Weight(NamedTuple):
    """MXFP8 operands of a weight: data and scales, rowwise and, when prepared, columnwise."""

    data: torch.Tensor
    scale_inv: torch.Tensor
    columnwise_data: Optional[torch.Tensor] = None
    columnwise_scale_inv: Optional[torch.Tensor] = None


class _PreparedWeight(NamedTuple):
    """Operands kept for ``is_first_microbatch=False`` and the ``_weight_key`` they were made at."""

    key: tuple
    operands: torch.Tensor | _MxFp8Weight


def _gemm_interface_reason(mxfp8: bool = False) -> Optional[str]:
    """Why the EP grouped GEMM kernel tree does not match Transformer Engine, or ``None``."""
    try:
        from ....common.CuTeDSL.ep_gemm._loader import (  # pylint: disable=import-outside-toplevel
            gemm_interface_reason,
        )
    except ImportError as err:
        return f"the CuTeDSL EP grouped GEMMs cannot be imported: {err}"
    return gemm_interface_reason(mxfp8)


def _weight_key(weight) -> Optional[tuple]:
    """Identity and in-place update counters of the tensors that store ``weight``, or ``None``
    when one has no counter (inference tensors)."""
    try:
        key = [weight._version]
        for name in ("rowwise_data", "columnwise_data", "scale_inv", "columnwise_scale_inv"):
            tensor = getattr(weight, name, None)
            if tensor is not None:
                key += [tensor.data_ptr(), tensor._version]
    except RuntimeError:
        return None
    return tuple(key)


def _ep_quantizer() -> MXFP8Quantizer:
    """Rowwise E4M3 quantizer of the MXFP8 EP transport."""
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    quantizer.internal = True
    return quantizer


def _quantize_for_dispatch(input_, device):
    """MXFP8 data and compact scales of a dispatch input, and whether the dispatch reads them in
    place: they are in the EP dispatch window whenever it exists, alike on every rank."""
    window = _dispatch_window(device)
    quantized, scale = quantize_for_ep(input_, _ep_quantizer(), window)
    return quantized._rowwise_data.view(torch.float8_e4m3fn), scale, window is not None


def _columnwise_rows(data, scale, tokens_per_expert):
    """Expert slabs of received MXFP8 rows with compact scales, given columnwise data for the
    weight gradient."""
    grouped = _make_grouped_mxfp8(
        data, scale, tokens_per_expert, tex.DType.kFloat8E4M3, torch.bfloat16
    )
    quantizer = _ep_quantizer()
    quantizer.set_usage(rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = True
    tex.group_requantize_inplace(
        grouped, quantizer, tokens_per_expert.numel(), tokens_per_expert, tex.DType.kBFloat16,
        tensor_offsets=grouped.tensor_offsets,
    )  # fmt: skip
    return grouped


def _mxfp8_weight(
    fc_op: GroupedLinear, num_groups: int, columnwise: bool, reorder_glu: bool = False
) -> _MxFp8Weight:
    """GEMM operands of the weight quantized to MXFP8, with the columnwise ones when asked; data
    is [num_groups, out_features, in_features].
    BF16 weights are quantized with GEMM-swizzled scales; the scales of MXFP8 weight parameters
    are swizzled into a copy. Each direction is swizzled along its own scaling axis.
    ``reorder_glu`` reorders FC1 rows as ``_interleave_glu_rows`` does, data and scales alike."""
    weight = fc_op.weight
    shape = (num_groups, fc_op.out_features, fc_op.in_features)
    if weight.quantizer is not None:
        # Only add usages, never drop: later weight updates keep what this op needs.
        weight.quantizer.set_usage(rowwise=True, columnwise=True if columnwise else None)
        if columnwise and weight.columnwise_data is None:
            raise RuntimeError("FusedMoE needs columnwise data in the MXFP8 weight parameters.")
        weight = weight.copy()
        if reorder_glu:
            # Rowwise scales follow the rows; a columnwise scale row covers one block of rows.
            rows, cols = shape[1:]
            weight.rowwise_data = _interleave_glu_rows(weight.rowwise_data.view(shape)).view(-1)
            weight.scale_inv = _interleave_glu_rows(
                weight.scale_inv.view(num_groups, rows, -1)
            ).view(-1)
            if weight.columnwise_data is not None:
                weight.columnwise_data = _interleave_glu_rows(
                    weight.columnwise_data.view(shape)
                ).view(-1)
                weight.columnwise_scale_inv = _interleave_glu_rows(
                    weight.columnwise_scale_inv.view(num_groups, rows // _SF_VEC, cols),
                    block=_GLU_BLOCK // _SF_VEC,
                ).view(-1)
        tex.grouped_swizzle_for_gemm(weight, rowwise=True, columnwise=columnwise)
    else:
        quantizer = fc_op.get_quantizer("forward", 1).copy()
        quantizer.set_usage(rowwise=True, columnwise=columnwise)
        quantizer.optimize_for_gemm = True
        data = weight.rowwise_data.view(shape)
        if reorder_glu:
            data = _interleave_glu_rows(data)
        weight = tex.group_quantize(data.view(weight.logical_shape), quantizer, num_groups, None)
    fp8 = torch.float8_e4m3fn
    operands = _MxFp8Weight(weight.rowwise_data.view(fp8).view(shape), weight.scale_inv.reshape(-1))
    if columnwise:
        operands = operands._replace(
            columnwise_data=weight.columnwise_data.view(fp8).view(shape),
            columnwise_scale_inv=weight.columnwise_scale_inv.reshape(-1),
        )
    return operands


def _bf16_weight(fc_op: GroupedLinear, num_groups: int) -> torch.Tensor:
    """Expert-major BF16 weight of a packed GroupedLinear weight, cast when its data has another
    dtype, as after ``Module.to(torch.bfloat16)`` on FP32 weights."""
    return _bf16_packed_weight(fc_op, num_groups).to(torch.bfloat16)


# TODO(phuonguyen): the cuDNN GEMMs require the interleaved GLU layout, so FC1 weights are reordered
# here. Drop this reorder and pass the non-interleaved layout through once cuDNN supports it.
def _interleave_glu_rows(t: torch.Tensor, block: int = _GLU_BLOCK) -> torch.Tensor:
    """Copy of per-expert rows [experts, 2 * n, ...], gate rows then up rows, with gate and up
    alternating in blocks of ``block`` rows, the GLU layout of ``glu_interleave_size=32``."""
    experts, rows = t.shape[:2]
    blocks = t.reshape(experts, 2, rows // (2 * block), block, *t.shape[2:])
    return blocks.transpose(1, 2).reshape(t.shape)


def _deinterleave_glu_columns(t: torch.Tensor) -> torch.Tensor:
    """Copy of [rows, 2 * n] with gate and up in alternating column blocks, gate columns first."""
    rows, cols = t.shape
    # Moved in 16-byte words (complex128); a column block spans whole words.
    words = t.view(torch.complex128)
    blocks = words.reshape(rows, cols // (2 * _GLU_BLOCK), 2, _GLU_BLOCK * t.element_size() // 16)
    return blocks.transpose(1, 2).reshape(rows, -1).view(t.dtype)


def _deinterleave_glu_col_scales(scale, tokens_per_expert, cols):
    """Columnwise MXFP8 scales, swizzled per expert, of a tensor reordered by
    ``_deinterleave_glu_columns``."""
    tiles = cols // 128
    blocks = scale.view(torch.int32).view(-1, 128)
    units = blocks.shape[0] // tiles  # 128-row units
    device = scale.device
    # First unit and unit count of each unit's expert; the rows after the experts form a last group.
    ends = tokens_per_expert.cumsum(0) // 128
    starts = torch.nn.functional.pad(ends, (1, 0))
    stops = torch.nn.functional.pad(ends, (0, 1), value=units)
    unit = torch.arange(units, device=device)
    group = torch.searchsorted(stops, unit, right=True)
    first = starts[group]
    count = stops[group] - first
    # Block of each (tile, unit) in the slabs.
    tile = torch.arange(tiles, device=device)[:, None]
    index = (torch.add(unit, first, alpha=tiles - 1) + tile * count).view(-1)
    # Tile 2 * i + h holds column blocks 8 * i + 4 * h + 2 * j + g, of gate (g = 0) or up (g = 1),
    # which go to tile g * tiles / 2 + i at slot 2 * h + j.
    src = blocks[index].view(tiles // 2, 2, units, 32, 2, 2)
    out = torch.empty_like(blocks)
    out[index] = src.permute(5, 0, 2, 3, 1, 4).reshape(-1, 128)
    return out.view(torch.uint8).view(scale.shape)


def _routing_extras_internal(dispatch, fc1, activation, fc2) -> bool:
    """Whether tokens_per_expert and the routing weights flow from the dispatch to the MLP."""
    channels = dispatch._extra_output_channels
    return (
        not any(dispatch._extra_output_to_caller)
        and channels[0] is not None
        and channels[1] is not None
        and fc1._extra_input_channels[0] == channels[0]
        and fc2._extra_input_channels[0] == channels[0]
        and activation._extra_input_channels[0] == channels[1]
    )


def moe_overlap_unsupported_reason(
    config: EpConfig,
    ffn_hidden_size: int,
    *,
    params_dtype: torch.dtype,
    activation: str = "swiglu",
    glu_interleave_size: Optional[int] = None,
    bias: bool = False,
    single_grouped_weight: bool = False,
    recipe: Optional[Recipe] = None,
    primary_weights_in_fp8: bool = False,
) -> Optional[str]:
    """Why a ``Sequential`` of ``MoeDispatch``, ``GroupedLinear``, ``ScaledSwiGLU``,
    ``GroupedLinear`` and ``MoeCombine`` built with these settings does not run as one
    ``FusedMoE``, or ``None`` when it does. The reason names an unsupported setting or the
    grouped GEMM backend failing to load on this device. See ``moe_overlap_supported`` for what
    the query covers.

    Parameters
    ----------
    config : EpConfig
        EP configuration shared by the dispatch and the combine.
    ffn_hidden_size : int
        Output features of the activation, input features of the second GroupedLinear.
    params_dtype : torch.dtype
        Dtype of the GroupedLinear weights.
    activation : str, default = "swiglu"
        Activation between the GroupedLinear ops.
    glu_interleave_size : int, optional
        ``glu_interleave_size`` of the ``ScaledSwiGLU``, ``None`` or 32.
    bias : bool, default = ``False``
        Whether the GroupedLinear ops have biases.
    single_grouped_weight : bool, default = ``False``
        ``single_grouped_weight`` of the GroupedLinear ops.
    recipe : Recipe, optional
        Quantization recipe the ops run under, ``None`` for BF16. ``MXFP8BlockScaling`` needs
        MXFP8 EP transport in ``config``.
    primary_weights_in_fp8 : bool, default = ``False``
        Whether the GroupedLinear weights are quantized parameters (``quantized_model_init``).
        They must be MXFP8 and need the ``MXFP8BlockScaling`` recipe.
    """
    if not FusedMoE.is_supported():
        reason = (
            "needs SM 10.0, 10.3 or 10.7, NVTE_ENABLE_CUTEDSL_BACKEND=1, NVTE_MOE_OVERLAP and"
            " the BF16 cuDNN grouped GEMM kernels"
        )
        mismatch = _gemm_interface_reason() if _moe_overlap_available() else None
        return f"{reason}: {mismatch}" if mismatch else reason
    if _deterministic_algorithms_required():
        return (
            "deterministic execution was requested, and the routing-weight gradient is reduced"
            " with atomics"
        )
    if torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") != torch.bfloat16:
        return "needs a BF16 autocast dtype"
    if config.payload_dtype is not torch.bfloat16:
        return "needs a BF16 payload"
    alignments = (64, 128, _TILE)
    if isinstance(recipe, MXFP8BlockScaling):
        # The SM 10.7 MXFP8 GEMMs have no tail tiles.
        alignments = (_TILE,) if torch.cuda.get_device_capability() == (10, 7) else (128, _TILE)
    if config.alignment not in alignments:
        return f"the EP alignment must be one of {alignments}"
    if config.zero_copy:
        return "zero-copy EP is not supported"
    if recipe is not None:
        if not isinstance(recipe, MXFP8BlockScaling):
            return "quantized GEMMs other than MXFP8BlockScaling are not supported"
        if not GroupedMLP_CuTeGEMMGLU.is_supported():
            return "MXFP8 needs the MXFP8 cuDNN grouped GEMM kernels"
        mismatch = _gemm_interface_reason(mxfp8=True)
        if mismatch:
            return mismatch
        if recipe.fp8_format != Format.E4M3 or recipe.backward_override is not None:
            return "MXFP8 needs fp8_format E4M3 and no backward_override"
        if not all(
            isinstance(r, MXFP8BlockScaling)
            for r in (config.dispatch_fwd_quant_recipe, config.combine_bwd_quant_recipe)
        ):
            return (
                "MXFP8 needs MXFP8 EP transport (dispatch_fwd_quant_recipe and"
                " combine_bwd_quant_recipe)"
            )
        if config.hidden_dim % 512 != 0 or ffn_hidden_size % _TILE != 0:
            return f"MXFP8 needs hidden_dim a multiple of 512 and ffn_hidden_size of {_TILE}"
    elif (
        config.dispatch_fwd_quant_recipe is not None or config.combine_bwd_quant_recipe is not None
    ):
        return "quantized EP transport is not supported"
    if config.hidden_dim % _TILE != 0:
        return f"hidden_dim must be a multiple of {_TILE}"
    capacity = config.recv_capacity_per_rank
    if capacity is not None and capacity % _TILE != 0:
        return f"recv_capacity_per_rank must be a multiple of {_TILE}"
    if ffn_hidden_size % 64 != 0:
        return "ffn_hidden_size must be a multiple of 64"
    if activation != "swiglu":
        return "needs the SwiGLU activation"
    if glu_interleave_size not in (None, _GLU_BLOCK):
        return f"needs glu_interleave_size None or {_GLU_BLOCK}"
    if not single_grouped_weight:
        return "needs single_grouped_weight=True"
    if bias:
        return "biases are not supported"
    if params_dtype != torch.bfloat16:
        return "needs BF16 weights"
    if primary_weights_in_fp8 and recipe is None:
        return "quantized weight parameters need the MXFP8BlockScaling recipe"
    return None


def moe_overlap_supported(
    config: EpConfig,
    ffn_hidden_size: int,
    *,
    params_dtype: torch.dtype,
    activation: str = "swiglu",
    glu_interleave_size: Optional[int] = None,
    bias: bool = False,
    single_grouped_weight: bool = False,
    recipe: Optional[Recipe] = None,
    primary_weights_in_fp8: bool = False,
) -> bool:
    """Whether a ``Sequential`` of ``MoeDispatch``, ``GroupedLinear``, ``ScaledSwiGLU``,
    ``GroupedLinear`` and ``MoeCombine`` built with these settings runs as one ``FusedMoE``;
    ``moe_overlap_unsupported_reason`` gives the reason when it does not and takes the same
    parameters.

    Call it after ``ep_bootstrap``. It only queries the device. Deterministic execution
    (``NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`` or ``torch.use_deterministic_algorithms``) and an
    autocast dtype other than BF16 run the unfused ops; the answer follows the settings in effect
    at the call. GroupedLinear settings it does not take, such as ``delay_wgrad_compute`` or
    ``accumulate_into_main_grad``, can still make the ops run unfused.
    """
    return (
        moe_overlap_unsupported_reason(
            config,
            ffn_hidden_size,
            params_dtype=params_dtype,
            activation=activation,
            glu_interleave_size=glu_interleave_size,
            bias=bias,
            single_grouped_weight=single_grouped_weight,
            recipe=recipe,
            primary_weights_in_fp8=primary_weights_in_fp8,
        )
        is None
    )


def _unsupported_reason(dispatch, fc1, activation, fc2, combine, recipe=None) -> Optional[str]:
    """Why a window cannot use the overlapped ops, or ``None`` when it can."""
    config = dispatch.config
    dtypes = {w.dtype for fc in (fc1, fc2) for w in fc._get_weight_tensors()}
    reason = moe_overlap_unsupported_reason(
        config,
        fc2.in_features,
        params_dtype=dtypes.pop() if len(dtypes) == 1 else None,
        activation="swiglu" if isinstance(activation, ScaledSwiGLU) else type(activation).__name__,
        glu_interleave_size=activation.glu_interleave_size,
        bias=fc1.has_bias or fc2.has_bias,
        single_grouped_weight=fc1.single_grouped_weight and fc2.single_grouped_weight,
        recipe=recipe,
        primary_weights_in_fp8=any(
            getattr(w, "quantizer", None) is not None
            for fc in (fc1, fc2)
            for w in fc._get_weight_tensors()
        ),
    )
    if reason is not None:
        return reason
    if any(
        not isinstance(getattr(fc.weight, "quantizer", None), (type(None), MXFP8Quantizer))
        for fc in (fc1, fc2)
    ):
        return "quantized weight parameters must be MXFP8"
    if combine.config != config:
        return "dispatch and combine must use the same EpConfig"
    if dispatch.buffer is not combine.buffer:
        return "dispatch and combine must share one EpBuffer"
    if not _routing_extras_internal(dispatch, fc1, activation, fc2):
        return "the dispatch extra outputs must feed the grouped MLP and not the caller"
    if any(is_distributed_weight(fc.weight) for fc in (fc1, fc2)):
        return "distributed (sharded) weights are not supported"
    if any(
        fc.wgrad_store is not None and fc.wgrad_store.delay_wgrad_compute() for fc in (fc1, fc2)
    ):
        return "delayed weight gradient computation (delay_wgrad_compute) is not supported"
    try:
        # The GLU layout is checked above.
        validate_grouped_mlp_dims(fc1, ScaledSwiGLU(glu_interleave_size=_GLU_BLOCK), fc2)
    except (TypeError, ValueError) as err:
        return str(err)
    if fc1.in_features != config.hidden_dim or fc2.out_features != config.hidden_dim:
        return "the GroupedLinear sizes do not match the EP hidden_dim"
    if (
        recipe is not None
        and torch.cuda.get_device_capability() == (10, 7)
        and config.recv_capacity_per_rank == _TILE
    ):
        # Some SM 10.7 MXFP8 weight gradients need more than one receive tile; one into
        # main_grad counts as added.
        from ....common.CuTeDSL.ep_gemm import (  # pylint: disable=import-outside-toplevel
            wgrad_gemm,
        )

        for fc in (fc1, fc2):
            cfg = wgrad_gemm.WgradConfig(
                _TILE, fc.out_features, fc.in_features, config.num_local_experts, 1,
                accumulate=fc._accumulate_into_main_grad, mxfp8=True,
                rows_alignment=config.alignment,
                routed_rows=config.max_tokens_per_rank * config.top_k,
            )  # fmt: skip
            try:
                wgrad_gemm.build_kernel(cfg)
            except ValueError as err:
                return str(err)
    return None


# The GLU activations of the grouped MLP; the unsupported ones run unfused with a warning.
_WINDOW = (
    MoeDispatch,
    GroupedLinear,
    (ScaledSwiGLU, ScaledClampedQGeGLU, ScaledSiTUGLU),
    GroupedLinear,
    MoeCombine,
)


_INTERNAL_DIRS = tuple(
    os.path.abspath(path) + os.sep
    for path in (
        os.path.join(os.path.dirname(__file__), *[os.pardir] * 3),
        os.path.dirname(torch.__file__),
    )
)


def _stacklevel_outside_package() -> int:
    """``warnings.warn`` stack level, from the caller of this function, of the first frame outside
    Transformer Engine and PyTorch: the fuser runs inside the user's forward call."""
    level, frame = 1, sys._getframe(1)  # pylint: disable=protected-access
    while frame is not None and frame.f_code.co_filename.startswith(_INTERNAL_DIRS):
        level, frame = level + 1, frame.f_back
    return level if frame is not None else 1


def fuse_moe_overlap_ops(
    ops: list[FusibleOperation],
    *,
    recipe=None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Fuse dispatch, grouped MLP and combine; other windows keep the unfused ops."""
    out = []
    idx = 0
    while idx < len(ops):
        window = ops[idx : idx + 5]
        if len(window) == len(_WINDOW) and all(map(isinstance, window, _WINDOW)):
            reason = _unsupported_reason(*window, recipe=recipe)
            if reason is None:
                out.append(FusedMoE(
                    dispatch=window[0], fc1=window[1], activation=window[2],
                    fc2=window[3], combine=window[4], recipe=recipe, _validated=True,
                ))  # fmt: skip
                idx += 5
                continue
            # Without the overlap enabled the unfused ops are the expected ones.
            if _moe_overlap_available():
                warnings.warn(
                    f"FusedMoE fusion skipped, running the unfused ops: {reason}.",
                    stacklevel=_stacklevel_outside_package(),
                )
        out.append(ops[idx])
        idx += 1
    return out


register_forward_backward_fusion(fuse_moe_overlap_ops, prepend=True)
