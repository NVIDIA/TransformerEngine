# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE Permutation API"""
import warnings
from typing import Optional, Tuple
import torch

# Allow warnings.warn inside torch.compile without graph breaks
torch._dynamo.config.reorderable_logging_functions.add(warnings.warn)

import transformer_engine_torch as tex
import transformer_engine.pytorch.triton.permutation as triton_permutation
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.quantized_tensor import (
    QuantizedTensor,
    _quantized_tensor_passthrough_ops,
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

__all__ = [
    "moe_permute",
    "moe_unpermute",
    "moe_sort_chunks_by_index",
]


# ===================== _moe_permute_index_map custom ops =====================

# Workspace state for moe_permute_index_map
_moe_permute_index_map_workspace = None
_moe_permute_index_map_max_expanded_token_num = 0


@torch.library.custom_op("te_moe::permute_index_map", mutates_args=[])
def moe_permute_index_map_forward(
    inp: torch.Tensor,
    index: torch.Tensor,
    num_out_tokens: int,
    max_token_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for MoE permute with index router map."""
    global _moe_permute_index_map_workspace, _moe_permute_index_map_max_expanded_token_num

    dtype = TE_DType[inp.dtype]

    topK = index.size(1)

    input_max_expanded_token_num = max(max_token_num, inp.size(0)) * topK
    if _moe_permute_index_map_max_expanded_token_num < input_max_expanded_token_num:
        _moe_permute_index_map_max_expanded_token_num = input_max_expanded_token_num
        _moe_permute_index_map_workspace = []

    permuted_act, row_id_map, _moe_permute_index_map_workspace = tex.moe_permute_fwd(
        inp,
        dtype,
        index,
        num_out_tokens,
        _moe_permute_index_map_workspace,
        _moe_permute_index_map_max_expanded_token_num,
    )

    return permuted_act, row_id_map


@moe_permute_index_map_forward.register_fake
def _moe_permute_index_map_fake(
    inp: torch.Tensor,
    index: torch.Tensor,
    num_out_tokens: int,
    max_token_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference."""
    num_tokens = inp.shape[0]
    topK = index.shape[1]

    # Infer output shape
    output_tokens = num_out_tokens if num_out_tokens > 0 else num_tokens * topK

    # row_id_map is 1D with size = num_tokens * topK
    fake_output = torch.empty(
        (output_tokens, inp.shape[1]), dtype=inp.dtype, device=inp.device
    )
    fake_row_id_map = torch.empty(
        (num_tokens * topK,), dtype=torch.int32, device=inp.device
    )

    return fake_output, fake_row_id_map


@torch.library.custom_op("te_moe::permute_index_map_bwd", mutates_args=[])
def moe_permute_index_map_backward(
    grad_permuted_act: torch.Tensor,
    row_id_map: torch.Tensor,
    num_tokens: int,
    topK: int,
) -> torch.Tensor:
    """Backward pass for MoE permute with index router map."""
    dtype = TE_DType[grad_permuted_act.dtype]
    act_grad = tex.moe_permute_bwd(
        grad_permuted_act, dtype, row_id_map, torch.empty(0), num_tokens, topK
    )
    return act_grad


@moe_permute_index_map_backward.register_fake
def _moe_permute_index_map_backward_fake(
    grad_permuted_act: torch.Tensor,
    row_id_map: torch.Tensor,
    num_tokens: int,
    topK: int,
) -> torch.Tensor:
    """Fake implementation for shape inference of backward."""
    return torch.empty(
        (num_tokens, grad_permuted_act.shape[1]),
        dtype=grad_permuted_act.dtype,
        device=grad_permuted_act.device,
    )


def _moe_permute_index_map_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, index, num_out_tokens, max_token_num = inputs
    permuted_act, row_id_map = output
    ctx.save_for_backward(row_id_map)
    ctx.num_tokens = index.size(0)
    ctx.topK = index.size(1)


def _moe_permute_index_map_backward_wrapper(ctx, grad_permuted_act, grad_row_id_map):
    """Backward pass wrapper that calls the custom backward op."""
    if not grad_permuted_act.is_contiguous():
        grad_permuted_act = grad_permuted_act.contiguous()

    (row_id_map,) = ctx.saved_tensors
    act_grad = torch.ops.te_moe.permute_index_map_bwd(
        grad_permuted_act, row_id_map, ctx.num_tokens, ctx.topK
    )

    return act_grad, None, None, None


moe_permute_index_map_forward.register_autograd(
    _moe_permute_index_map_backward_wrapper,
    setup_context=_moe_permute_index_map_setup_context,
)


# ===================== _moe_unpermute_index_map custom ops =====================

@torch.library.custom_op("te_moe::unpermute_index_map_fwd", mutates_args=[])
def moe_unpermute_index_map_forward(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    topK: int,
) -> torch.Tensor:
    """Forward pass for MoE unpermute with index router map."""
    dtype = TE_DType[inp.dtype]
    return tex.moe_unpermute_fwd(inp, dtype, row_id_map, probs, num_tokens, topK)


@moe_unpermute_index_map_forward.register_fake
def _moe_unpermute_index_map_forward_fake(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    topK: int,
) -> torch.Tensor:
    """Fake implementation for shape inference."""
    # Output shape: (num_tokens, hidden_size)
    return torch.empty(
        (num_tokens, inp.shape[1]), dtype=inp.dtype, device=inp.device
    )


@torch.library.custom_op("te_moe::unpermute_index_map_bwd", mutates_args=[])
def moe_unpermute_index_map_backward(
    unpermuted_act_grad: torch.Tensor,
    fwd_input: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for MoE unpermute with index router map."""
    dtype = TE_DType[unpermuted_act_grad.dtype]
    act_grad, prob_grad = tex.moe_unpermute_bwd(
        unpermuted_act_grad, fwd_input, dtype, row_id_map, probs
    )
    return act_grad, prob_grad


@moe_unpermute_index_map_backward.register_fake
def _moe_unpermute_index_map_backward_fake(
    unpermuted_act_grad: torch.Tensor,
    fwd_input: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference of backward."""
    # act_grad shape: (fwd_input.size(0), hidden_size)
    # prob_grad shape: (num_tokens, topK)
    topK = probs.size(1) if probs.numel() > 0 else 1
    num_tokens = probs.size(0) if probs.numel() > 0 else row_id_map.size(0)
    act_grad = torch.empty(
        (fwd_input.size(0), unpermuted_act_grad.shape[1]),
        dtype=unpermuted_act_grad.dtype,
        device=unpermuted_act_grad.device,
    )
    prob_grad = torch.empty(
        (num_tokens, topK), dtype=torch.float32, device=unpermuted_act_grad.device
    )
    return act_grad, prob_grad



def _moe_unpermute_index_map_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, row_id_map, probs, num_tokens, topK = inputs
    ctx.save_for_backward(inp, row_id_map, probs)
    ctx.needs_probs_grad = probs.requires_grad


def _moe_unpermute_index_map_backward_wrapper(ctx, unpermuted_act_grad):
    """Backward pass wrapper that calls the custom backward op."""
    if not unpermuted_act_grad.is_contiguous():
        unpermuted_act_grad = unpermuted_act_grad.contiguous()

    inp, row_id_map, probs = ctx.saved_tensors

    act_grad, prob_grad = torch.ops.te_moe.unpermute_index_map_bwd(
        unpermuted_act_grad, inp, row_id_map, probs
    )

    if not ctx.needs_probs_grad:
        prob_grad = None

    return act_grad, None, prob_grad, None, None


moe_unpermute_index_map_forward.register_autograd(
    _moe_unpermute_index_map_backward_wrapper,
    setup_context=_moe_unpermute_index_map_setup_context,
)


# ===================== _moe_permute_mask_map custom ops =====================

@torch.library.custom_op("te_moe::permute_mask_map_fwd", mutates_args=[])
def moe_permute_mask_map_forward(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    probs: Optional[torch.Tensor],
    pad_offsets: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for MoE permute with mask router map."""
    num_tokens, hidden_size = inp.size()
    num_experts = routing_map.size(1)

    row_id_map = triton_permutation.make_row_id_map(routing_map, num_tokens, num_experts)

    # FP8 handling
    fp8 = isinstance(inp, QuantizedTensor)
    per_tensor_recipe = isinstance(inp, Float8Tensor)
    blockwise_recipe = isinstance(inp, Float8BlockwiseQTensor)
    mxfp8_recipe = isinstance(inp, MXFP8Tensor)

    if fp8:
        fp8_dtype = inp._fp8_dtype
        fake_dtype = inp.dtype
        if blockwise_recipe:
            fp8_scale = inp._rowwise_scale_inv.T.contiguous()
            scale_hidden_dim = fp8_scale.shape[1]
            assert num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
            inp = inp._rowwise_data
        elif mxfp8_recipe:
            fp8_scale = inp._rowwise_scale_inv.contiguous()
            scale_hidden_dim = fp8_scale.shape[1]
            assert num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
            inp = inp._rowwise_data
        elif per_tensor_recipe:
            fp8_scale = None
            scale_hidden_dim = None
            fp8_scale_inv = inp._scale_inv
            inp = inp._data
        else:
            raise ValueError("Unsupported FP8 recipe")
    else:
        fp8_scale = None
        fp8_dtype = None
        scale_hidden_dim = None

    output, permuted_scale, permuted_probs = triton_permutation.permute_with_mask_map(
        inp, row_id_map, probs, fp8_scale, pad_offsets,
        num_tokens, num_experts, num_out_tokens, hidden_size, scale_hidden_dim,
    )

    if fp8:
        if per_tensor_recipe:
            output = Float8Tensor(
                data=output, fp8_dtype=fp8_dtype, fp8_scale_inv=fp8_scale_inv,
                shape=output.shape, dtype=fake_dtype,
            )
        elif blockwise_recipe:
            output = Float8BlockwiseQTensor(
                shape=output.shape, dtype=fake_dtype, rowwise_data=output,
                rowwise_scale_inv=permuted_scale.T.contiguous(),
                columnwise_data=None, columnwise_scale_inv=None,
                fp8_dtype=fp8_dtype, quantizer=None, is_2D_scaled=False,
                requires_grad=output.requires_grad,
            )
        elif mxfp8_recipe:
            output = MXFP8Tensor(
                shape=output.shape, dtype=fake_dtype, fp8_dtype=fp8_dtype,
                rowwise_data=output, rowwise_scale_inv=permuted_scale.contiguous(),
                columnwise_data=None, columnwise_scale_inv=None,
                quantizer=None, requires_grad=output.requires_grad,
                with_gemm_swizzled_scales=False,
            )

    # If permuted_probs is None, return empty tensor (custom ops need concrete tensors)
    if permuted_probs is None:
        permuted_probs = torch.empty(0, device=inp.device)


    return output, row_id_map, permuted_probs


@moe_permute_mask_map_forward.register_fake
def _moe_permute_mask_map_forward_fake(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    probs: Optional[torch.Tensor],
    pad_offsets: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference."""
    num_tokens = inp.shape[0]
    hidden_size = inp.shape[1]
    num_experts = routing_map.shape[1]
    # row_id_map: (num_tokens, num_experts * 2 + 1)
    fake_output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device=inp.device)
    fake_row_id_map = torch.empty(
        (num_tokens, num_experts * 2 + 1), dtype=torch.int32, device=inp.device
    )
    if probs is not None:
        fake_permuted_probs = torch.empty((num_out_tokens,), dtype=probs.dtype, device=inp.device)
    else:
        fake_permuted_probs = torch.empty(0, device=inp.device)
    return fake_output, fake_row_id_map, fake_permuted_probs


@torch.library.custom_op("te_moe::permute_mask_map_bwd", mutates_args=[])
def moe_permute_mask_map_backward(
    permuted_act_grad: torch.Tensor,
    permuted_probs_grad: Optional[torch.Tensor],
    row_id_map: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for MoE permute with mask router map."""
    act_grad, probs_grad = triton_permutation.unpermute_with_mask_map(
        permuted_act_grad, row_id_map, None, permuted_probs_grad, pad_offsets,
        num_tokens, num_experts, hidden_size,
    )
    if probs_grad is None:
        probs_grad = torch.empty(0, device=permuted_act_grad.device)
    return act_grad, probs_grad


@moe_permute_mask_map_backward.register_fake
def _moe_permute_mask_map_backward_fake(
    permuted_act_grad: torch.Tensor,
    permuted_probs_grad: Optional[torch.Tensor],
    row_id_map: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake for backward shape inference."""
    act_grad = torch.empty(
        (num_tokens, hidden_size), dtype=permuted_act_grad.dtype, device=permuted_act_grad.device
    )
    if permuted_probs_grad is not None:
        probs_grad = torch.empty(
            (num_tokens, num_experts), dtype=permuted_probs_grad.dtype,
            device=permuted_act_grad.device,
        )
    else:
        probs_grad = torch.empty(0, device=permuted_act_grad.device)
    return act_grad, probs_grad


def _moe_permute_mask_map_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, routing_map, num_out_tokens, probs, pad_offsets = inputs
    output_tensor, row_id_map, permuted_probs = output
    ctx.save_for_backward(row_id_map, pad_offsets)
    ctx.num_experts = routing_map.size(1)
    ctx.num_tokens = inp.size(0)
    ctx.hidden_size = inp.size(1)
    ctx.needs_probs_grad = probs is not None and probs.requires_grad


def _moe_permute_mask_map_backward_wrapper(ctx, grad_output, grad_row_id_map, grad_permuted_probs):
    """Backward wrapper calling the custom backward op."""
    assert not isinstance(
        grad_output, QuantizedTensor
    ), "The backward of moe_permute does not support FP8."

    row_id_map, pad_offsets = ctx.saved_tensors

    # Pass permuted_probs_grad only if it has content
    probs_grad_input = grad_permuted_probs if grad_permuted_probs.numel() > 0 else None

    act_grad, probs_grad = torch.ops.te_moe.permute_mask_map_bwd(
        grad_output, probs_grad_input, row_id_map, pad_offsets,
        ctx.num_tokens, ctx.num_experts, ctx.hidden_size,
    )

    if not ctx.needs_probs_grad or probs_grad.numel() == 0:
        probs_grad = None

    return act_grad, None, None, probs_grad, None


moe_permute_mask_map_forward.register_autograd(
    _moe_permute_mask_map_backward_wrapper,
    setup_context=_moe_permute_mask_map_setup_context,
)


# ===================== _moe_unpermute_mask_map custom ops =====================

@torch.library.custom_op("te_moe::unpermute_mask_map_fwd", mutates_args=[])
def moe_unpermute_mask_map_forward(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    pad_offsets: Optional[torch.Tensor],
) -> torch.Tensor:
    """Forward pass for MoE unpermute with mask router map."""
    assert not isinstance(
        inp, QuantizedTensor
    ), "The forward of moe_unpermute does not support FP8."
    unpermuted_output, _ = triton_permutation.unpermute_with_mask_map(
        inp, row_id_map, merging_probs, None, pad_offsets,
        num_tokens, num_experts, hidden_size,
    )
    return unpermuted_output


@moe_unpermute_mask_map_forward.register_fake
def _moe_unpermute_mask_map_forward_fake(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    pad_offsets: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fake implementation for shape inference."""
    return torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)


@torch.library.custom_op("te_moe::unpermute_mask_map_bwd_with_probs", mutates_args=[])
def moe_unpermute_mask_map_backward_with_probs(
    unpermuted_act_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_permuted_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for MoE unpermute with merging probs."""
    act_grad, probs_grad = triton_permutation.unpermute_with_mask_map_bwd_with_merging_probs(
        unpermuted_act_grad, row_id_map, fwd_input, merging_probs, pad_offsets,
        num_tokens, num_experts, num_permuted_tokens, hidden_size,
    )
    return act_grad, probs_grad


@moe_unpermute_mask_map_backward_with_probs.register_fake
def _moe_unpermute_mask_map_bwd_with_probs_fake(
    unpermuted_act_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_permuted_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake for backward shape inference with merging probs."""
    act_grad = torch.empty(
        (num_permuted_tokens, hidden_size),
        dtype=unpermuted_act_grad.dtype, device=unpermuted_act_grad.device,
    )
    probs_grad = torch.empty(
        (num_tokens, num_experts),
        dtype=merging_probs.dtype, device=unpermuted_act_grad.device,
    )
    return act_grad, probs_grad


@torch.library.custom_op("te_moe::unpermute_mask_map_bwd_no_probs", mutates_args=[])
def moe_unpermute_mask_map_backward_no_probs(
    unpermuted_act_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_permuted_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    """Backward pass for MoE unpermute without merging probs (permute grad back)."""
    # FP8 handling
    fp8 = isinstance(unpermuted_act_grad, QuantizedTensor)
    per_tensor_recipe = isinstance(unpermuted_act_grad, Float8Tensor)
    blockwise_recipe = isinstance(unpermuted_act_grad, Float8BlockwiseQTensor)
    mxfp8_recipe = isinstance(unpermuted_act_grad, MXFP8Tensor)

    if fp8:
        fp8_dtype = unpermuted_act_grad._fp8_dtype
        fake_dtype = unpermuted_act_grad.dtype
        if per_tensor_recipe:
            fp8_scale = None
            scale_hidden_dim = None
            fp8_scale_inv = unpermuted_act_grad._scale_inv
            unpermuted_act_grad = unpermuted_act_grad._data
        elif blockwise_recipe:
            fp8_scale = unpermuted_act_grad._rowwise_scale_inv.T.contiguous()
            unpermuted_act_grad = unpermuted_act_grad._rowwise_data
            scale_hidden_dim = fp8_scale.shape[1]
            assert num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
        elif mxfp8_recipe:
            fp8_scale = unpermuted_act_grad._rowwise_scale_inv.contiguous()
            unpermuted_act_grad = unpermuted_act_grad._rowwise_data
            scale_hidden_dim = fp8_scale.shape[1]
            assert num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
        else:
            raise ValueError("Unsupported FP8 recipe")
    else:
        scale_hidden_dim = None
        fp8_dtype = None
        fp8_scale = None

    act_grad, permuted_scale, _ = triton_permutation.permute_with_mask_map(
        unpermuted_act_grad, row_id_map, None, fp8_scale, pad_offsets,
        num_tokens, num_experts, num_permuted_tokens, hidden_size, scale_hidden_dim,
    )

    if fp8:
        if per_tensor_recipe:
            act_grad = Float8Tensor(
                data=act_grad, fp8_dtype=fp8_dtype, fp8_scale_inv=fp8_scale_inv,
                shape=act_grad.shape, dtype=fake_dtype,
            )
        elif blockwise_recipe:
            act_grad = Float8BlockwiseQTensor(
                shape=act_grad.shape, dtype=fake_dtype, rowwise_data=act_grad,
                rowwise_scale_inv=permuted_scale.T.contiguous(),
                columnwise_data=None, columnwise_scale_inv=None,
                fp8_dtype=fp8_dtype, quantizer=None, is_2D_scaled=False,
                requires_grad=act_grad.requires_grad,
            )
        elif mxfp8_recipe:
            act_grad = MXFP8Tensor(
                shape=act_grad.shape, dtype=fake_dtype, fp8_dtype=fp8_dtype,
                rowwise_data=act_grad, rowwise_scale_inv=permuted_scale.contiguous(),
                columnwise_data=None, columnwise_scale_inv=None,
                quantizer=None, requires_grad=act_grad.requires_grad,
                with_gemm_swizzled_scales=False,
            )

    return act_grad


@moe_unpermute_mask_map_backward_no_probs.register_fake
def _moe_unpermute_mask_map_bwd_no_probs_fake(
    unpermuted_act_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_permuted_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    """Fake for backward shape inference without probs."""
    return torch.empty(
        (num_permuted_tokens, hidden_size),
        dtype=unpermuted_act_grad.dtype, device=unpermuted_act_grad.device,
    )


def _moe_unpermute_mask_map_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, row_id_map, merging_probs, num_tokens, num_experts, hidden_size, pad_offsets = inputs
    ctx.num_experts = num_experts
    ctx.num_tokens = num_tokens
    ctx.num_permuted_tokens = inp.size(0)
    ctx.hidden_size = hidden_size
    ctx.with_probs = merging_probs is not None
    if ctx.with_probs:
        ctx.save_for_backward(inp, row_id_map, merging_probs, pad_offsets)
        ctx.needs_probs_grad = merging_probs.requires_grad
    else:
        ctx.save_for_backward(row_id_map, pad_offsets)
        ctx.needs_probs_grad = False


def _moe_unpermute_mask_map_backward_wrapper(ctx, unpermuted_act_grad):
    """Backward wrapper calling the appropriate custom backward op."""
    act_grad = None
    probs_grad = None

    if ctx.with_probs:
        fwd_input, row_id_map, merging_probs, pad_offsets = ctx.saved_tensors
        assert not isinstance(
            unpermuted_act_grad, QuantizedTensor
        ), "The backward of moe_unpermute with merging probs does not support FP8."
        act_grad, probs_grad = torch.ops.te_moe.unpermute_mask_map_bwd_with_probs(
            unpermuted_act_grad, row_id_map, fwd_input, merging_probs, pad_offsets,
            ctx.num_tokens, ctx.num_experts, ctx.num_permuted_tokens, ctx.hidden_size,
        )
    else:
        row_id_map, pad_offsets = ctx.saved_tensors
        act_grad = torch.ops.te_moe.unpermute_mask_map_bwd_no_probs(
            unpermuted_act_grad, row_id_map, pad_offsets,
            ctx.num_tokens, ctx.num_experts, ctx.num_permuted_tokens, ctx.hidden_size,
        )

    if not ctx.needs_probs_grad:
        probs_grad = None

    return act_grad, None, probs_grad, None, None, None, None


moe_unpermute_mask_map_forward.register_autograd(
    _moe_unpermute_mask_map_backward_wrapper,
    setup_context=_moe_unpermute_mask_map_setup_context,
)

# Register all te_moe custom ops as passthrough in QuantizedTensor.__torch_dispatch__
# so that FP8 tensors are not unwrapped before entering these ops.
_quantized_tensor_passthrough_ops.update({
    torch.ops.te_moe.permute_mask_map_fwd.default,
    torch.ops.te_moe.permute_mask_map_bwd.default,
    torch.ops.te_moe.unpermute_mask_map_fwd.default,
    torch.ops.te_moe.unpermute_mask_map_bwd_with_probs.default,
    torch.ops.te_moe.unpermute_mask_map_bwd_no_probs.default,
})


def moe_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
    max_token_num: int = -1,
    map_type: str = "mask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens based on the routing_map. Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    routing_map : torch.Tensor
        The token to expert mapping tensor.
        If map_type is 'mask', routing_map is of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
        If map_type is 'index', routing_map is of shape [num_tokens, topK] and dtype 'int32'.
        The values in it are the routed expert indices.
    num_out_tokens : int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    max_token_num : int, default = -1
        The maximum number of tokens, used for workspace allocation.
        By default, set to '-1', meaning the calculation of the size of workspace is
        automatically taken over by the operator.
    map_type : str, default = 'mask'
        Type of the routing map tensor.
        Options are: 'mask', 'index'.
        Refer to `routing_map` for more details.
    """
    if not inp.numel():
        return inp, torch.tensor([], device=inp.device)
    assert inp.is_cuda, "TransformerEngine needs CUDA."
    assert routing_map.is_cuda, "TransformerEngine needs CUDA."
    assert inp.size(0) == routing_map.size(0), "Permute not possible"
    if routing_map.dtype != torch.int32:
        warnings.warn(
            f"The data type of the input `routing_map` of Permute is {routing_map.dtype}! "
            "The recommended type is torch.int32."
        )
        routing_map = routing_map.to(torch.int32)
    if map_type == "index":
        return torch.ops.te_moe.permute_index_map(inp, routing_map, num_out_tokens, max_token_num)
    if map_type == "mask":
        output, row_id_map, _ = torch.ops.te_moe.permute_mask_map_fwd(
            inp, routing_map, num_out_tokens, None, None
        )
        return output, row_id_map
    raise ValueError("map_type should be one of 'mask' or 'index'")


def moe_permute_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens and probs based on the routing_map.
    Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs : torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens, num_experts]. It will be permuted with the tokens
        according to the routing_map.
    routing_map : torch.Tensor
        The token to expert mapping tensor of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
    num_out_tokens : int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    """
    if not inp.numel():
        # Keep probs in autograd graph so that probs.grad is an empty tensor
        # instead of None after backward (backward compatibility).
        return (
            inp + probs.sum() * 0,
            probs.sum(dim=1),
            torch.tensor([], device=inp.device),
        )
    output, row_id_map, permuted_probs = torch.ops.te_moe.permute_mask_map_fwd(
        inp, routing_map, num_out_tokens, probs, None
    )
    return output, permuted_probs, row_id_map


def moe_permute_and_pad_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    align_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Permute the tokens and probs based on the routing_map.
    Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs: torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens, num_experts]. It will be permuted with the tokens
        according to the routing_map.
    routing_map: torch.Tensor
        The token to expert mapping tensor of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
    tokens_per_expert : torch.Tensor
        Tensor of shape `[num_experts]` containing actual token counts per expert.
    align_size : int
        the alignment size for the input tensor.
    """
    assert (
        tokens_per_expert is not None
    ), "tokens_per_expert must be provided to the fused permute padding function."
    assert align_size > 0, f"align_size must be positive, got {align_size}"

    # Ensure tokens_per_expert is on the same device as input to avoid device transfers
    if tokens_per_expert.device != inp.device:
        tokens_per_expert = tokens_per_expert.to(inp.device)

    # Calculate aligned token counts per expert
    target_tokens_per_expert = (torch.ceil(tokens_per_expert / align_size) * align_size).long()

    if torch.equal(tokens_per_expert, target_tokens_per_expert):
        pad_offsets = None
    else:
        pad_lengths = target_tokens_per_expert - tokens_per_expert
        cum_pad = torch.cumsum(pad_lengths, dim=0)
        pad_offsets = torch.cat(
            [torch.zeros(1, dtype=cum_pad.dtype, device=inp.device), cum_pad[:-1]]
        )

    output, row_id_map, permuted_probs = torch.ops.te_moe.permute_mask_map_fwd(
        inp, routing_map, target_tokens_per_expert.sum().item(), probs, pad_offsets
    )
    return output, permuted_probs, row_id_map, pad_offsets, target_tokens_per_expert


def moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor] = None,
    restore_shape: Optional[torch.Size] = None,
    map_type: str = "mask",
    probs: Optional[torch.Tensor] = None,
    pad_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    row_id_map : torch.Tensor
        The tensor of a mapping table for sorted indices used to unpermute the tokens,
        which is the second output tensor of `Permute`.
    merging_probs : torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    restore_shape : torch.Size, default = None
        The output shape after the unpermute operation.
    map_type : str, default = 'mask'
        Type of the routing map tensor. Should be the same as the value passed to moe_permute.
        Options are: 'mask', 'index'.
    probs : torch.Tensor, default = None
        Renamed to merging_probs. Keep for backward compatibility.
    pad_offsets : torch.Tensor, default = None
        Tensor of per-expert cumulative padding offsets used to remove padding added
        during permutation. This is the fourth output of `moe_permute_and_pad_with_probs`
        and is required when unpermuting padded outputs.
    """
    if probs is not None:
        if merging_probs is not None:
            raise ValueError(
                "Both merging_probs and probs kwarg are provided. probs is deprecated."
            )
        warnings.warn("probs kwarg is deprecated. Use merging_probs kwarg instead.")
        merging_probs = probs
    if map_type == "index":
        # Empty input check
        if not inp.numel():
            return inp

        # Normalize probs
        if merging_probs is not None:
            assert merging_probs.is_cuda, "TransformerEngine needs CUDA."
            if merging_probs.dtype != torch.float32:
                warnings.warn(
                    f"The data type of the input `probs` of Unpermute is {merging_probs.dtype}! "
                    "The recommended type is torch.float32."
                )
                merging_probs = merging_probs.to(torch.float32)
            num_tokens = merging_probs.size(0)
            topK = merging_probs.size(1)
        else:
            num_tokens = row_id_map.size(0)
            topK = 1
            merging_probs = torch.empty(0, device=inp.device)

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."
        if row_id_map.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `row_id_map` of Unpermute is {row_id_map.dtype}! "
                "The recommended type is torch.int32."
            )
            row_id_map = row_id_map.to(torch.int32)

        return torch.ops.te_moe.unpermute_index_map_fwd(inp, row_id_map, merging_probs, num_tokens, topK)
    if map_type == "mask":
        if not inp.numel():
            # Keep merging_probs in autograd graph so that probs.grad is an empty
            # tensor instead of None after backward (backward compatibility).
            if merging_probs is not None:
                return inp + merging_probs.sum() * 0
            return inp

        if restore_shape is None:
            restore_shape = inp.shape
        num_tokens, hidden_size = restore_shape
        num_experts = (row_id_map.size(1) - 1) // 2

        if merging_probs is not None:
            assert merging_probs.is_cuda, "TransformerEngine needs CUDA."
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."
        if pad_offsets is not None:
            assert pad_offsets.is_cuda, "TransformerEngine needs CUDA."

        return torch.ops.te_moe.unpermute_mask_map_fwd(
            inp, row_id_map, merging_probs,
            num_tokens, num_experts, hidden_size, pad_offsets,
        )
    raise ValueError("map_type should be one of 'mask' or 'index'")


# ===================== _moe_chunk_sort custom ops =====================

@torch.library.custom_op("te_moe::chunk_sort_fwd", mutates_args=[])
def moe_chunk_sort_forward(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for MoE chunk sort. Returns (output, permuted_probs, row_id_map)."""
    num_tokens, hidden_size = inp.shape
    num_splits = split_sizes.size(0)

    fp8 = isinstance(inp, Float8Tensor)
    if fp8:
        fp8_dtype = inp._fp8_dtype
        fp8_scale_inv = inp._scale_inv
        fake_dtype = inp.dtype
        inp = inp._data

    row_id_map = triton_permutation.make_chunk_sort_map(
        split_sizes, sorted_idxs, num_tokens, num_splits,
    )
    output, permuted_probs = triton_permutation.sort_chunks_by_map(
        inp, row_id_map, probs, num_tokens, hidden_size, is_forward=True,
    )
    if fp8:
        output = Float8Tensor(
            data=output, fp8_dtype=fp8_dtype, fp8_scale_inv=fp8_scale_inv,
            shape=output.shape, dtype=fake_dtype,
        )

    if permuted_probs is None:
        permuted_probs = torch.empty(0, device=output.device)

    return output, permuted_probs, row_id_map


@moe_chunk_sort_forward.register_fake
def _moe_chunk_sort_forward_fake(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake for shape inference."""
    num_tokens = inp.shape[0]
    hidden_size = inp.shape[1]
    fake_output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)
    if probs is not None:
        fake_probs = torch.empty((num_tokens,), dtype=probs.dtype, device=inp.device)
    else:
        fake_probs = torch.empty(0, device=inp.device)
    # row_id_map: 1D, size num_tokens
    fake_row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device=inp.device)
    return fake_output, fake_probs, fake_row_id_map


@torch.library.custom_op("te_moe::chunk_sort_bwd", mutates_args=[])
def moe_chunk_sort_backward(
    permuted_act_grad: torch.Tensor,
    permuted_probs_grad: Optional[torch.Tensor],
    row_id_map: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for MoE chunk sort."""
    fp8 = isinstance(permuted_act_grad, Float8Tensor)
    if fp8:
        fp8_dtype = permuted_act_grad._fp8_dtype
        fp8_scale_inv = permuted_act_grad._scale_inv
        fake_dtype = permuted_act_grad.dtype
        permuted_act_grad = permuted_act_grad._data

    act_grad, probs_grad = triton_permutation.sort_chunks_by_map(
        permuted_act_grad, row_id_map, permuted_probs_grad,
        num_tokens, hidden_size, is_forward=False,
    )

    if fp8:
        act_grad = Float8Tensor(
            data=act_grad, fp8_dtype=fp8_dtype, fp8_scale_inv=fp8_scale_inv,
            shape=act_grad.shape, dtype=fake_dtype,
        )

    if probs_grad is None:
        probs_grad = torch.empty(0, device=act_grad.device)

    return act_grad, probs_grad


@moe_chunk_sort_backward.register_fake
def _moe_chunk_sort_backward_fake(
    permuted_act_grad: torch.Tensor,
    permuted_probs_grad: Optional[torch.Tensor],
    row_id_map: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake for backward shape inference."""
    fake_act_grad = torch.empty(
        (num_tokens, hidden_size), dtype=permuted_act_grad.dtype, device=permuted_act_grad.device,
    )
    if permuted_probs_grad is not None:
        fake_probs_grad = torch.empty(
            (num_tokens,), dtype=permuted_probs_grad.dtype, device=permuted_act_grad.device,
        )
    else:
        fake_probs_grad = torch.empty(0, device=permuted_act_grad.device)
    return fake_act_grad, fake_probs_grad


def _moe_chunk_sort_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, split_sizes, sorted_idxs, probs = inputs
    output_tensor, permuted_probs, row_id_map = output

    ctx.save_for_backward(row_id_map)
    ctx.num_tokens = inp.size(0)
    ctx.hidden_size = inp.size(1)
    ctx.needs_probs_grad = probs is not None and probs.requires_grad


def _moe_chunk_sort_backward_wrapper(ctx, permuted_act_grad, permuted_probs_grad, _row_id_map_grad):
    """Backward wrapper calling the custom backward op."""
    (row_id_map,) = ctx.saved_tensors

    probs_grad_input = permuted_probs_grad if permuted_probs_grad.numel() > 0 else None

    act_grad, probs_grad = torch.ops.te_moe.chunk_sort_bwd(
        permuted_act_grad, probs_grad_input, row_id_map,
        ctx.num_tokens, ctx.hidden_size,
    )

    if not ctx.needs_probs_grad or probs_grad.numel() == 0:
        probs_grad = None

    return act_grad, None, None, probs_grad


moe_chunk_sort_forward.register_autograd(
    _moe_chunk_sort_backward_wrapper,
    setup_context=_moe_chunk_sort_setup_context,
)

# Register chunk sort ops as passthrough in QuantizedTensor.__torch_dispatch__
_quantized_tensor_passthrough_ops.update({
    torch.ops.te_moe.chunk_sort_fwd.default,
    torch.ops.te_moe.chunk_sort_bwd.default,
})


def moe_sort_chunks_by_index(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    split_sizes : torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices : torch.Tensor
        Chunk indices used to permute the chunks.
    """
    if not inp.numel():
        return inp
    output, _, _ = torch.ops.te_moe.chunk_sort_fwd(inp, split_sizes, sorted_index, None)
    return output


def moe_sort_chunks_by_index_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor and probs based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs : torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens]. It will be permuted with the tokens according to
        the split_sizes and sorted_indices.
    split_sizes : torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices : torch.Tensor
        Chunk indices used to permute the chunks.
    """
    if not inp.numel():
        return inp, probs
    output, permuted_probs, _ = torch.ops.te_moe.chunk_sort_fwd(inp, split_sizes, sorted_index, probs)
    return output, permuted_probs
