# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python shim for stable ABI ops.

This module wraps torch.ops.transformer_engine_stable.* ops, providing
the same interface as the corresponding functions in transformer_engine_torch.
During migration, cpp_extensions/__init__.py can switch individual ops
from the unstable pybind11 module to this shim.
"""

import torch

# Lazy import: the stable extension must be loaded before these ops are available.
# torch.ops.transformer_engine_stable is populated when the shared library is loaded.
_ops_loaded = False


def _ensure_loaded():
    """Ensure the stable ABI extension is loaded.

    The stable extension is not a Python module (no PyInit_ function).
    It registers ops via STABLE_TORCH_LIBRARY at load time, so we use
    torch.ops.load_library() or ctypes to load the shared library.
    """
    global _ops_loaded
    if _ops_loaded:
        return

    import importlib.util
    from pathlib import Path

    # Find the .so file adjacent to the transformer_engine package
    te_spec = importlib.util.find_spec("transformer_engine")
    if te_spec is not None and te_spec.origin is not None:
        te_dir = Path(te_spec.origin).parent.parent
        import glob

        candidates = glob.glob(str(te_dir / "te_stable_abi*"))
        if candidates:
            torch.ops.load_library(candidates[0])
            _ops_loaded = True
            return

    _ops_loaded = True


# ============================================================================
# Softmax ops
# ============================================================================


def scaled_softmax_forward(input, scale_factor):
    """Stable ABI version of scaled_softmax_forward."""
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.scaled_softmax_forward(input, scale_factor)


def scaled_softmax_backward(output_grad, softmax_results, scale_factor):
    """Stable ABI version of scaled_softmax_backward."""
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.scaled_softmax_backward(
        output_grad, softmax_results, scale_factor
    )


def scaled_masked_softmax_forward(input, mask, scale_factor):
    """Stable ABI version of scaled_masked_softmax_forward."""
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.scaled_masked_softmax_forward(
        input, mask, scale_factor
    )


def scaled_masked_softmax_backward(output_grad, softmax_results, scale_factor):
    """Stable ABI version of scaled_masked_softmax_backward."""
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.scaled_masked_softmax_backward(
        output_grad, softmax_results, scale_factor
    )


def scaled_upper_triang_masked_softmax_forward(input, scale_factor):
    """Stable ABI version of scaled_upper_triang_masked_softmax_forward."""
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.scaled_upper_triang_masked_softmax_forward(
        input, scale_factor
    )


def scaled_upper_triang_masked_softmax_backward(output_grads, softmax_results, scale_factor):
    """Stable ABI version of scaled_upper_triang_masked_softmax_backward."""
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.scaled_upper_triang_masked_softmax_backward(
        output_grads, softmax_results, scale_factor
    )


def scaled_aligned_causal_masked_softmax_forward(input, scale_factor):
    """Stable ABI version of scaled_aligned_causal_masked_softmax_forward."""
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.scaled_aligned_causal_masked_softmax_forward(
        input, scale_factor
    )


def scaled_aligned_causal_masked_softmax_backward(output_grad, softmax_results, scale_factor):
    """Stable ABI version of scaled_aligned_causal_masked_softmax_backward."""
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.scaled_aligned_causal_masked_softmax_backward(
        output_grad, softmax_results, scale_factor
    )


# ============================================================================
# Padding ops
# ============================================================================


def fused_multi_row_padding(input, output, input_row_list, padded_input_row_list):
    _ensure_loaded()
    torch.ops.transformer_engine_stable.fused_multi_row_padding(
        input, output, input_row_list, padded_input_row_list
    )


def fused_multi_row_unpadding(input, output, input_row_list, unpadded_input_row_list):
    _ensure_loaded()
    torch.ops.transformer_engine_stable.fused_multi_row_unpadding(
        input, output, input_row_list, unpadded_input_row_list
    )


# ============================================================================
# Misc ops
# ============================================================================


def splits_to_offsets(first_dims, logical_last_dim):
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.splits_to_offsets(first_dims, logical_last_dim)


# ============================================================================
# RoPE ops
# ============================================================================


def fused_rope_forward(
    input, freqs, start_positions, qkv_format, interleaved, cu_seqlens, cp_size, cp_rank
):
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.fused_rope_forward(
        input, freqs, start_positions, int(qkv_format), interleaved, cu_seqlens, cp_size, cp_rank
    )


def fused_rope_backward(
    output_grads, freqs, start_positions, qkv_format, interleaved, cu_seqlens, cp_size, cp_rank
):
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.fused_rope_backward(
        output_grads,
        freqs,
        start_positions,
        int(qkv_format),
        interleaved,
        cu_seqlens,
        cp_size,
        cp_rank,
    )


def fused_qkv_rope_forward(
    qkv_input,
    q_freqs,
    k_freqs,
    start_positions,
    qkv_split_arg_list,
    qkv_format,
    interleaved,
    cp_size,
    cp_rank,
):
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.fused_qkv_rope_forward(
        qkv_input,
        q_freqs,
        k_freqs,
        start_positions,
        qkv_split_arg_list,
        int(qkv_format),
        interleaved,
        cp_size,
        cp_rank,
    )


def fused_qkv_rope_backward(
    q_grad_out,
    k_grad_out,
    v_grad_out,
    q_freqs,
    k_freqs,
    qkv_split_arg_list,
    qkv_format,
    interleaved,
    cp_size,
    cp_rank,
):
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.fused_qkv_rope_backward(
        q_grad_out,
        k_grad_out,
        v_grad_out,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        int(qkv_format),
        interleaved,
        cp_size,
        cp_rank,
    )


# ============================================================================
# Router ops
# ============================================================================


def fused_topk_with_score_function_fwd(
    logits,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    expert_bias,
):
    _ensure_loaded()
    ng = num_groups if num_groups is not None else -1
    gt = group_topk if group_topk is not None else -1
    sf = scaling_factor if scaling_factor is not None else 1.0
    return torch.ops.transformer_engine_stable.fused_topk_with_score_function_fwd(
        logits, topk, use_pre_softmax, ng, gt, sf, score_function, expert_bias
    )


def fused_topk_with_score_function_bwd(
    num_tokens,
    num_experts,
    routing_map,
    intermediate_output,
    grad_probs,
    grad_logits,
    topk,
    use_pre_softmax,
    scaling_factor,
    score_function,
):
    _ensure_loaded()
    sf = scaling_factor if scaling_factor is not None else 1.0
    torch.ops.transformer_engine_stable.fused_topk_with_score_function_bwd(
        num_tokens,
        num_experts,
        routing_map,
        intermediate_output,
        grad_probs,
        grad_logits,
        topk,
        use_pre_softmax,
        sf,
        score_function,
    )


def fused_score_for_moe_aux_loss_fwd(logits, topk, score_function):
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.fused_score_for_moe_aux_loss_fwd(
        logits, topk, score_function
    )


def fused_score_for_moe_aux_loss_bwd(
    num_tokens, num_experts, intermediate_output, grad_scores, grad_logits, topk, score_function
):
    _ensure_loaded()
    torch.ops.transformer_engine_stable.fused_score_for_moe_aux_loss_bwd(
        num_tokens, num_experts, intermediate_output, grad_scores, grad_logits, topk, score_function
    )


def fused_moe_aux_loss_fwd(
    probs, tokens_per_expert, total_num_tokens, num_experts, num_rows, num_cols, topk, coeff
):
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.fused_moe_aux_loss_fwd(
        probs, tokens_per_expert, total_num_tokens, num_experts, num_rows, num_cols, topk, coeff
    )


def fused_moe_aux_loss_bwd(Const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss):
    _ensure_loaded()
    return torch.ops.transformer_engine_stable.fused_moe_aux_loss_bwd(
        Const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss
    )
