/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../stable_common.h"

// This file defines the transformer_engine_stable library namespace.
// All other stable ABI files use STABLE_TORCH_LIBRARY_FRAGMENT to add schemas
// and STABLE_TORCH_LIBRARY_IMPL to add implementations.
STABLE_TORCH_LIBRARY(transformer_engine_stable, m) {
  // Softmax ops
  m.def("scaled_softmax_forward(Tensor input, float scale_factor) -> Tensor");
  m.def("scaled_softmax_backward(Tensor output_grad, Tensor softmax_results, float scale_factor) -> Tensor");
  m.def("scaled_masked_softmax_forward(Tensor input, Tensor mask, float scale_factor) -> Tensor");
  m.def("scaled_masked_softmax_backward(Tensor output_grad, Tensor softmax_results, float scale_factor) -> Tensor");
  m.def("scaled_upper_triang_masked_softmax_forward(Tensor input, float scale_factor) -> Tensor");
  m.def("scaled_upper_triang_masked_softmax_backward(Tensor output_grads, Tensor softmax_results, float scale_factor) -> Tensor");
  m.def("scaled_aligned_causal_masked_softmax_forward(Tensor input, float scale_factor) -> Tensor");
  m.def("scaled_aligned_causal_masked_softmax_backward(Tensor output_grad, Tensor softmax_results, float scale_factor) -> Tensor");

  // Padding ops
  m.def("fused_multi_row_padding(Tensor input, Tensor output, int[] input_row_list, int[] padded_input_row_list) -> ()");
  m.def("fused_multi_row_unpadding(Tensor input, Tensor output, int[] input_row_list, int[] unpadded_input_row_list) -> ()");

  // Misc ops
  m.def("splits_to_offsets(Tensor first_dims, int logical_last_dim) -> Tensor");

  // RoPE ops
  m.def("fused_rope_forward(Tensor input, Tensor freqs, Tensor? start_positions, int qkv_format, bool interleaved, Tensor? cu_seqlens, int cp_size, int cp_rank) -> Tensor");
  m.def("fused_rope_backward(Tensor output_grads, Tensor freqs, Tensor? start_positions, int qkv_format, bool interleaved, Tensor? cu_seqlens, int cp_size, int cp_rank) -> Tensor");
  m.def("fused_qkv_rope_forward(Tensor qkv_input, Tensor q_freqs, Tensor k_freqs, Tensor? start_positions, int[] qkv_split_arg_list, int qkv_format, bool interleaved, int cp_size, int cp_rank) -> (Tensor, Tensor, Tensor)");
  m.def("fused_qkv_rope_backward(Tensor q_grad_out, Tensor k_grad_out, Tensor v_grad_out, Tensor q_freqs, Tensor k_freqs, int[] qkv_split_arg_list, int qkv_format, bool interleaved, int cp_size, int cp_rank) -> Tensor");

  // Router ops
  m.def("fused_topk_with_score_function_fwd(Tensor logits, int topk, bool use_pre_softmax, int num_groups, int group_topk, float scaling_factor, str score_function, Tensor? expert_bias) -> (Tensor, Tensor, Tensor)");
  m.def("fused_topk_with_score_function_bwd(int num_tokens, int num_experts, Tensor routing_map, Tensor intermediate_output, Tensor grad_probs, Tensor grad_logits, int topk, bool use_pre_softmax, float scaling_factor, str score_function) -> ()");
  m.def("fused_score_for_moe_aux_loss_fwd(Tensor logits, int topk, str score_function) -> (Tensor, Tensor, Tensor)");
  m.def("fused_score_for_moe_aux_loss_bwd(int num_tokens, int num_experts, Tensor intermediate_output, Tensor grad_scores, Tensor grad_logits, int topk, str score_function) -> ()");
  m.def("fused_moe_aux_loss_fwd(Tensor probs, Tensor tokens_per_expert, int total_num_tokens, int num_experts, int num_rows, int num_cols, int topk, float coeff) -> (Tensor, Tensor)");
  m.def("fused_moe_aux_loss_bwd(Tensor Const_buf, Tensor tokens_per_expert, int num_rows, int num_cols, Tensor grad_aux_loss) -> Tensor");
}
