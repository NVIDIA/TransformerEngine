/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common.h"

namespace transformer_engine::pytorch {

static std::map<std::string, int> score_function_map = {
    {"sigmoid", 0}, {"softmax", 1}, {"sqrtsoftplus", 2}};

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_topk_with_score_function_fwd(
    at::Tensor logits, int topk, bool use_pre_softmax, c10::optional<int> num_groups,
    c10::optional<int> group_topk, c10::optional<float> scaling_factor, std::string score_function,
    c10::optional<at::Tensor> expert_bias) {
  int num_tokens = logits.size(0);
  int num_experts = logits.size(1);
  // Check if the input is valid
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  // Expert bias only happens at the sigmoid case
  if (expert_bias.has_value()) {
    TORCH_CHECK(score_function == "sigmoid" || score_function == "sqrtsoftplus",
                "score_function must be sigmoid when expert_bias is not None");
  }
  // Check if the score function is valid
  TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                  score_function == "sqrtsoftplus",
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  if (score_function == "sigmoid" || score_function == "sqrtsoftplus") {
    use_pre_softmax = false;  // Pre-softmax only happens at the softmax case
  }

  // Reformat the input to make it compatible with the kernel
  int group_topk_value = group_topk.has_value() ? group_topk.value() : -1;
  int num_groups_value = num_groups.has_value() ? num_groups.value() : -1;
  float scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;

  // Construct the output tensor
  at::Tensor probs =
      at::empty({num_tokens, num_experts}, at::dtype(logits.scalar_type()).device(at::kCUDA));
  at::Tensor routing_map =
      at::empty({num_tokens, num_experts}, at::dtype(at::kBool).device(at::kCUDA));
  // Intermediate output is used to store the output of the softmax/sigmoid function
  at::Tensor intermediate_output =
      at::empty({num_tokens, num_experts}, at::dtype(logits.scalar_type()).device(at::kCUDA));

  auto logits_cu = makeTransformerEngineTensor(logits);
  auto probs_cu = makeTransformerEngineTensor(probs);
  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto expert_bias_cu = TensorWrapper();  // empty expert_bias_cu tensor
  if (expert_bias.has_value()) {
    expert_bias_cu = makeTransformerEngineTensor(expert_bias.value());
  }

  nvte_fused_topk_with_score_function_forward(
      logits_cu.data(), num_tokens, num_experts, topk, use_pre_softmax, num_groups_value,
      group_topk_value, scaling_factor_value, score_function_map[score_function],
      expert_bias_cu.data(), probs_cu.data(), routing_map_cu.data(), intermediate_output_cu.data(),
      at::cuda::getCurrentCUDAStream());

  return std::make_tuple(probs, routing_map, intermediate_output);
}

at::Tensor fused_topk_with_score_function_bwd(int num_tokens, int num_experts,
                                              at::Tensor routing_map,
                                              at::Tensor intermediate_output, at::Tensor grad_probs,
                                              int topk, bool use_pre_softmax,
                                              c10::optional<float> scaling_factor,
                                              std::string score_function) {
  // Get the value of the parameters
  auto scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;
  auto score_function_value = score_function_map[score_function];
  // Init the output tensor
  at::Tensor grad_logits = at::empty(
      {num_tokens, num_experts}, at::dtype(intermediate_output.scalar_type()).device(at::kCUDA));

  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits);

  nvte_fused_topk_with_score_function_backward(
      routing_map_cu.data(), intermediate_output_cu.data(), grad_probs_cu.data(), num_tokens,
      num_experts, topk, use_pre_softmax, scaling_factor_value, score_function_value,
      grad_logits_cu.data(), at::cuda::getCurrentCUDAStream());

  return grad_logits;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_score_for_moe_aux_loss_fwd(
    at::Tensor logits, int topk, std::string score_function) {
  int num_tokens = logits.size(0);
  int num_experts = logits.size(1);
  // Check if the input is valid
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  TORCH_CHECK(topk > 0, "topk must be greater than 0");
  // Check if the score function is valid
  TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                  score_function == "sqrtsoftplus",
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  int score_function_value = score_function_map[score_function];

  // Construct the output tensor
  at::Tensor scores =
      at::empty({num_tokens, num_experts}, at::dtype(logits.scalar_type()).device(at::kCUDA));
  at::Tensor routing_map =
      at::empty({num_tokens, num_experts}, at::dtype(at::kBool).device(at::kCUDA));
  at::Tensor intermediate_output =
      at::empty({num_tokens, num_experts}, at::dtype(logits.scalar_type()).device(at::kCUDA));

  auto logits_cu = makeTransformerEngineTensor(logits);
  auto scores_cu = makeTransformerEngineTensor(scores);
  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);

  nvte_fused_score_for_moe_aux_loss_forward(
      logits_cu.data(), num_tokens, num_experts, topk, score_function_value, scores_cu.data(),
      routing_map_cu.data(), intermediate_output_cu.data(), at::cuda::getCurrentCUDAStream());

  return std::make_tuple(scores, routing_map, intermediate_output);
}

at::Tensor fused_score_for_moe_aux_loss_bwd(int num_tokens, int num_experts,
                                            at::Tensor intermediate_output, at::Tensor grad_scores,
                                            int topk, std::string score_function) {
  // Get the value of the parameters
  int score_function_value = score_function_map[score_function];
  // Init the output tensor
  at::Tensor grad_logits = at::empty(
      {num_tokens, num_experts}, at::dtype(intermediate_output.scalar_type()).device(at::kCUDA));

  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto grad_scores_cu = makeTransformerEngineTensor(grad_scores);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits);

  nvte_fused_score_for_moe_aux_loss_backward(
      intermediate_output_cu.data(), grad_scores_cu.data(), num_tokens, num_experts, topk,
      score_function_value, grad_logits_cu.data(), at::cuda::getCurrentCUDAStream());

  return grad_logits;
}

std::tuple<at::Tensor, at::Tensor> fused_moe_aux_loss_fwd(at::Tensor probs,
                                                          at::Tensor tokens_per_expert,
                                                          int total_num_tokens, int num_experts,
                                                          int num_rows, int num_cols, int topk,
                                                          float coeff) {
  TORCH_CHECK(topk > 0, "topk must be greater than 0");
  TORCH_CHECK(total_num_tokens > 0, "total_num_tokens must be greater than 0");
  TORCH_CHECK(num_experts > 0, "num_experts must be greater than 0");

  // Create the output tensor
  at::Tensor aux_loss = at::empty({}, at::dtype(probs.scalar_type()).device(at::kCUDA));
  at::Tensor Const_buf = at::empty({}, at::dtype(at::kFloat).device(at::kCUDA));

  auto probs_cu = makeTransformerEngineTensor(probs);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto aux_loss_cu = makeTransformerEngineTensor(aux_loss);
  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);

  nvte_fused_moe_aux_loss_forward(probs_cu.data(), tokens_per_expert_cu.data(), total_num_tokens,
                                  num_experts, num_rows, num_cols, topk, coeff, aux_loss_cu.data(),
                                  Const_buf_cu.data(), at::cuda::getCurrentCUDAStream());

  return std::make_tuple(aux_loss, Const_buf);
}

at::Tensor fused_moe_aux_loss_bwd(at::Tensor Const_buf, at::Tensor tokens_per_expert, int num_rows,
                                  int num_cols, at::Tensor grad_aux_loss) {
  // Create the output tensor
  at::Tensor grad_probs =
      at::empty({num_rows, num_cols}, at::dtype(grad_aux_loss.scalar_type()).device(at::kCUDA));

  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto grad_aux_loss_cu = makeTransformerEngineTensor(grad_aux_loss);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs);

  // Meta data for the kernel
  nvte_fused_moe_aux_loss_backward(Const_buf_cu.data(), tokens_per_expert_cu.data(), num_rows,
                                   num_cols, grad_aux_loss_cu.data(), grad_probs_cu.data(),
                                   at::cuda::getCurrentCUDAStream());

  return grad_probs;
}

}  // namespace transformer_engine::pytorch
