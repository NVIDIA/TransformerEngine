/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../stable_common.h"

#include <transformer_engine/fused_router.h>

#include <map>

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

static std::map<std::string, int> score_function_map = {
    {"sigmoid", 0}, {"softmax", 1}, {"sqrtsoftplus", 2}};

std::tuple<Tensor, Tensor, Tensor> fused_topk_with_score_function_fwd(
    Tensor logits, int64_t topk, bool use_pre_softmax, int64_t num_groups,
    int64_t group_topk, double scaling_factor, std::string score_function,
    std::optional<Tensor> expert_bias) {
  int64_t num_tokens = logits.size(0);
  int64_t num_experts = logits.size(1);

  STD_TORCH_CHECK(num_tokens > 0 && num_experts > 0,
                  "num_tokens and num_experts must be greater than 0");
  if (expert_bias.has_value()) {
    STD_TORCH_CHECK(
        score_function == "sigmoid" || score_function == "sqrtsoftplus",
        "score_function must be sigmoid or sqrtsoftplus when expert_bias is not None");
  }
  STD_TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                      score_function == "sqrtsoftplus",
                  "score_function must be softmax, sigmoid or sqrtsoftplus");

  if (score_function == "sigmoid" || score_function == "sqrtsoftplus") {
    use_pre_softmax = false;
  }

  int group_topk_value = static_cast<int>(group_topk);
  int num_groups_value = static_cast<int>(num_groups);
  float scaling_factor_value = static_cast<float>(scaling_factor);

  auto device_idx = logits.get_device_index();
  auto probs = allocateStableTensor(
      {num_tokens, num_experts}, logits.scalar_type(), device_idx);
  auto routing_map = allocateStableTensor(
      {num_tokens, num_experts}, ScalarType::Bool, device_idx);
  auto intermediate_output = allocateStableTensor(
      {num_tokens, num_experts}, ScalarType::Float, device_idx);

  auto logits_cu = makeTransformerEngineTensor(logits);
  auto probs_cu = makeTransformerEngineTensor(probs);
  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto expert_bias_cu = TensorWrapper();
  if (expert_bias.has_value()) {
    expert_bias_cu = makeTransformerEngineTensor(expert_bias.value());
  }

  nvte_fused_topk_with_score_function_forward(
      logits_cu.data(), static_cast<int>(num_tokens),
      static_cast<int>(num_experts), static_cast<int>(topk), use_pre_softmax,
      num_groups_value, group_topk_value, scaling_factor_value,
      score_function_map[score_function], expert_bias_cu.data(),
      probs_cu.data(), routing_map_cu.data(), intermediate_output_cu.data(),
      getCurrentCUDAStreamRaw(device_idx));

  return std::make_tuple(probs, routing_map, intermediate_output);
}

void fused_topk_with_score_function_bwd(
    int64_t num_tokens, int64_t num_experts, Tensor routing_map,
    Tensor intermediate_output, Tensor grad_probs, Tensor grad_logits,
    int64_t topk, bool use_pre_softmax, double scaling_factor,
    std::string score_function) {
  float scaling_factor_value = static_cast<float>(scaling_factor);
  auto score_function_value = score_function_map[score_function];

  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits);

  nvte_fused_topk_with_score_function_backward(
      routing_map_cu.data(), intermediate_output_cu.data(),
      grad_probs_cu.data(), static_cast<int>(num_tokens),
      static_cast<int>(num_experts), static_cast<int>(topk), use_pre_softmax,
      scaling_factor_value, score_function_value, grad_logits_cu.data(),
      getCurrentCUDAStreamRaw(routing_map.get_device_index()));
}

std::tuple<Tensor, Tensor, Tensor> fused_score_for_moe_aux_loss_fwd(
    Tensor logits, int64_t topk, std::string score_function) {
  int64_t num_tokens = logits.size(0);
  int64_t num_experts = logits.size(1);

  STD_TORCH_CHECK(num_tokens > 0 && num_experts > 0,
                  "num_tokens and num_experts must be greater than 0");
  STD_TORCH_CHECK(topk > 0, "topk must be greater than 0");
  int score_function_value = score_function_map[score_function];

  auto device_idx = logits.get_device_index();
  auto scores = allocateStableTensor(
      {num_tokens, num_experts}, ScalarType::Float, device_idx);
  auto routing_map = allocateStableTensor(
      {num_tokens, num_experts}, ScalarType::Bool, device_idx);
  auto intermediate_output = allocateStableTensor(
      {num_tokens, num_experts}, ScalarType::Float, device_idx);

  auto logits_cu = makeTransformerEngineTensor(logits);
  auto scores_cu = makeTransformerEngineTensor(scores);
  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);

  nvte_fused_score_for_moe_aux_loss_forward(
      logits_cu.data(), static_cast<int>(num_tokens),
      static_cast<int>(num_experts), static_cast<int>(topk),
      score_function_value, scores_cu.data(), routing_map_cu.data(),
      intermediate_output_cu.data(),
      getCurrentCUDAStreamRaw(device_idx));

  return std::make_tuple(scores, routing_map, intermediate_output);
}

void fused_score_for_moe_aux_loss_bwd(
    int64_t num_tokens, int64_t num_experts, Tensor intermediate_output,
    Tensor grad_scores, Tensor grad_logits, int64_t topk,
    std::string score_function) {
  int score_function_value = score_function_map[score_function];

  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto grad_scores_cu = makeTransformerEngineTensor(grad_scores);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits);

  nvte_fused_score_for_moe_aux_loss_backward(
      intermediate_output_cu.data(), grad_scores_cu.data(),
      static_cast<int>(num_tokens), static_cast<int>(num_experts),
      static_cast<int>(topk), score_function_value, grad_logits_cu.data(),
      getCurrentCUDAStreamRaw(intermediate_output.get_device_index()));
}

std::tuple<Tensor, Tensor> fused_moe_aux_loss_fwd(
    Tensor probs, Tensor tokens_per_expert, int64_t total_num_tokens,
    int64_t num_experts, int64_t num_rows, int64_t num_cols, int64_t topk,
    double coeff) {
  STD_TORCH_CHECK(topk > 0, "topk must be greater than 0");
  STD_TORCH_CHECK(total_num_tokens > 0,
                  "total_num_tokens must be greater than 0");

  auto device_idx = probs.get_device_index();
  // Scalar tensors (0-dim)
  auto aux_loss = allocateStableTensor({}, probs.scalar_type(), device_idx);
  auto Const_buf = allocateStableTensor({}, ScalarType::Float, device_idx);

  auto probs_cu = makeTransformerEngineTensor(probs);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto aux_loss_cu = makeTransformerEngineTensor(aux_loss);
  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);

  nvte_fused_moe_aux_loss_forward(
      probs_cu.data(), tokens_per_expert_cu.data(),
      static_cast<int>(total_num_tokens), static_cast<int>(num_experts),
      static_cast<int>(num_rows), static_cast<int>(num_cols),
      static_cast<int>(topk), static_cast<float>(coeff), aux_loss_cu.data(),
      Const_buf_cu.data(), getCurrentCUDAStreamRaw(device_idx));

  return std::make_tuple(aux_loss, Const_buf);
}

Tensor fused_moe_aux_loss_bwd(Tensor Const_buf, Tensor tokens_per_expert,
                              int64_t num_rows, int64_t num_cols,
                              Tensor grad_aux_loss) {
  auto device_idx = grad_aux_loss.get_device_index();
  auto grad_probs = allocateStableTensor(
      {num_rows, num_cols}, grad_aux_loss.scalar_type(), device_idx);

  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto grad_aux_loss_cu = makeTransformerEngineTensor(grad_aux_loss);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs);

  nvte_fused_moe_aux_loss_backward(
      Const_buf_cu.data(), tokens_per_expert_cu.data(),
      static_cast<int>(num_rows), static_cast<int>(num_cols),
      grad_aux_loss_cu.data(), grad_probs_cu.data(),
      getCurrentCUDAStreamRaw(device_idx));

  return grad_probs;
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  m.impl("fused_topk_with_score_function_fwd", TORCH_BOX(fused_topk_with_score_function_fwd));
  m.impl("fused_topk_with_score_function_bwd", TORCH_BOX(fused_topk_with_score_function_bwd));
  m.impl("fused_score_for_moe_aux_loss_fwd", TORCH_BOX(fused_score_for_moe_aux_loss_fwd));
  m.impl("fused_score_for_moe_aux_loss_bwd", TORCH_BOX(fused_score_for_moe_aux_loss_bwd));
  m.impl("fused_moe_aux_loss_fwd", TORCH_BOX(fused_moe_aux_loss_fwd));
  m.impl("fused_moe_aux_loss_bwd", TORCH_BOX(fused_moe_aux_loss_bwd));
}

}  // namespace transformer_engine::pytorch::stable
