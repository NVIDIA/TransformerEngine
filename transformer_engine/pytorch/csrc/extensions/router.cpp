/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <numeric>

#include "../extensions.h"
#include "common.h"

namespace transformer_engine::pytorch {

static std::map<std::string, int> score_function_map = {
    {"sigmoid", 0}, {"softmax", 1}, {"sqrtsoftplus", 2}};

static int get_score_function_value(const std::string &score_function) {
  auto it = score_function_map.find(score_function);
  TORCH_CHECK(it != score_function_map.end(),
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion, got ",
              score_function);
  return it->second;
}

// Allocate a routing_map output tensor:
//   BYTEMAP   -> bool [*leading_dims, num_experts]
//   BITMAP_U8 -> uint8[*leading_dims, ceil(num_experts/8)], LSB-first
static at::Tensor allocate_routing_map(c10::IntArrayRef leading_dims, int64_t num_experts,
                                       int routing_map_format) {
  std::vector<int64_t> shape(leading_dims.begin(), leading_dims.end());
  if (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    shape.push_back((num_experts + 7) / 8);
    return at::empty(shape, at::dtype(at::kByte).device(at::kCUDA));
  }
  shape.push_back(num_experts);
  return at::empty(shape, at::dtype(at::kBool).device(at::kCUDA));
}

static void check_routing_map_format(int routing_map_format) {
  TORCH_CHECK(routing_map_format == NVTE_ROUTING_MAP_FORMAT_BYTEMAP ||
                  routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8,
              "routing_map_format must be BYTEMAP (0) or BITMAP_U8 (1), got ", routing_map_format);
}

static bool is_supported_dense_index_dtype(at::ScalarType dtype) {
  return dtype == at::kShort || dtype == at::kInt || dtype == at::kLong;
}

static void check_dense_topk_indices(const at::Tensor &topk_indices, const at::Tensor &ref,
                                     int64_t num_tokens, int topk) {
  TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
  TORCH_CHECK(topk_indices.device() == ref.device(), "topk_indices must be on the same device as ",
              "the logits/grad tensor");
  TORCH_CHECK(topk_indices.is_contiguous(), "topk_indices must be contiguous");
  TORCH_CHECK(is_supported_dense_index_dtype(topk_indices.scalar_type()),
              "topk_indices dtype must be int16, int32, or int64, got ",
              topk_indices.scalar_type());
  TORCH_CHECK(topk_indices.numel() == num_tokens * static_cast<int64_t>(topk),
              "topk_indices must contain num_tokens * topk elements, got ", topk_indices.numel(),
              " but expected ", num_tokens * static_cast<int64_t>(topk));
  TORCH_CHECK(topk_indices.dim() >= 1 && topk_indices.size(-1) == topk,
              "topk_indices last dimension must be topk=", topk, ", got shape ",
              topk_indices.sizes());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_topk_with_score_function_fwd(
    at::Tensor logits, int topk, bool use_pre_softmax, std::optional<int> num_groups,
    std::optional<int> group_topk, std::optional<float> scaling_factor, std::string score_function,
    std::optional<at::Tensor> expert_bias, int routing_map_format,
    std::optional<at::Tensor> topk_indices) {
  check_routing_map_format(routing_map_format);
  TORCH_CHECK(logits.dim() >= 1, "logits must have at least 1 dim");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  auto sizes = logits.sizes();
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  TORCH_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts], got topk=", topk,
              " num_experts=", num_experts);
  // Expert bias only happens at the sigmoid case
  if (expert_bias.has_value()) {
    TORCH_CHECK(score_function == "sigmoid" || score_function == "sqrtsoftplus",
                "score_function must be sigmoid or sqrtsoftplus when expert_bias is not None");
    TORCH_CHECK(expert_bias.value().scalar_type() == at::kFloat,
                "expert_bias must be a float32 tensor");
  }
  // Check if the score function is valid
  TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                  score_function == "sqrtsoftplus",
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  if (score_function == "sigmoid" || score_function == "sqrtsoftplus") {
    use_pre_softmax = false;  // Pre-softmax only happens at the softmax case
  }
  if (topk_indices.has_value()) {
    check_dense_topk_indices(topk_indices.value(), logits, num_tokens, topk);
  }

  // Reformat the input to make it compatible with the kernel
  int group_topk_value = group_topk.has_value() ? group_topk.value() : -1;
  int num_groups_value = num_groups.has_value() ? num_groups.value() : -1;
  float scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;

  at::Tensor probs = at::empty(sizes, at::dtype(logits.scalar_type()).device(at::kCUDA));
  at::Tensor routing_map =
      topk_indices.has_value()
          ? topk_indices.value()
          : allocate_routing_map(sizes.slice(0, sizes.size() - 1), num_experts, routing_map_format);
  at::Tensor intermediate_output = at::empty(sizes, at::dtype(at::kFloat).device(at::kCUDA));

  // 2D shape for the kernel (common-layer NVTE_CHECKs require {num_tokens, trailing_dim}).
  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> routing_map_shape_2d =
      topk_indices.has_value()
          ? std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(topk)}
          : std::vector<size_t>{
                static_cast<size_t>(num_tokens),
                static_cast<size_t>(routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8
                                        ? (num_experts + 7) / 8
                                        : num_experts)};
  auto logits_dtype = GetTransformerEngineDType(logits.scalar_type());
  auto routing_map_dtype = GetTransformerEngineDType(routing_map.scalar_type());

  auto logits_cu = makeTransformerEngineTensor(logits.data_ptr(), shape_2d, logits_dtype);
  auto probs_cu = makeTransformerEngineTensor(probs.data_ptr(), shape_2d, logits_dtype);
  auto routing_map_cu =
      makeTransformerEngineTensor(routing_map.data_ptr(), routing_map_shape_2d, routing_map_dtype);
  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);
  auto expert_bias_cu = TensorWrapper();  // empty expert_bias_cu tensor
  if (expert_bias.has_value()) {
    expert_bias_cu = makeTransformerEngineTensor(expert_bias.value());
  }

  if (topk_indices.has_value()) {
    nvte_fused_topk_with_score_function_forward_with_indices(
        logits_cu.data(), static_cast<int>(num_tokens), static_cast<int>(num_experts), topk,
        use_pre_softmax, num_groups_value, group_topk_value, scaling_factor_value,
        get_score_function_value(score_function), expert_bias_cu.data(), probs_cu.data(),
        routing_map_cu.data(), intermediate_output_cu.data(), at::cuda::getCurrentCUDAStream());
  } else {
    nvte_fused_topk_with_score_function_forward_v2(
        logits_cu.data(), static_cast<int>(num_tokens), static_cast<int>(num_experts), topk,
        use_pre_softmax, num_groups_value, group_topk_value, scaling_factor_value,
        get_score_function_value(score_function), expert_bias_cu.data(), probs_cu.data(),
        routing_map_cu.data(), static_cast<NVTERoutingMapFormat>(routing_map_format),
        intermediate_output_cu.data(), at::cuda::getCurrentCUDAStream());
  }

  return std::make_tuple(probs, routing_map, intermediate_output);
}

void fused_topk_with_score_function_bwd(at::Tensor routing_map, at::Tensor intermediate_output,
                                        at::Tensor grad_probs, at::Tensor grad_logits, int topk,
                                        bool use_pre_softmax, std::optional<float> scaling_factor,
                                        std::string score_function, bool use_dense_indices,
                                        int routing_map_format) {
  check_routing_map_format(routing_map_format);
  TORCH_CHECK(grad_probs.dim() >= 1, "grad_probs must have at least 1 dim");
  TORCH_CHECK(grad_probs.is_contiguous(), "grad_probs must be contiguous");
  TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
  auto sizes = grad_probs.sizes();
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  TORCH_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts], got topk=", topk,
              " num_experts=", num_experts);
  if (use_dense_indices) {
    check_dense_topk_indices(routing_map, grad_probs, num_tokens, topk);
  }

  auto scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;
  auto score_function_value = get_score_function_value(score_function);

  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> routing_map_shape_2d = {
      static_cast<size_t>(num_tokens),
      static_cast<size_t>(use_dense_indices
                              ? topk
                              : (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8
                                     ? (num_experts + 7) / 8
                                     : num_experts))};
  auto grad_dtype = GetTransformerEngineDType(grad_probs.scalar_type());
  auto routing_map_dtype = GetTransformerEngineDType(routing_map.scalar_type());

  auto routing_map_cu =
      makeTransformerEngineTensor(routing_map.data_ptr(), routing_map_shape_2d, routing_map_dtype);
  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs.data_ptr(), shape_2d, grad_dtype);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits.data_ptr(), shape_2d, grad_dtype);

  if (use_dense_indices) {
    nvte_fused_topk_with_score_function_backward_with_indices(
        routing_map_cu.data(), intermediate_output_cu.data(), grad_probs_cu.data(),
        static_cast<int>(num_tokens), static_cast<int>(num_experts), topk, use_pre_softmax,
        scaling_factor_value, score_function_value, grad_logits_cu.data(),
        at::cuda::getCurrentCUDAStream());
  } else {
    nvte_fused_topk_with_score_function_backward_v2(
        routing_map_cu.data(), static_cast<NVTERoutingMapFormat>(routing_map_format),
        intermediate_output_cu.data(), grad_probs_cu.data(), static_cast<int>(num_tokens),
        static_cast<int>(num_experts), topk, use_pre_softmax, scaling_factor_value,
        score_function_value, grad_logits_cu.data(), at::cuda::getCurrentCUDAStream());
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_score_for_moe_aux_loss_fwd(
    at::Tensor logits, int topk, std::string score_function, int routing_map_format) {
  check_routing_map_format(routing_map_format);
  TORCH_CHECK(logits.dim() >= 1, "logits must have at least 1 dim");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  auto sizes = logits.sizes();
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  TORCH_CHECK(topk > 0, "topk must be greater than 0");
  TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                  score_function == "sqrtsoftplus",
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  int score_function_value = get_score_function_value(score_function);

  at::Tensor scores = at::empty(sizes, at::dtype(at::kFloat).device(at::kCUDA));
  at::Tensor routing_map =
      allocate_routing_map(sizes.slice(0, sizes.size() - 1), num_experts, routing_map_format);
  at::Tensor intermediate_output = at::empty(sizes, at::dtype(at::kFloat).device(at::kCUDA));

  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> routing_map_shape_2d = {
      static_cast<size_t>(num_tokens),
      static_cast<size_t>(routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8
                              ? (num_experts + 7) / 8
                              : num_experts)};
  auto logits_dtype = GetTransformerEngineDType(logits.scalar_type());
  auto routing_map_dtype = GetTransformerEngineDType(routing_map.scalar_type());

  auto logits_cu = makeTransformerEngineTensor(logits.data_ptr(), shape_2d, logits_dtype);
  auto scores_cu = makeTransformerEngineTensor(scores.data_ptr(), shape_2d, DType::kFloat32);
  auto routing_map_cu =
      makeTransformerEngineTensor(routing_map.data_ptr(), routing_map_shape_2d, routing_map_dtype);
  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);

  nvte_fused_score_for_moe_aux_loss_forward_v2(
      logits_cu.data(), static_cast<int>(num_tokens), static_cast<int>(num_experts), topk,
      score_function_value, scores_cu.data(), routing_map_cu.data(),
      static_cast<NVTERoutingMapFormat>(routing_map_format), intermediate_output_cu.data(),
      at::cuda::getCurrentCUDAStream());

  return std::make_tuple(scores, routing_map, intermediate_output);
}

void fused_score_for_moe_aux_loss_bwd(at::Tensor intermediate_output, at::Tensor grad_scores,
                                      at::Tensor grad_logits, int topk,
                                      std::string score_function) {
  TORCH_CHECK(grad_scores.dim() >= 1, "grad_scores must have at least 1 dim");
  TORCH_CHECK(grad_scores.is_contiguous(), "grad_scores must be contiguous");
  TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
  auto sizes = grad_scores.sizes();
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());

  int score_function_value = get_score_function_value(score_function);

  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  auto grad_logits_dtype = GetTransformerEngineDType(grad_logits.scalar_type());

  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);
  auto grad_scores_cu =
      makeTransformerEngineTensor(grad_scores.data_ptr(), shape_2d, DType::kFloat32);
  auto grad_logits_cu =
      makeTransformerEngineTensor(grad_logits.data_ptr(), shape_2d, grad_logits_dtype);

  nvte_fused_score_for_moe_aux_loss_backward(
      intermediate_output_cu.data(), grad_scores_cu.data(), static_cast<int>(num_tokens),
      static_cast<int>(num_experts), topk, score_function_value, grad_logits_cu.data(),
      at::cuda::getCurrentCUDAStream());
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
  at::Tensor Const_buf = at::empty({2}, at::dtype(at::kFloat).device(at::kCUDA));

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
