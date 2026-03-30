/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/permutation.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// ============================================================================
// MOE Permutation forward
//
// The workspace tensors (sorted_indices, row_id, sorted_row_id, temp_storage)
// are allocated on first call and reused. In stable ABI, the Python shim
// manages the workspace list.
// ============================================================================

std::tuple<Tensor, Tensor> moe_permute_fwd(Tensor input, int64_t dtype, Tensor sorted_row_id,
                                           Tensor row_id_map, int64_t num_tokens, int64_t topK,
                                           int64_t num_out_tokens) {
  auto te_dtype = static_cast<DType>(dtype);
  auto shape = getStableTensorShape(input);
  NVTE_CHECK(shape.size() == 2, "Permutation input must be 2D.");
  const size_t num_cols = shape[1];

  auto device_idx = input.get_device_index();
  int64_t actual_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * topK;

  auto permuted_output = allocateStableTensor({actual_out_tokens, static_cast<int64_t>(num_cols)},
                                              GetStableScalarType(te_dtype), device_idx);

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(), std::vector<size_t>{static_cast<size_t>(num_tokens * topK), num_cols},
      te_dtype);
  auto output_cu = makeTransformerEngineTensor(
      permuted_output.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(actual_out_tokens), num_cols}, te_dtype);
  auto sorted_row_id_cu = makeTransformerEngineTensor(sorted_row_id);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  TensorWrapper empty;

  auto stream = getCurrentCUDAStreamRaw(device_idx);
  nvte_permute(input_cu.data(), output_cu.data(), sorted_row_id_cu.data(), row_id_map_cu.data(),
               empty.data(), empty.data(), empty.data(), static_cast<size_t>(num_tokens),
               static_cast<size_t>(topK), num_cols, static_cast<size_t>(actual_out_tokens), stream);

  return std::make_tuple(permuted_output, row_id_map);
}

// ============================================================================
// MOE Unpermute forward (also used as permute backward)
// ============================================================================

Tensor moe_unpermute_fwd(Tensor input, int64_t dtype, Tensor row_id_map, Tensor prob,
                         int64_t num_tokens, int64_t topK) {
  auto te_dtype = static_cast<DType>(dtype);
  auto shape = getStableTensorShape(input);
  NVTE_CHECK(shape.size() == 2, "Unpermutation input must be 2D.");
  const size_t num_cols = shape[1];

  auto device_idx = input.get_device_index();
  auto unpermuted_output = allocateStableTensor({num_tokens, static_cast<int64_t>(num_cols)},
                                                GetStableScalarType(te_dtype), device_idx);

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens) * static_cast<size_t>(topK), num_cols},
      te_dtype);
  auto output_cu = makeTransformerEngineTensor(
      unpermuted_output.data_ptr(), std::vector<size_t>{static_cast<size_t>(num_tokens), num_cols},
      te_dtype);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  auto prob_cu = makeTransformerEngineTensor(prob);

  nvte_unpermute(input_cu.data(), output_cu.data(), row_id_map_cu.data(), prob_cu.data(),
                 static_cast<size_t>(num_tokens), static_cast<size_t>(topK), num_cols,
                 getCurrentCUDAStreamRaw(device_idx));

  return unpermuted_output;
}

// ============================================================================
// MOE Unpermute backward
// ============================================================================

std::tuple<Tensor, Tensor> moe_unpermute_bwd(Tensor input_bwd, Tensor input_fwd, int64_t dtype,
                                             Tensor row_id_map, Tensor prob) {
  auto te_dtype = static_cast<DType>(dtype);
  auto shape = getStableTensorShape(input_bwd);
  NVTE_CHECK(shape.size() == 2, "Input must be 2D.");
  const size_t num_tokens = shape[0];
  const size_t num_cols = shape[1];

  auto fwd_shape = getStableTensorShape(input_fwd);
  const size_t topK = fwd_shape[0] / num_tokens;

  auto device_idx = input_bwd.get_device_index();
  auto act_grad = allocateStableTensor(
      {static_cast<int64_t>(num_tokens * topK), static_cast<int64_t>(num_cols)},
      GetStableScalarType(te_dtype), device_idx);
  auto prob_grad = allocateStableTensorZeros({static_cast<int64_t>(num_tokens * topK)},
                                             ScalarType::Float, device_idx);

  auto input_bwd_cu = makeTransformerEngineTensor(
      input_bwd.data_ptr(), std::vector<size_t>{num_tokens, num_cols}, te_dtype);
  auto act_grad_cu = makeTransformerEngineTensor(
      act_grad.data_ptr(), std::vector<size_t>{num_tokens * topK, num_cols}, te_dtype);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  auto prob_cu = makeTransformerEngineTensor(prob);
  auto prob_grad_cu = makeTransformerEngineTensor(prob_grad);
  auto input_fwd_cu = makeTransformerEngineTensor(
      input_fwd.data_ptr(), std::vector<size_t>{num_tokens * topK, num_cols}, te_dtype);
  TensorWrapper empty;

  nvte_permute(input_bwd_cu.data(), act_grad_cu.data(), empty.data(), row_id_map_cu.data(),
               prob_cu.data(), prob_grad_cu.data(), input_fwd_cu.data(), num_tokens, topK, num_cols,
               0, getCurrentCUDAStreamRaw(device_idx));

  return std::make_tuple(act_grad, prob_grad);
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def(
      "moe_permute_fwd(Tensor input, int dtype, Tensor sorted_row_id, Tensor row_id_map, int "
      "num_tokens, int topK, int num_out_tokens) -> (Tensor, Tensor)");
  m.def(
      "moe_unpermute_fwd(Tensor input, int dtype, Tensor row_id_map, Tensor prob, int num_tokens, "
      "int topK) -> Tensor");
  m.def(
      "moe_unpermute_bwd(Tensor input_bwd, Tensor input_fwd, int dtype, Tensor row_id_map, Tensor "
      "prob) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("moe_permute_fwd", TORCH_BOX(moe_permute_fwd));
  m.impl("moe_unpermute_fwd", TORCH_BOX(moe_unpermute_fwd));
  m.impl("moe_unpermute_bwd", TORCH_BOX(moe_unpermute_bwd));
}
