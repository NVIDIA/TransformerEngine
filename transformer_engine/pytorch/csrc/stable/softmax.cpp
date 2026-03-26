/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../stable_common.h"

#include <transformer_engine/softmax.h>

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

Tensor scaled_softmax_forward(Tensor input, double scale_factor) {
  STD_TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  auto dtype = input.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");

  const int64_t batches = input.size(0);
  const int64_t attn_heads = input.size(1);
  const int64_t query_seq_len = input.size(2);
  const int64_t key_seq_len = input.size(3);

  STD_TORCH_CHECK(key_seq_len <= 16384,
                  "Key sequence length must be 16384 or less");
  STD_TORCH_CHECK(key_seq_len % 8 == 0,
                  "Key sequence length must be divisible by 8");
  STD_TORCH_CHECK(query_seq_len > 1,
                  "Query sequence length must be greater than 1");

  // Allocate output
  std::vector<int64_t> out_shape = {batches, attn_heads, query_seq_len,
                                    key_seq_len};
  auto softmax_results = allocateStableTensor(out_shape, dtype,
                                              input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(),
                              static_cast<float>(scale_factor),
                              getCurrentCUDAStreamRaw(input.get_device_index()));

  return softmax_results;
}

Tensor scaled_softmax_backward(Tensor output_grad_, Tensor softmax_results_,
                               double scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  STD_TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(softmax_results.dim() == 4, "expected 4D tensor");

  auto og_dtype = output_grads.scalar_type();
  auto sr_dtype = softmax_results.scalar_type();
  STD_TORCH_CHECK(og_dtype == ScalarType::Half || og_dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK(sr_dtype == ScalarType::Half || sr_dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(),
      output_grads_cu.data(), static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return output_grads;
}

Tensor scaled_masked_softmax_forward(Tensor input, Tensor mask,
                                     double scale_factor) {
  STD_TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  auto dtype = input.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK(mask.dim() == 4, "expected 4D tensor");

  if (!input.is_contiguous()) input = torch::stable::contiguous(input);
  if (!mask.is_contiguous()) mask = torch::stable::contiguous(mask);

  const int64_t batches = input.size(0);
  const int64_t pad_batches = mask.size(0);
  const int64_t attn_heads = input.size(1);
  const int64_t query_seq_len = input.size(2);
  const int64_t key_seq_len = input.size(3);

  STD_TORCH_CHECK(key_seq_len <= 16384,
                  "Key sequence length must be 16384 or less");
  STD_TORCH_CHECK(key_seq_len % 8 == 0,
                  "Key sequence length must be divisible by 8");
  STD_TORCH_CHECK(query_seq_len > 1,
                  "Query sequence length must be greater than 1");
  STD_TORCH_CHECK(pad_batches == 1 || pad_batches == batches,
                  "Mask batch size must be 1 or match input batch size");
  STD_TORCH_CHECK(mask.size(1) == 1, "Mask dim 1 must be 1");
  STD_TORCH_CHECK(mask.size(2) == query_seq_len,
                  "Mask dim 2 must match query_seq_len");
  STD_TORCH_CHECK(mask.size(3) == key_seq_len,
                  "Mask dim 3 must match key_seq_len");

  std::vector<int64_t> out_shape = {batches, attn_heads, query_seq_len,
                                    key_seq_len};
  auto softmax_results = allocateStableTensor(out_shape, dtype,
                                              input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto mask_cu = makeTransformerEngineTensor(mask);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_masked_softmax_forward(
      input_cu.data(), mask_cu.data(), softmax_results_cu.data(),
      static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(input.get_device_index()));

  return softmax_results;
}

Tensor scaled_masked_softmax_backward(Tensor output_grad_,
                                      Tensor softmax_results_,
                                      double scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  STD_TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(softmax_results.dim() == 4, "expected 4D tensor");

  auto og_dtype = output_grads.scalar_type();
  auto sr_dtype = softmax_results.scalar_type();
  STD_TORCH_CHECK(og_dtype == ScalarType::Half || og_dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK(sr_dtype == ScalarType::Half || sr_dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(),
      output_grads_cu.data(), static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return output_grads;
}

Tensor scaled_upper_triang_masked_softmax_forward(Tensor input,
                                                  double scale_factor) {
  STD_TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  auto dtype = input.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");

  const int64_t attn_batches = input.size(0);
  const int64_t seq_len = input.size(1);
  STD_TORCH_CHECK(seq_len <= 16384, "Sequence length must be 16384 or less");

  std::vector<int64_t> out_shape = {attn_batches, seq_len, seq_len};
  auto softmax_results = allocateStableTensor(out_shape, dtype,
                                              input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_upper_triang_masked_softmax_forward(
      input_cu.data(), softmax_results_cu.data(),
      static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(input.get_device_index()));

  return softmax_results;
}

Tensor scaled_upper_triang_masked_softmax_backward(Tensor output_grads_,
                                                   Tensor softmax_results_,
                                                   double scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grads_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  STD_TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  STD_TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");

  auto og_dtype = output_grads.scalar_type();
  auto sr_dtype = softmax_results.scalar_type();
  STD_TORCH_CHECK(og_dtype == ScalarType::Half || og_dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK(sr_dtype == ScalarType::Half || sr_dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");

  STD_TORCH_CHECK(output_grads.size(1) == output_grads.size(2),
                  "Output grads dim 1 and dim 2 must match");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_upper_triang_masked_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(),
      output_grads_cu.data(), static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return output_grads;
}

Tensor scaled_aligned_causal_masked_softmax_forward(Tensor input,
                                                    double scale_factor) {
  STD_TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  auto dtype = input.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");

  const int64_t batches = input.size(0);
  const int64_t attn_heads = input.size(1);
  const int64_t query_seq_len = input.size(2);
  const int64_t key_seq_len = input.size(3);

  STD_TORCH_CHECK(key_seq_len <= 16384,
                  "Key sequence length must be 16384 or less");
  STD_TORCH_CHECK(key_seq_len % 8 == 0,
                  "Key sequence length must be divisible by 8");
  STD_TORCH_CHECK(query_seq_len >= 1,
                  "Query sequence length must be greater or equal to 1");

  std::vector<int64_t> out_shape = {batches, attn_heads, query_seq_len,
                                    key_seq_len};
  auto softmax_results = allocateStableTensor(out_shape, dtype,
                                              input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_aligned_causal_masked_softmax_forward(
      input_cu.data(), softmax_results_cu.data(),
      static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(input.get_device_index()));

  return softmax_results;
}

Tensor scaled_aligned_causal_masked_softmax_backward(Tensor output_grad_,
                                                     Tensor softmax_results_,
                                                     double scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  STD_TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(softmax_results.dim() == 4, "expected 4D tensor");

  auto og_dtype = output_grads.scalar_type();
  auto sr_dtype = softmax_results.scalar_type();
  STD_TORCH_CHECK(og_dtype == ScalarType::Half || og_dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK(sr_dtype == ScalarType::Half || sr_dtype == ScalarType::BFloat16,
                  "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_aligned_causal_masked_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(),
      output_grads_cu.data(), static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return output_grads;
}

// ============================================================================
// Op registration via stable ABI (schemas defined in registration.cpp)
// ============================================================================

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  m.impl("scaled_softmax_forward", TORCH_BOX(scaled_softmax_forward));
  m.impl("scaled_softmax_backward", TORCH_BOX(scaled_softmax_backward));
  m.impl("scaled_masked_softmax_forward", TORCH_BOX(scaled_masked_softmax_forward));
  m.impl("scaled_masked_softmax_backward", TORCH_BOX(scaled_masked_softmax_backward));
  m.impl("scaled_upper_triang_masked_softmax_forward", TORCH_BOX(scaled_upper_triang_masked_softmax_forward));
  m.impl("scaled_upper_triang_masked_softmax_backward", TORCH_BOX(scaled_upper_triang_masked_softmax_backward));
  m.impl("scaled_aligned_causal_masked_softmax_forward", TORCH_BOX(scaled_aligned_causal_masked_softmax_forward));
  m.impl("scaled_aligned_causal_masked_softmax_backward", TORCH_BOX(scaled_aligned_causal_masked_softmax_backward));
}

}  // namespace transformer_engine::pytorch::stable
