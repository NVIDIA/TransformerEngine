/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/softmax.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

Tensor scaled_softmax_forward(Tensor input, double scale_factor) {
  NVTE_CHECK(input.dim() == 4, "expected 4D tensor");
  check_fp16_bf16(input, "scaled_softmax_forward");

  auto sizes = input.sizes();
  const int64_t batches = sizes[0];
  const int64_t attn_heads = sizes[1];
  const int64_t query_seq_len = sizes[2];
  const int64_t key_seq_len = sizes[3];

  NVTE_CHECK(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  NVTE_CHECK(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  NVTE_CHECK(query_seq_len > 1, "Query sequence length must be greater than 1");

  auto softmax_results = allocateStableTensor({batches, attn_heads, query_seq_len, key_seq_len},
                                              input.scalar_type(), input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(),
                              static_cast<float>(scale_factor),
                              getCurrentCUDAStreamRaw(input.get_device_index()));

  return softmax_results;
}

Tensor scaled_softmax_backward(Tensor output_grad, Tensor softmax_results, double scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad);
  softmax_results = torch::stable::contiguous(softmax_results);

  NVTE_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  NVTE_CHECK(softmax_results.dim() == 4, "expected 4D tensor");
  check_fp16_bf16(output_grads, "scaled_softmax_backward");
  check_fp16_bf16(softmax_results, "scaled_softmax_backward");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                               output_grads_cu.data(), static_cast<float>(scale_factor),
                               getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return output_grads;
}

Tensor scaled_masked_softmax_forward(Tensor input, Tensor mask, double scale_factor) {
  NVTE_CHECK(input.dim() == 4, "expected 4D tensor");
  NVTE_CHECK(mask.dim() == 4, "expected 4D tensor");
  check_fp16_bf16(input, "scaled_masked_softmax_forward");
  input = torch::stable::contiguous(input);
  mask = torch::stable::contiguous(mask);

  auto sizes = input.sizes();
  const int64_t batches = sizes[0];
  const int64_t attn_heads = sizes[1];
  const int64_t query_seq_len = sizes[2];
  const int64_t key_seq_len = sizes[3];

  auto mask_sizes = mask.sizes();
  const int64_t pad_batches = mask_sizes[0];
  NVTE_CHECK(pad_batches == 1 || pad_batches == batches,
             "Mask batch dim must be 1 or match input batch dim");
  NVTE_CHECK(mask_sizes[1] == 1, "Mask second dim must be 1");
  NVTE_CHECK(mask_sizes[2] == query_seq_len, "Mask query dim must match input");
  NVTE_CHECK(mask_sizes[3] == key_seq_len, "Mask key dim must match input");

  NVTE_CHECK(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  NVTE_CHECK(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  NVTE_CHECK(query_seq_len > 1, "Query sequence length must be greater than 1");

  auto softmax_results = allocateStableTensor({batches, attn_heads, query_seq_len, key_seq_len},
                                              input.scalar_type(), input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto mask_cu = makeTransformerEngineTensor(mask);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_masked_softmax_forward(input_cu.data(), mask_cu.data(), softmax_results_cu.data(),
                                     static_cast<float>(scale_factor),
                                     getCurrentCUDAStreamRaw(input.get_device_index()));

  return softmax_results;
}

Tensor scaled_masked_softmax_backward(Tensor output_grad, Tensor softmax_results,
                                      double scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad);
  softmax_results = torch::stable::contiguous(softmax_results);

  NVTE_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  NVTE_CHECK(softmax_results.dim() == 4, "expected 4D tensor");
  check_fp16_bf16(output_grads, "scaled_masked_softmax_backward");
  check_fp16_bf16(softmax_results, "scaled_masked_softmax_backward");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                               output_grads_cu.data(), static_cast<float>(scale_factor),
                               getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return output_grads;
}

Tensor scaled_upper_triang_masked_softmax_forward(Tensor input, double scale_factor) {
  NVTE_CHECK(input.dim() == 3, "expected 3D tensor");
  check_fp16_bf16(input, "scaled_upper_triang_masked_softmax_forward");

  auto sizes = input.sizes();
  const int64_t attn_batches = sizes[0];
  const int64_t seq_len = sizes[1];
  NVTE_CHECK(seq_len <= 16384, "Sequence length must be 16384 or less");

  auto softmax_results = allocateStableTensor({attn_batches, seq_len, seq_len}, input.scalar_type(),
                                              input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_upper_triang_masked_softmax_forward(
      input_cu.data(), softmax_results_cu.data(), static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(input.get_device_index()));

  return softmax_results;
}

Tensor scaled_upper_triang_masked_softmax_backward(Tensor output_grads_, Tensor softmax_results_,
                                                   double scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grads_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  NVTE_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  NVTE_CHECK(softmax_results.dim() == 3, "expected 3D tensor");
  check_fp16_bf16(output_grads, "scaled_upper_triang_masked_softmax_backward");
  check_fp16_bf16(softmax_results, "scaled_upper_triang_masked_softmax_backward");
  NVTE_CHECK(output_grads.sizes()[1] == output_grads.sizes()[2],
             "Upper triangular softmax requires square attention matrix");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_upper_triang_masked_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(),
      static_cast<float>(scale_factor), getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return output_grads;
}

Tensor scaled_aligned_causal_masked_softmax_forward(Tensor input, double scale_factor) {
  NVTE_CHECK(input.dim() == 4, "expected 4D tensor");
  check_fp16_bf16(input, "scaled_aligned_causal_masked_softmax_forward");

  auto sizes = input.sizes();
  const int64_t batches = sizes[0];
  const int64_t attn_heads = sizes[1];
  const int64_t query_seq_len = sizes[2];
  const int64_t key_seq_len = sizes[3];

  NVTE_CHECK(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  NVTE_CHECK(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  NVTE_CHECK(query_seq_len >= 1, "Query sequence length must be greater or equal to 1");

  auto softmax_results = allocateStableTensor({batches, attn_heads, query_seq_len, key_seq_len},
                                              input.scalar_type(), input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_aligned_causal_masked_softmax_forward(
      input_cu.data(), softmax_results_cu.data(), static_cast<float>(scale_factor),
      getCurrentCUDAStreamRaw(input.get_device_index()));

  return softmax_results;
}

Tensor scaled_aligned_causal_masked_softmax_backward(Tensor output_grad, Tensor softmax_results_,
                                                     double scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  NVTE_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  NVTE_CHECK(softmax_results.dim() == 4, "expected 4D tensor");
  check_fp16_bf16(output_grads, "scaled_aligned_causal_masked_softmax_backward");
  check_fp16_bf16(softmax_results, "scaled_aligned_causal_masked_softmax_backward");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_aligned_causal_masked_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(),
      static_cast<float>(scale_factor), getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return output_grads;
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_IMPL(transformer_engine, CUDA, m) {
  m.impl("scaled_softmax_forward",
         TORCH_BOX(&transformer_engine::pytorch::stable::scaled_softmax_forward));
  m.impl("scaled_softmax_backward",
         TORCH_BOX(&transformer_engine::pytorch::stable::scaled_softmax_backward));
  m.impl("scaled_masked_softmax_forward",
         TORCH_BOX(&transformer_engine::pytorch::stable::scaled_masked_softmax_forward));
  m.impl("scaled_masked_softmax_backward",
         TORCH_BOX(&transformer_engine::pytorch::stable::scaled_masked_softmax_backward));
  m.impl(
      "scaled_upper_triang_masked_softmax_forward",
      TORCH_BOX(&transformer_engine::pytorch::stable::scaled_upper_triang_masked_softmax_forward));
  m.impl(
      "scaled_upper_triang_masked_softmax_backward",
      TORCH_BOX(&transformer_engine::pytorch::stable::scaled_upper_triang_masked_softmax_backward));
  m.impl("scaled_aligned_causal_masked_softmax_forward",
         TORCH_BOX(
             &transformer_engine::pytorch::stable::scaled_aligned_causal_masked_softmax_forward));
  m.impl("scaled_aligned_causal_masked_softmax_backward",
         TORCH_BOX(
             &transformer_engine::pytorch::stable::scaled_aligned_causal_masked_softmax_backward));
}
