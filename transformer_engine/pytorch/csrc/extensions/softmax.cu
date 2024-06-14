/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

at::Tensor scaled_softmax_forward(at::Tensor input, float scale_factor) {
  using namespace transformer_engine;
  AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
                 (input.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");

  const int batches = input.size(0);
  const int attn_heads = input.size(1);
  const int query_seq_len = input.size(2);
  const int key_seq_len = input.size(3);

  AT_ASSERTM(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  AT_ASSERTM(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  AT_ASSERTM(query_seq_len > 1, "Query sequence length must be greater than 1");

  // Output
  auto act_options = input.options().requires_grad(false);
  auto softmax_results =
      torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(), scale_factor,
                              at::cuda::getCurrentCUDAStream());

  return softmax_results;
}

at::Tensor scaled_softmax_backward(at::Tensor output_grad_, at::Tensor softmax_results_,
                                   float scale_factor) {
  using namespace transformer_engine;

  auto output_grads = output_grad_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  AT_ASSERTM(output_grads.dim() == 4, "expected 4D tensor");
  AT_ASSERTM(softmax_results.dim() == 4, "expected 4D tensor");

  AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
                 (output_grads.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");
  AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
                 (softmax_results.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                               output_grads_cu.data(), scale_factor,
                               at::cuda::getCurrentCUDAStream());

  return output_grads;
}

at::Tensor scaled_masked_softmax_forward(at::Tensor input, at::Tensor mask, float scale_factor) {
  using namespace transformer_engine;

  AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
                 (input.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");
  AT_ASSERTM(mask.dim() == 4, "expected 4D tensor");
  if (!input.is_contiguous()) input = input.contiguous();
  if (!mask.is_contiguous()) mask = mask.contiguous();

  const int batches = input.size(0);
  const int pad_batches = mask.size(0);
  const int attn_heads = input.size(1);
  const int query_seq_len = input.size(2);
  const int key_seq_len = input.size(3);

  AT_ASSERTM(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  AT_ASSERTM(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  AT_ASSERTM(query_seq_len > 1, "Query sequence length must be greater than 1");
  TORCH_CHECK(pad_batches == 1 || pad_batches == batches);
  TORCH_CHECK(mask.size(1) == 1);
  TORCH_CHECK(mask.size(2) == query_seq_len);
  TORCH_CHECK(mask.size(3) == key_seq_len);

  auto act_options = input.options().requires_grad(false);
  auto softmax_results =
      torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto mask_cu = makeTransformerEngineTensor(mask);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_masked_softmax_forward(input_cu.data(), mask_cu.data(), softmax_results_cu.data(),
                                     scale_factor, at::cuda::getCurrentCUDAStream());

  return softmax_results;
}

at::Tensor scaled_masked_softmax_backward(at::Tensor output_grad_, at::Tensor softmax_results_,
                                          float scale_factor) {
  using namespace transformer_engine;

  auto output_grads = output_grad_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  AT_ASSERTM(output_grads.dim() == 4, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim() == 4, "expected 3D tensor");

  AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
                 (output_grads.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");
  AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
                 (softmax_results.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                               output_grads_cu.data(), scale_factor,
                               at::cuda::getCurrentCUDAStream());

  return output_grads;
}

at::Tensor scaled_upper_triang_masked_softmax_forward(at::Tensor input, float scale_factor) {
  using namespace transformer_engine;

  AT_ASSERTM(input.dim() == 3, "expected 3D tensor");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
                 (input.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");

  const int attn_batches = input.size(0);
  const int seq_len = input.size(1);
  AT_ASSERTM(seq_len <= 16384, "Sequence length must be 16384 or less");

  // Output
  auto act_options = input.options().requires_grad(false);
  auto softmax_results = torch::empty({attn_batches, seq_len, seq_len}, act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_upper_triang_masked_softmax_forward(input_cu.data(), softmax_results_cu.data(),
                                                  scale_factor, at::cuda::getCurrentCUDAStream());

  return softmax_results;
}

at::Tensor scaled_upper_triang_masked_softmax_backward(at::Tensor output_grads_,
                                                       at::Tensor softmax_results_,
                                                       float scale_factor) {
  using namespace transformer_engine;

  auto output_grads = output_grads_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  AT_ASSERTM(output_grads.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim() == 3, "expected 3D tensor");

  AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
                 (output_grads.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");
  AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
                 (softmax_results.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");

  TORCH_CHECK(output_grads.size(1) == output_grads.size(2));

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_upper_triang_masked_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(), scale_factor,
      at::cuda::getCurrentCUDAStream());

  return output_grads;
}

at::Tensor scaled_aligned_causal_masked_softmax_forward(at::Tensor input, float scale_factor) {
  using namespace transformer_engine;
  AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
                 (input.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");

  const int batches = input.size(0);
  const int attn_heads = input.size(1);
  const int query_seq_len = input.size(2);
  const int key_seq_len = input.size(3);

  AT_ASSERTM(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  AT_ASSERTM(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  AT_ASSERTM(query_seq_len >= 1, "Query sequence length must be greater or equal to 1");

  // Output
  auto act_options = input.options().requires_grad(false);
  auto softmax_results =
      torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_aligned_causal_masked_softmax_forward(input_cu.data(), softmax_results_cu.data(),
                                                    scale_factor, at::cuda::getCurrentCUDAStream());

  return softmax_results;
}

at::Tensor scaled_aligned_causal_masked_softmax_backward(at::Tensor output_grad_,
                                                         at::Tensor softmax_results_,
                                                         float scale_factor) {
  using namespace transformer_engine;

  auto output_grads = output_grad_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  AT_ASSERTM(output_grads.dim() == 4, "expected 4D tensor");
  AT_ASSERTM(softmax_results.dim() == 4, "expected 4D tensor");

  AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
                 (output_grads.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");
  AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
                 (softmax_results.scalar_type() == at::ScalarType::BFloat16),
             "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_aligned_causal_masked_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(), scale_factor,
      at::cuda::getCurrentCUDAStream());

  return output_grads;
}
