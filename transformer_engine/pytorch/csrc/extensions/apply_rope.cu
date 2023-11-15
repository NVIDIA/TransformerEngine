/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

at::Tensor fused_rope_forward(const at::Tensor &input_, const at::Tensor &cos_,
                              const at::Tensor &sin_) {
  using namespace transformer_engine;
  auto input = input_.contiguous();
  auto cos = cos_.contiguous();
  auto sin = sin_.contiguous();
  TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(cos.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(sin.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(input.size(0) == cos.size(0),
              "expected input and cos tensor have the same sequence length");
  TORCH_CHECK(input.size(0) == sin.size(0),
              "expected input and sin tensor have the same sequence length");
  TORCH_CHECK(cos.size(1) == 1 && cos.size(2) == 1,
              "expected the second and third dims of the cos tensor equal 1");
  TORCH_CHECK(sin.size(1) == 1 && sin.size(2) == 1,
              "expected the second and third dims of the sin tensor equal 1");
  TORCH_CHECK(input.size(3) >= cos.size(3),
              "expected the last dim of the input tensor is greater than the "
              "cos tensor");
  TORCH_CHECK(input.size(3) >= sin.size(3),
              "expected the last dim of the input tensor is greater than the "
              "sin tensor");

  const int sq = input.size(0);
  const int b = input.size(1);
  const int np = input.size(2);
  const int hn = input.size(3);
  const int hn2 = cos.size(3);

  // Output
  auto act_options = input.options().requires_grad(false);
  auto output = torch::empty({sq, b, np, hn}, act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto output_cu = makeTransformerEngineTensor(output);

  nvte_fused_rope_forward(input_cu.data(), cos_cu.data(), sin_cu.data(),
                          output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor fused_rope_backward(const at::Tensor &incoming_grads_,
                               const at::Tensor &cos_, const at::Tensor &sin_) {
  using namespace transformer_engine;
  auto incoming_grads = incoming_grads_.contiguous();
  auto cos = cos_.contiguous();
  auto sin = sin_.contiguous();
  TORCH_CHECK(incoming_grads.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(cos.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(sin.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(
      incoming_grads.size(0) == cos.size(0),
      "expected incoming_grads and cos tensor have the same sequence length");
  TORCH_CHECK(
      incoming_grads.size(0) == sin.size(0),
      "expected incoming_grads and sin tensor have the same sequence length");
  TORCH_CHECK(cos.size(1) == 1 && cos.size(2) == 1,
              "expected the second and third dims of the cos tensor equal 1");
  TORCH_CHECK(sin.size(1) == 1 && sin.size(2) == 1,
              "expected the second and third dims of the sin tensor equal 1");
  TORCH_CHECK(
      incoming_grads.size(3) >= cos.size(3),
      "expected the last dim of the incoming_grads tensor is greater than the "
      "cos tensor");
  TORCH_CHECK(
      incoming_grads.size(3) >= sin.size(3),
      "expected the last dim of the incoming_grads tensor is greater than the "
      "sin tensor");

  const int sq = incoming_grads.size(0);
  const int b = incoming_grads.size(1);
  const int np = incoming_grads.size(2);
  const int hn = incoming_grads.size(3);
  const int hn2 = cos.size(3);

  // Output grads
  auto act_options = incoming_grads.options().requires_grad(false);
  auto output_grads = torch::empty({sq, b, np, hn}, act_options);

  auto incoming_grads_cu = makeTransformerEngineTensor(incoming_grads);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto output_grads_cu = makeTransformerEngineTensor(output_grads);

  nvte_fused_rope_backward(
      incoming_grads_cu.data(), cos_cu.data(), sin_cu.data(),
      output_grads_cu.data(), at::cuda::getCurrentCUDAStream());

  return output_grads;
}
