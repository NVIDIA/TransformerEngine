/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include <optional>
#include <vector>

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine {
namespace pytorch {

at::Tensor fp8_transpose(at::Tensor input, DType otype, std::optional<at::Tensor> output) {
  init_extension();

  const auto dim = input.dim();
  NVTE_CHECK(dim >= 2, "Need at least 2D tensor to transpose.");

  if (input.dim() > 2) {
    input = input.view({-1, input.size(dim - 1)});
  }

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    out = allocateTorchTensor(input.size(1), input.size(0), DType::kByte);
  }
  if (M == 0 || N == 0) return out;

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), std::vector<size_t>{M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), std::vector<size_t>{N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

at::Tensor swap_first_dims(at::Tensor input, std::optional<at::Tensor> output) {
  init_extension();

  // Make sure input is contiguous
  input = input.contiguous();

  // Allocate output tensor if needed
  if (!output) {
    auto in_shape = getTensorShape(input);
    NVTE_CHECK(in_shape.size() >= 2, "Invalid input tensor dimensions (shape=", in_shape, ")");
    std::vector<int64_t> out_shape_int64(in_shape.begin(), in_shape.end());
    out_shape_int64[0] = static_cast<int64_t>(in_shape[1]);
    out_shape_int64[1] = static_cast<int64_t>(in_shape[0]);
    auto opts = at::TensorOptions().dtype(input.dtype()).device(input.device());
    output = at::empty(out_shape_int64, opts);
  }

  // Launch kernel
  const TensorWrapper te_input = makeTransformerEngineTensor(input);
  TensorWrapper te_output = makeTransformerEngineTensor(*output);
  nvte_swap_first_dims(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());

  return std::move(*output);
}

}  // namespace pytorch
}  // namespace transformer_engine
