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

  // Tensor dimensions
  const NVTEShape& shape = getTensorShape(input);
  std::vector<int64_t> transpose_shape_int64;
  if (shape.ndim > 0) {
    transpose_shape_int64.push_back(shape.data[shape.ndim - 1]);
    for (size_t i = 0; i < shape.ndim - 1; ++i) {
      transpose_shape_int64.push_back(shape.data[i]);
    }
  }
  const size_t M = shape.ndim > 0 ? product(shape) / shape.data[shape.ndim - 1] : 1;
  const size_t N = shape.ndim > 0 ? shape.data[shape.ndim - 1] : 1;

  // Output tensor
  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    out = at::empty(transpose_shape_int64, opts);
  }

  // Return immediately if tensor is empty
  if (M == 0 || N == 0) {
    return out;
  }

  // Compute transpose
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), make_nvte_2d_shape(M, N), otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), make_nvte_2d_shape(N, M), otype);
  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

at::Tensor swap_first_dims(at::Tensor tensor, std::optional<at::Tensor> out) {
  init_extension();

  // Make sure input is contiguous
  const auto &input = tensor.contiguous();

  // Allocate output tensor if needed
  if (!out) {
    const NVTEShape& in_shape = getTensorShape(input);
    NVTE_CHECK(in_shape.ndim >= 2, "Invalid input tensor dimensions with ", in_shape.ndim, " number of dimensions");
    std::vector<int64_t> out_shape_int64(in_shape.data, in_shape.data + in_shape.ndim);
    out_shape_int64[0] = static_cast<int64_t>(in_shape.data[1]);
    out_shape_int64[1] = static_cast<int64_t>(in_shape.data[0]);
    auto opts = at::TensorOptions().dtype(input.dtype()).device(input.device());
    out = at::empty(out_shape_int64, opts);
  }

  // Launch kernel
  const TensorWrapper te_input = makeTransformerEngineTensor(input);
  TensorWrapper te_output = makeTransformerEngineTensor(*out);
  nvte_swap_first_dims(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());

  return std::move(*out);
}

}  // namespace pytorch
}  // namespace transformer_engine
