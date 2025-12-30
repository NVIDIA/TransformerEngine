/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include <optional>
#include <vector>

#include <transformer_engine/recipe.h>

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine {
namespace pytorch {

at::Tensor fp8_transpose(at::Tensor input, DType otype, std::optional<at::Tensor> output) {
  init_extension();

  // Tensor dimensions
  const auto shape = getTensorShape(input);
  std::vector<int64_t> transpose_shape_int64;
  if (shape.size() > 0) {
    transpose_shape_int64.push_back(shape.back());
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      transpose_shape_int64.push_back(shape[i]);
    }
  }
  const size_t M = shape.size() > 0 ? product(shape) / shape.back() : 1;
  const size_t N = shape.size() > 0 ? shape.back() : 1;

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
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), std::vector<size_t>{M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), std::vector<size_t>{N, M}, otype);
  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

at::Tensor nvfp4_transpose(at::Tensor input, std::optional<at::Tensor> output) {
  init_extension();

  // Input is packed FP4: logical [M, K] stored as [M, K/2] bytes
  // Output is packed FP4: logical [K, M] stored as [K, M/2] bytes
  const auto shape = getTensorShape(input);
  NVTE_CHECK(shape.size() == 2, "NVFP4 transpose expects 2D input (packed storage).");

  const size_t M = shape[0];
  const size_t K_packed = shape[1];
  const size_t K = K_packed * 2;  // logical K
  const size_t M_packed = M / 2;

  NVTE_CHECK(M % 2 == 0, "NVFP4 transpose requires M (", M, ") to be even.");

  // Output shape: [K, M/2]
  std::vector<int64_t> output_shape = {static_cast<int64_t>(K), static_cast<int64_t>(M_packed)};

  // Output tensor
  at::Tensor out;
  if (output.has_value()) {
    out = *output;
    NVTE_CHECK(static_cast<size_t>(out.size(0)) == K &&
                   static_cast<size_t>(out.size(1)) == M_packed,
               "Output shape mismatch for NVFP4 transpose.");
  } else {
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    out = at::empty(output_shape, opts);
  }

  // Return immediately if tensor is empty
  if (M == 0 || K == 0) {
    return out;
  }

  // Call the NVFP4 transpose kernel
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), std::vector<size_t>{M, K_packed},
                                              DType::kByte);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), std::vector<size_t>{K, M_packed},
                                               DType::kByte);
  nvte_nvfp4_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

void nvfp4_scale_transpose(at::Tensor input, at::Tensor output,
                           int64_t M_tiles, int64_t K_tiles) {
  init_extension();

  // Input: rowwise_scale_inv [M_padded, K_tiles], uint8 (E4M3 stored as bytes)
  // Output: columnwise_scale_inv [K_padded, M_tiles], uint8 (E4M3 stored as bytes)
  const auto in_shape = getTensorShape(input);
  const auto out_shape = getTensorShape(output);
  NVTE_CHECK(in_shape.size() == 2, "NVFP4 scale transpose expects 2D input.");
  NVTE_CHECK(out_shape.size() == 2, "NVFP4 scale transpose expects 2D output.");
  NVTE_CHECK(input.scalar_type() == at::kByte, "NVFP4 scale transpose input must be uint8 (E4M3).");
  NVTE_CHECK(output.scalar_type() == at::kByte, "NVFP4 scale transpose output must be uint8 (E4M3).");

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(), std::vector<size_t>{in_shape[0], in_shape[1]}, DType::kByte);
  auto output_cu = makeTransformerEngineTensor(
      output.data_ptr(), std::vector<size_t>{out_shape[0], out_shape[1]}, DType::kByte);

  nvte_nvfp4_scale_transpose(input_cu.data(), output_cu.data(),
                             static_cast<size_t>(M_tiles), static_cast<size_t>(K_tiles),
                             at::cuda::getCurrentCUDAStream());
}

at::Tensor swap_first_dims(at::Tensor tensor, std::optional<at::Tensor> out) {
  init_extension();

  // Make sure input is contiguous
  const auto &input = tensor.contiguous();

  // Allocate output tensor if needed
  if (!out) {
    auto in_shape = getTensorShape(input);
    NVTE_CHECK(in_shape.size() >= 2, "Invalid input tensor dimensions (shape=", in_shape, ")");
    std::vector<int64_t> out_shape_int64(in_shape.begin(), in_shape.end());
    out_shape_int64[0] = static_cast<int64_t>(in_shape[1]);
    out_shape_int64[1] = static_cast<int64_t>(in_shape[0]);
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
