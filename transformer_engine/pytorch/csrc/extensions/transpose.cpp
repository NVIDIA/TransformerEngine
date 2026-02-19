/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void nvfp4_expand_scale_to_fp8(at::Tensor input, at::Tensor output,
                               int64_t tile_rows, int64_t tile_cols,
                               int64_t rows_padded, int64_t block_len) {
  init_extension();

  // Input: per_block_decode_scale [tile_rows, tile_cols], float32
  // Output: target_scale [rows_padded, tile_cols], uint8 (E4M3)
  const auto in_shape = getTensorShape(input);
  const auto out_shape = getTensorShape(output);
  NVTE_CHECK(in_shape.size() == 2, "NVFP4 expand scale expects 2D input.");
  NVTE_CHECK(out_shape.size() == 2, "NVFP4 expand scale expects 2D output.");
  NVTE_CHECK(input.scalar_type() == at::kFloat, "NVFP4 expand scale input must be float32.");
  NVTE_CHECK(output.scalar_type() == at::kByte, "NVFP4 expand scale output must be uint8 (E4M3).");

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(), std::vector<size_t>{in_shape[0], in_shape[1]}, DType::kFloat32);
  auto output_cu = makeTransformerEngineTensor(
      output.data_ptr(), std::vector<size_t>{out_shape[0], out_shape[1]}, DType::kByte);

  nvte_nvfp4_expand_scale_to_fp8(input_cu.data(), output_cu.data(),
                                 static_cast<size_t>(tile_rows),
                                 static_cast<size_t>(tile_cols),
                                 static_cast<size_t>(rows_padded),
                                 static_cast<size_t>(block_len),
                                 at::cuda::getCurrentCUDAStream());
}

void nvfp4_compute_per_block_scale(at::Tensor block_amax, at::Tensor scale, at::Tensor global_amax) {
  init_extension();

  // block_amax and scale: [tile_rows, tile_cols], float32
  // global_amax: single element tensor, float32 (avoids D2H transfer)
  NVTE_CHECK(block_amax.scalar_type() == at::kFloat, "Block amax must be float32.");
  NVTE_CHECK(scale.scalar_type() == at::kFloat, "Scale must be float32.");
  NVTE_CHECK(global_amax.scalar_type() == at::kFloat, "Global amax must be float32.");
  NVTE_CHECK(global_amax.numel() == 1, "Global amax must be a single element tensor.");

  auto block_amax_cu = makeTransformerEngineTensor(block_amax);
  auto scale_cu = makeTransformerEngineTensor(scale);
  auto global_amax_cu = makeTransformerEngineTensor(global_amax);

  nvte_nvfp4_compute_per_block_scale(block_amax_cu.data(), scale_cu.data(),
                                     global_amax_cu.data(), at::cuda::getCurrentCUDAStream());
}

void nvfp4_fused_scale(at::Tensor block_amax, at::Tensor global_amax,
                       at::Tensor per_block_scale, at::Tensor target_scale,
                       at::Tensor target_amax,
                       int64_t tile_rows, int64_t tile_cols,
                       int64_t rows_padded, int64_t block_len) {
  init_extension();

  // block_amax: [tile_rows, tile_cols], float32
  // global_amax: [1], float32
  // per_block_scale: [tile_rows, tile_cols], float32 (for partial_cast)
  // target_scale: [rows_padded, tile_cols], uint8 (E4M3)
  // target_amax: [1], float32
  NVTE_CHECK(block_amax.scalar_type() == at::kFloat, "Block amax must be float32.");
  NVTE_CHECK(global_amax.scalar_type() == at::kFloat, "Global amax must be float32.");
  NVTE_CHECK(per_block_scale.scalar_type() == at::kFloat, "Per-block scale must be float32.");
  NVTE_CHECK(target_scale.scalar_type() == at::kByte, "Target scale must be uint8 (E4M3).");
  NVTE_CHECK(target_amax.scalar_type() == at::kFloat, "Target amax must be float32.");
  NVTE_CHECK(global_amax.numel() == 1, "Global amax must be a single element tensor.");
  NVTE_CHECK(target_amax.numel() == 1, "Target amax must be a single element tensor.");

  auto block_amax_cu = makeTransformerEngineTensor(block_amax);
  auto global_amax_cu = makeTransformerEngineTensor(global_amax);
  auto per_block_scale_cu = makeTransformerEngineTensor(per_block_scale);
  auto target_scale_cu = makeTransformerEngineTensor(target_scale);
  auto target_amax_cu = makeTransformerEngineTensor(target_amax);

  nvte_nvfp4_fused_scale(block_amax_cu.data(), global_amax_cu.data(),
                         per_block_scale_cu.data(), target_scale_cu.data(),
                         target_amax_cu.data(),
                         static_cast<size_t>(tile_rows), static_cast<size_t>(tile_cols),
                         static_cast<size_t>(rows_padded), static_cast<size_t>(block_len),
                         at::cuda::getCurrentCUDAStream());
}

void nvfp4_multi_tensor_fused_scale(
    std::vector<at::Tensor> block_amax_list,
    std::vector<at::Tensor> global_amax_list,
    std::vector<at::Tensor> per_block_scale_list,
    std::vector<at::Tensor> target_scale_list,
    std::vector<at::Tensor> target_amax_list,
    std::vector<int64_t> tile_rows_list,
    std::vector<int64_t> tile_cols_list,
    std::vector<int64_t> rows_padded_list,
    int64_t block_len) {
  init_extension();

  const size_t num_tensors = block_amax_list.size();
  NVTE_CHECK(global_amax_list.size() == num_tensors, "global_amax_list size mismatch");
  NVTE_CHECK(per_block_scale_list.size() == num_tensors, "per_block_scale_list size mismatch");
  NVTE_CHECK(target_scale_list.size() == num_tensors, "target_scale_list size mismatch");
  NVTE_CHECK(target_amax_list.size() == num_tensors, "target_amax_list size mismatch");
  NVTE_CHECK(tile_rows_list.size() == num_tensors, "tile_rows_list size mismatch");
  NVTE_CHECK(tile_cols_list.size() == num_tensors, "tile_cols_list size mismatch");
  NVTE_CHECK(rows_padded_list.size() == num_tensors, "rows_padded_list size mismatch");

  if (num_tensors == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  for (size_t i = 0; i < num_tensors; ++i) {
    const auto& block_amax = block_amax_list[i];
    const auto& global_amax = global_amax_list[i];
    auto& per_block_scale = per_block_scale_list[i];
    auto& target_scale = target_scale_list[i];
    auto& target_amax = target_amax_list[i];
    const size_t tile_rows = static_cast<size_t>(tile_rows_list[i]);
    const size_t tile_cols = static_cast<size_t>(tile_cols_list[i]);
    const size_t rows_padded = static_cast<size_t>(rows_padded_list[i]);

    NVTE_CHECK(block_amax.scalar_type() == at::kFloat, "Block amax must be float32.");
    NVTE_CHECK(global_amax.scalar_type() == at::kFloat, "Global amax must be float32.");
    NVTE_CHECK(per_block_scale.scalar_type() == at::kFloat, "Per-block scale must be float32.");
    NVTE_CHECK(target_scale.scalar_type() == at::kByte, "Target scale must be uint8 (E4M3).");
    NVTE_CHECK(target_amax.scalar_type() == at::kFloat, "Target amax must be float32.");
    NVTE_CHECK(global_amax.numel() == 1, "Global amax must be a single element tensor.");
    NVTE_CHECK(target_amax.numel() == 1, "Target amax must be a single element tensor.");

    auto block_amax_cu = makeTransformerEngineTensor(block_amax);
    auto global_amax_cu = makeTransformerEngineTensor(global_amax);
    auto per_block_scale_cu = makeTransformerEngineTensor(per_block_scale);
    auto target_scale_cu = makeTransformerEngineTensor(target_scale);
    auto target_amax_cu = makeTransformerEngineTensor(target_amax);

    nvte_nvfp4_fused_scale(block_amax_cu.data(), global_amax_cu.data(),
                           per_block_scale_cu.data(), target_scale_cu.data(),
                           target_amax_cu.data(),
                           tile_rows, tile_cols, rows_padded,
                           static_cast<size_t>(block_len), stream);
  }
}

void nvfp4_compute_global_scale(at::Tensor global_amax, at::Tensor global_scale) {
  init_extension();

  // global_amax and global_scale: [num_params], float32
  NVTE_CHECK(global_amax.scalar_type() == at::kFloat, "Global amax must be float32.");
  NVTE_CHECK(global_scale.scalar_type() == at::kFloat, "Global scale must be float32.");

  auto global_amax_cu = makeTransformerEngineTensor(global_amax);
  auto global_scale_cu = makeTransformerEngineTensor(global_scale);

  nvte_nvfp4_compute_global_scale(global_amax_cu.data(), global_scale_cu.data(),
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

void nvfp4_multi_tensor_create_columnwise(
    std::vector<at::Tensor> rowwise_data_list,
    std::vector<at::Tensor> columnwise_data_list,
    std::vector<at::Tensor> rowwise_scale_inv_list,
    std::vector<at::Tensor> columnwise_scale_inv_list,
    std::vector<int64_t> M_list,
    std::vector<int64_t> K_list) {
  init_extension();

  const size_t num_tensors = rowwise_data_list.size();
  NVTE_CHECK(columnwise_data_list.size() == num_tensors, "Tensor list size mismatch");
  NVTE_CHECK(rowwise_scale_inv_list.size() == num_tensors, "Tensor list size mismatch");
  NVTE_CHECK(columnwise_scale_inv_list.size() == num_tensors, "Tensor list size mismatch");
  NVTE_CHECK(M_list.size() == num_tensors, "M_list size mismatch");
  NVTE_CHECK(K_list.size() == num_tensors, "K_list size mismatch");

  if (num_tensors == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  // Process each tensor - the main benefit is reduced Python overhead
  // by doing the iteration in C++ rather than Python
  constexpr size_t TILE_SIZE = 16;

  for (size_t i = 0; i < num_tensors; ++i) {
    const auto& rowwise_data = rowwise_data_list[i];
    auto& columnwise_data = columnwise_data_list[i];
    const auto& rowwise_scale_inv = rowwise_scale_inv_list[i];
    auto& columnwise_scale_inv = columnwise_scale_inv_list[i];
    const int64_t M = M_list[i];
    const int64_t K = K_list[i];

    // Transpose data: [M, K/2] -> [K, M/2]
    const auto data_shape = getTensorShape(rowwise_data);
    NVTE_CHECK(data_shape.size() == 2, "NVFP4 data must be 2D.");
    const size_t M_packed = static_cast<size_t>(M) / 2;
    const size_t K_packed = data_shape[1];

    auto input_cu = makeTransformerEngineTensor(
        rowwise_data.data_ptr(), std::vector<size_t>{static_cast<size_t>(M), K_packed},
        DType::kByte);
    auto output_cu = makeTransformerEngineTensor(
        columnwise_data.data_ptr(), std::vector<size_t>{static_cast<size_t>(K), M_packed},
        DType::kByte);
    nvte_nvfp4_transpose(input_cu.data(), output_cu.data(), stream);

    // Transpose scales
    const size_t M_tiles = (static_cast<size_t>(M) + TILE_SIZE - 1) / TILE_SIZE;
    const size_t K_tiles = (static_cast<size_t>(K) + TILE_SIZE - 1) / TILE_SIZE;

    const auto scale_in_shape = getTensorShape(rowwise_scale_inv);
    const auto scale_out_shape = getTensorShape(columnwise_scale_inv);

    auto scale_input_cu = makeTransformerEngineTensor(
        rowwise_scale_inv.data_ptr(),
        std::vector<size_t>{scale_in_shape[0], scale_in_shape[1]}, DType::kByte);
    auto scale_output_cu = makeTransformerEngineTensor(
        columnwise_scale_inv.data_ptr(),
        std::vector<size_t>{scale_out_shape[0], scale_out_shape[1]}, DType::kByte);

    nvte_nvfp4_scale_transpose(scale_input_cu.data(), scale_output_cu.data(),
                               M_tiles, K_tiles, stream);
  }
}

}  // namespace pytorch
}  // namespace transformer_engine
