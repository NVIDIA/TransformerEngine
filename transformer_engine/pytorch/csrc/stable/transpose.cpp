/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>
#include <transformer_engine/transpose.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

Tensor fp8_transpose(Tensor input, int64_t otype, std::optional<Tensor> output) {
  auto shape = getStableTensorShape(input);
  auto te_otype = static_cast<DType>(otype);

  std::vector<int64_t> transpose_shape_int64;
  if (!shape.empty()) {
    transpose_shape_int64.push_back(static_cast<int64_t>(shape.back()));
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      transpose_shape_int64.push_back(static_cast<int64_t>(shape[i]));
    }
  }
  const size_t M = shape.empty() ? 1 : (shape.size() > 1 ? 1 : shape[0]);
  size_t total = 1;
  for (auto s : shape) total *= s;
  const size_t N = shape.empty() ? 1 : shape.back();
  const size_t M_actual = shape.empty() ? 1 : total / N;

  Tensor out;
  if (output.has_value()) {
    out = output.value();
  } else {
    out = allocateStableTensor(transpose_shape_int64, ScalarType::Byte, input.get_device_index());
  }

  if (M_actual == 0 || N == 0) return out;

  auto input_cu =
      makeTransformerEngineTensor(input.data_ptr(), std::vector<size_t>{M_actual, N}, te_otype);
  auto output_cu =
      makeTransformerEngineTensor(out.data_ptr(), std::vector<size_t>{N, M_actual}, te_otype);
  nvte_transpose(input_cu.data(), output_cu.data(),
                 getCurrentCUDAStreamRaw(input.get_device_index()));

  return out;
}

Tensor nvfp4_data_transpose(Tensor input, std::optional<Tensor> output) {
  auto shape = getStableTensorShape(input);
  NVTE_CHECK(shape.size() == 2, "NVFP4 transpose expects 2D input.");

  const size_t M = shape[0];
  const size_t K_packed = shape[1];
  const size_t K = K_packed * 2;
  const size_t M_packed = M / 2;
  NVTE_CHECK(M % 2 == 0, "NVFP4 transpose requires M to be even.");

  Tensor out;
  if (output.has_value()) {
    out = output.value();
  } else {
    out = allocateStableTensor({static_cast<int64_t>(K), static_cast<int64_t>(M_packed)},
                               ScalarType::Byte, input.get_device_index());
  }

  if (M == 0 || K == 0) return out;

  auto input_cu =
      makeTransformerEngineTensor(input.data_ptr(), std::vector<size_t>{M, K_packed}, DType::kByte);
  auto output_cu =
      makeTransformerEngineTensor(out.data_ptr(), std::vector<size_t>{K, M_packed}, DType::kByte);
  nvte_nvfp4_data_transpose(input_cu.data(), output_cu.data(),
                            getCurrentCUDAStreamRaw(input.get_device_index()));

  return out;
}

void nvfp4_2d_scale_transpose(Tensor input, Tensor output, int64_t M_tiles, int64_t K_tiles) {
  auto in_shape = getStableTensorShape(input);
  auto out_shape = getStableTensorShape(output);

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), in_shape, DType::kByte);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), out_shape, DType::kByte);

  nvte_nvfp4_scale_transpose(input_cu.data(), output_cu.data(), static_cast<size_t>(M_tiles),
                             static_cast<size_t>(K_tiles),
                             getCurrentCUDAStreamRaw(input.get_device_index()));
}

void nvfp4_expand_scale_to_fp8(Tensor input, Tensor output, int64_t tile_rows, int64_t tile_cols,
                               int64_t rows_padded, int64_t block_len) {
  auto in_shape = getStableTensorShape(input);
  auto out_shape = getStableTensorShape(output);

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), in_shape, DType::kFloat32);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), out_shape, DType::kByte);

  nvte_nvfp4_expand_scale_to_fp8(input_cu.data(), output_cu.data(), static_cast<size_t>(tile_rows),
                                 static_cast<size_t>(tile_cols), static_cast<size_t>(rows_padded),
                                 static_cast<size_t>(block_len),
                                 getCurrentCUDAStreamRaw(input.get_device_index()));
}

void nvfp4_compute_per_block_scale(Tensor block_amax, Tensor scale, Tensor global_amax) {
  auto block_amax_cu = makeTransformerEngineTensor(block_amax);
  auto scale_cu = makeTransformerEngineTensor(scale);
  auto global_amax_cu = makeTransformerEngineTensor(global_amax);

  nvte_nvfp4_compute_per_block_scale(block_amax_cu.data(), scale_cu.data(), global_amax_cu.data(),
                                     getCurrentCUDAStreamRaw(block_amax.get_device_index()));
}

void nvfp4_fused_scale(Tensor block_amax, Tensor global_amax, Tensor per_block_scale,
                       Tensor target_scale, Tensor target_amax, int64_t tile_rows,
                       int64_t tile_cols, int64_t rows_padded, int64_t block_len) {
  auto block_amax_cu = makeTransformerEngineTensor(block_amax);
  auto global_amax_cu = makeTransformerEngineTensor(global_amax);
  auto per_block_scale_cu = makeTransformerEngineTensor(per_block_scale);
  auto target_scale_cu = makeTransformerEngineTensor(target_scale);
  auto target_amax_cu = makeTransformerEngineTensor(target_amax);

  nvte_nvfp4_fused_scale(block_amax_cu.data(), global_amax_cu.data(), per_block_scale_cu.data(),
                         target_scale_cu.data(), target_amax_cu.data(),
                         static_cast<size_t>(tile_rows), static_cast<size_t>(tile_cols),
                         static_cast<size_t>(rows_padded), static_cast<size_t>(block_len),
                         getCurrentCUDAStreamRaw(block_amax.get_device_index()));
}

void nvfp4_compute_global_scale(Tensor global_amax, Tensor global_scale) {
  auto global_amax_cu = makeTransformerEngineTensor(global_amax);
  auto global_scale_cu = makeTransformerEngineTensor(global_scale);

  nvte_nvfp4_compute_global_scale(global_amax_cu.data(), global_scale_cu.data(),
                                  getCurrentCUDAStreamRaw(global_amax.get_device_index()));
}

Tensor swap_first_dims(Tensor tensor, std::optional<Tensor> out) {
  auto input = torch::stable::contiguous(tensor);
  auto shape = getStableTensorShape(input);
  NVTE_CHECK(shape.size() >= 2, "Invalid input tensor dimensions.");

  if (!out.has_value()) {
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[0] = static_cast<int64_t>(shape[1]);
    out_shape[1] = static_cast<int64_t>(shape[0]);
    out = allocateStableTensor(out_shape, input.scalar_type(), input.get_device_index());
  }

  auto te_input = makeTransformerEngineTensor(input);
  auto te_output = makeTransformerEngineTensor(out.value());
  nvte_swap_first_dims(te_input.data(), te_output.data(),
                       getCurrentCUDAStreamRaw(input.get_device_index()));

  return out.value();
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def("fp8_transpose(Tensor input, int otype, Tensor? output) -> Tensor");
  m.def("nvfp4_data_transpose(Tensor input, Tensor? output) -> Tensor");
  m.def("nvfp4_2d_scale_transpose(Tensor input, Tensor output, int M_tiles, int K_tiles) -> ()");
  m.def(
      "nvfp4_expand_scale_to_fp8(Tensor input, Tensor output, int tile_rows, int tile_cols, int "
      "rows_padded, int block_len) -> ()");
  m.def("nvfp4_compute_per_block_scale(Tensor block_amax, Tensor scale, Tensor global_amax) -> ()");
  m.def(
      "nvfp4_fused_scale(Tensor block_amax, Tensor global_amax, Tensor per_block_scale, Tensor "
      "target_scale, Tensor target_amax, int tile_rows, int tile_cols, int rows_padded, int "
      "block_len) -> ()");
  m.def("nvfp4_compute_global_scale(Tensor global_amax, Tensor global_scale) -> ()");
  m.def("swap_first_dims(Tensor tensor, Tensor? out) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("fp8_transpose", TORCH_BOX(fp8_transpose));
  m.impl("nvfp4_data_transpose", TORCH_BOX(nvfp4_data_transpose));
  m.impl("nvfp4_2d_scale_transpose", TORCH_BOX(nvfp4_2d_scale_transpose));
  m.impl("nvfp4_expand_scale_to_fp8", TORCH_BOX(nvfp4_expand_scale_to_fp8));
  m.impl("nvfp4_compute_per_block_scale", TORCH_BOX(nvfp4_compute_per_block_scale));
  m.impl("nvfp4_fused_scale", TORCH_BOX(nvfp4_fused_scale));
  m.impl("nvfp4_compute_global_scale", TORCH_BOX(nvfp4_compute_global_scale));
  m.impl("swap_first_dims", TORCH_BOX(swap_first_dims));
}
