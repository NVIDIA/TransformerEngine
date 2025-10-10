/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine {
namespace jax {

std::vector<size_t> MakeShapeVector(NVTEShape shape) {
  return std::vector<size_t>(shape.data, shape.data + shape.ndim);
}

void Shape::from_vector(const std::vector<size_t> &shape) {
  num_dim = shape.size();
  assert(num_dim <= kMaxNumDim);
  std::memcpy(dims, shape.data(), num_dim * sizeof(size_t));
}

std::vector<size_t> Shape::to_vector() const {
  assert(num_dim <= kMaxNumDim);
  std::vector<size_t> shape(num_dim);
  std::memcpy(shape.data(), dims, num_dim * sizeof(size_t));
  return shape;
}

std::vector<size_t> get_block_scale_shape(JAXX_Scaling_Mode scaling_mode, size_t M, size_t N,
                                          bool is_colwise) {
  auto block_size = BLOCK_SIZE(1, 1);
  if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
    block_size = MXFP8_BLOCK_SIZE;
  } else if (scaling_mode == JAXX_Scaling_Mode::NVFP4_1D_SCALING ||
             scaling_mode == JAXX_Scaling_Mode::NVFP4_2D_SCALING) {
    block_size = NVFP4_BLOCK_SIZE;
  } else {
    NVTE_ERROR("Unsupported scaling_mode = ", static_cast<int>(scaling_mode));
  }
  auto block_x = is_colwise ? block_size.y : block_size.x;
  auto block_y = is_colwise ? block_size.x : block_size.y;
  auto alignment_x = is_colwise ? BLOCK_SCALE_ALIGNMENT.y : BLOCK_SCALE_ALIGNMENT.x;
  auto alignment_y = is_colwise ? BLOCK_SCALE_ALIGNMENT.x : BLOCK_SCALE_ALIGNMENT.y;

  NVTE_CHECK(M % block_x == 0, "M must be divisble by %zu (got %zu)", block_x, M);
  NVTE_CHECK(N % block_y == 0, "N must be divisble by %zu (got %zu)", block_y, N);
  size_t scale_x = DIVUP((M / block_x), alignment_x) * alignment_x;
  size_t scale_y = DIVUP((N / block_y), alignment_y) * alignment_y;

  return {scale_x, scale_y};
}

}  // namespace jax
}  // namespace transformer_engine
