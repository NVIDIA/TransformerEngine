/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

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

std::vector<size_t> get_mxfp8_scale_shape(size_t M, size_t N, bool is_colwise) {
  auto block_x = is_colwise ? MXFP8_BLOCK_SIZE.y : MXFP8_BLOCK_SIZE.x;
  auto block_y = is_colwise ? MXFP8_BLOCK_SIZE.x : MXFP8_BLOCK_SIZE.y;
  auto alignment_x = is_colwise ? MXFP8_ALIGNMENT.y : MXFP8_ALIGNMENT.x;
  auto alignment_y = is_colwise ? MXFP8_ALIGNMENT.x : MXFP8_ALIGNMENT.y;

  NVTE_CHECK(M % block_x == 0, "M must be divisble by %zu (got %zu)", block_x, M);
  NVTE_CHECK(N % block_y == 0, "N must be divisble by %zu (got %zu)", block_y, N);
  size_t scale_x = DIVUP((M / block_x), alignment_x) * alignment_x;
  size_t scale_y = DIVUP((N / block_y), alignment_y) * alignment_y;

  return {scale_x, scale_y};
}

}  // namespace jax
}  // namespace transformer_engine
