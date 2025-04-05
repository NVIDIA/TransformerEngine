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

NVTEScalingMode jaxScalingModeToNVTEScalingMode(JAXScalingMode scaling_mode) {
  switch (scaling_mode) {
    case JAXScalingMode::NO_SCALING:
      return NVTE_NO_SCALING;
    case JAXScalingMode::DELAYED_TENSOR_SCALING:
      return NVTE_DELAYED_TENSOR_SCALING;
    case JAXScalingMode::CURRENT_TENSOR_SCALING:
      return NVTE_DELAYED_TENSOR_SCALING;
    case JAXScalingMode::MXFP8_1D_SCALING:
      return NVTE_MXFP8_1D_SCALING;
    default:
      assert(false && "Invalid JAX scaling mode");
      return NVTE_INVALID_SCALING;
  }
}

}  // namespace jax
}  // namespace transformer_engine
