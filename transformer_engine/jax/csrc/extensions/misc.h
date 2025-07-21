/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <cassert>
#include <string>
#include <vector>

namespace transformer_engine {
namespace jax {

constexpr int kMaxNumDim = 8;

struct Shape {
  int num_dim;
  size_t dims[kMaxNumDim];

  void from_vector(const std::vector<size_t> &shape);

  std::vector<size_t> to_vector() const;
};

std::vector<size_t> MakeShapeVector(NVTEShape shape);

inline size_t product(const std::vector<size_t> &shape) {
  size_t ret = 1;
  for (const auto &elem : shape) {
    ret *= elem;
  }
  return ret;
}

enum class QuantizeLayout {
  ROWWISE,
  COLWISE,
  ROWWISE_COLWISE,
};

enum class JAXX_Scaling_Mode : int64_t {
  NO_SCALING = 0,
  DELAYED_TENSOR_SCALING = 1,
  MXFP8_1D_SCALING = 2,
  CURRENT_TENSOR_SCALING = 3,
};

static NVTEScalingMode get_nvte_scaling_mode(const JAXX_Scaling_Mode &mode) {
  switch (mode) {
    case JAXX_Scaling_Mode::NO_SCALING:
      return NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING;
      break;
    case JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING:
      return NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING;
      break;
    case JAXX_Scaling_Mode::MXFP8_1D_SCALING:
      return NVTEScalingMode::NVTE_MXFP8_1D_SCALING;
      break;
    case JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING:
      return NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING;
      break;
    default:
      NVTE_ERROR("Invalid Scaling Mode ", static_cast<int>(mode));
      break;
  }
}

constexpr struct BlockSize {
  size_t x;
  size_t y;
} MXFP8_BLOCK_SIZE{1, 32};
constexpr struct Alignment {
  size_t x;
  size_t y;
} MXFP8_ALIGNMENT{128, 4};

std::vector<size_t> get_mxfp8_scale_shape(size_t M, size_t N, bool is_colwise);

}  // namespace jax
}  // namespace transformer_engine
