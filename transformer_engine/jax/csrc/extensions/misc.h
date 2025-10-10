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
  NVFP4_1D_SCALING = 4,
  NVFP4_2D_SCALING = 5,
};

inline bool is_tensor_scaling(const JAXX_Scaling_Mode &mode) {
  return (mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING ||
          mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING);
}

inline bool is_block_scaling(const JAXX_Scaling_Mode &mode) {
  return (mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING);
}

inline bool is_nvfp4_scaling(const JAXX_Scaling_Mode &mode) {
  return (mode == JAXX_Scaling_Mode::NVFP4_1D_SCALING ||
          mode == JAXX_Scaling_Mode::NVFP4_2D_SCALING);
}

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
    case JAXX_Scaling_Mode::NVFP4_1D_SCALING:
      return NVTEScalingMode::NVTE_NVFP4_1D_SCALING;
      break;
    case JAXX_Scaling_Mode::NVFP4_2D_SCALING:
      // TE common uses the same enum value for 1D and 2D fp4 scaling and instead differentiates them via quant_config.nvfp4_2d_quantization
      return NVTEScalingMode::NVTE_NVFP4_1D_SCALING;
      break;
    default:
      NVTE_ERROR("Invalid Scaling Mode ", static_cast<int>(mode));
      break;
  }
}

struct BLOCK_SIZE {
  size_t x;
  size_t y;
  constexpr BLOCK_SIZE(int _x, int _y) : x(_x), y(_y) {}
};

constexpr BLOCK_SIZE MXFP8_BLOCK_SIZE{1, 32};
constexpr BLOCK_SIZE NVFP4_BLOCK_SIZE{1, 16};

constexpr BLOCK_SIZE BLOCK_SCALE_ALIGNMENT{128, 4};

std::vector<size_t> get_block_scale_shape(JAXX_Scaling_Mode scaling_mode, size_t M, size_t N,
                                          bool is_colwise);

template <typename T, typename... Rest>
void hash_combine(int64_t &seed, const T &v, Rest... rest) {
  seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  (hash_combine(seed, rest), ...);
}

enum class JAXX_Collective_Op : int64_t {
  NONE = 0,
  ALL_GATHER = 1,
  REDUCE_SCATTER = 2,
};

static CommOverlapType get_nvte_collective_op(const JAXX_Collective_Op &op) {
  switch (op) {
    case JAXX_Collective_Op::ALL_GATHER:
      return CommOverlapType::AG;
      break;
    case JAXX_Collective_Op::REDUCE_SCATTER:
      return CommOverlapType::RS;
      break;
    default:
      NVTE_ERROR("Invalid Collective Op ", static_cast<int>(op));
      break;
  }
}

}  // namespace jax
}  // namespace transformer_engine
