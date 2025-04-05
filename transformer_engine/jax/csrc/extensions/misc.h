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

enum class QuantizeAxis {
  ROWWISE,
  COLWISE,
  ROWWISE_COLWISE,
};


enum class JAXScalingMode {
  // Note: these start at 100 to avoid conflicting with NVTEScalingMode. These are distinct as what the information the common kernels
  // need versus the framework extensions is different. For example, the TE common kernels treat delayed scaling and current scaling the same and both are marked as NVTE_DELAYED_TENSOR_SCALING, it is up to the frameworks to handle current scaling differently, such as calculating the amax beforehand and passing nullptr as the amax_ptr in nvte_quantize to prevent recalculating the fused amax.
  INVALID_SCALING = 100,
  NO_SCALING = 101,
  DELAYED_TENSOR_SCALING = 102,
  CURRENT_TENSOR_SCALING = 103,
  MXFP8_1D_SCALING = 104,
};

NVTEScalingMode jaxScalingModeToNVTEScalingMode(JAXScalingMode scaling_mode);

}  // namespace jax
}  // namespace transformer_engine
