/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_UTILS_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_UTILS_H_

#include "math.h"

namespace transformer_engine {
namespace cutedsl_backend {

// Encodes the CUDA activation function selected by the dispatcher as the enum and string,
// forwarded to python to compile CuTeDSL kernels.
enum class Activation {
  kNone,
  kReLU,
  kGeLU,
  kSiLU,
  kQGeLU,
  kSReLU,
  kDReLU,
  kDGeLU,
  kDSiLU,
  kDQGeLU,
  kDSReLU,
  kUnsupported,
  kNumTypes
};

inline const char *activation_to_str(Activation act) {
  switch (act) {
    case Activation::kReLU:
      return "relu";
    case Activation::kGeLU:
      return "gelu";
    case Activation::kSiLU:
      return "silu";
    case Activation::kQGeLU:
      return "qgelu";
    case Activation::kSReLU:
      return "srelu";
    case Activation::kDReLU:
      return "drelu";
    case Activation::kDGeLU:
      return "dgelu";
    case Activation::kDSiLU:
      return "dsilu";
    case Activation::kDQGeLU:
      return "dqgelu";
    case Activation::kDSReLU:
      return "dsrelu";
    case Activation::kUnsupported:
      return "unsupported";
    case Activation::kNone:
    case Activation::kNumTypes:
      return "none";
  }
  return "none";
}

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr Activation activation_func_to_enum() {
  return Activation::kUnsupported;
}

#define NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(OP, ACTIVATION)     \
  template <>                                                               \
  constexpr Activation activation_func_to_enum<Empty, OP<float, float>>() { \
    return Activation::ACTIVATION;                                          \
  }

template <>
constexpr Activation activation_func_to_enum<Empty, nullptr>() {
  return Activation::kNone;
}

NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(relu, kReLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(gelu, kGeLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(silu, kSiLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(qgelu, kQGeLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(srelu, kSReLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(drelu, kDReLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(dgelu, kDGeLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(dsilu, kDSiLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(dqgelu, kDQGeLU)
NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM(dsrelu, kDSReLU)

#undef NVTE_SPECIALIZE_CUTEDSL_ACTIVATION_FUNC_TO_ENUM

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_UTILS_H_
