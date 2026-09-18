/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_UTILS_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_UTILS_H_

#include <type_traits>

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
  if constexpr (OP == nullptr) {
    return Activation::kNone;
  } else if constexpr (std::is_same_v<ParamOP, Empty>) {
    if constexpr (OP == relu<float, float>) {
      return Activation::kReLU;
    } else if constexpr (OP == gelu<float, float>) {
      return Activation::kGeLU;
    } else if constexpr (OP == silu<float, float>) {
      return Activation::kSiLU;
    } else if constexpr (OP == qgelu<float, float>) {
      return Activation::kQGeLU;
    } else if constexpr (OP == srelu<float, float>) {
      return Activation::kSReLU;
    } else if constexpr (OP == drelu<float, float>) {
      return Activation::kDReLU;
    } else if constexpr (OP == dgelu<float, float>) {
      return Activation::kDGeLU;
    } else if constexpr (OP == dsilu<float, float>) {
      return Activation::kDSiLU;
    } else if constexpr (OP == dqgelu<float, float>) {
      return Activation::kDQGeLU;
    } else if constexpr (OP == dsrelu<float, float>) {
      return Activation::kDSReLU;
    }
  }
  return Activation::kUnsupported;
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_UTILS_H_
