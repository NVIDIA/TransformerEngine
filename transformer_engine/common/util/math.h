/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_MATH_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_MATH_H_

namespace transformer_engine {
namespace {

template <typename OType, typename IType>
__device__ inline OType gelu(const IType val) {
    const float cval = val;
    return cval * (0.5F + 0.5F * tanhf(cval * (0.79788456F + 0.03567741F * cval * cval)));
}

template <typename OType, typename IType>
__device__ inline OType dgelu(const IType val) {
    const float cval = val;
    const float tanh_out = tanhf(0.79788456f * cval * (1.f + 0.044715f * cval * cval));
    return 0.5f * cval * ((1.f - tanh_out * tanh_out) *
                          (0.79788456f + 0.1070322243f * cval * cval)) +
           0.5f * (1.f + tanh_out);
}

}  // namespace
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_MATH_H_
