/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file activation_template.h
 *  \brief Activation functions template.
 */

#ifndef TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_
#define TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_

#include <cuda_runtime.h>
#include <transformer_engine/activation.h>

#include "../common.h"
#include "../util/cast_gated_kernels.cuh"
#include "../util/cast_kernels.cuh"
#include "../util/math.h"
#include "../util/vectorized_pointwise.h"

namespace transformer_engine {

template <typename ComputeType, typename Param, ComputeType (*OP)(ComputeType, const Param &)>
void act_fn(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  using namespace detail;
  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor grad = nullptr;

  quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, OP>(input, grad, nullptr, output, dbias,
                                                        workspace, stream);
}

template <typename ComputeType, typename Param, ComputeType (*OP)(ComputeType, const Param &)>
void dact_fn(const NVTETensor grad, const NVTETensor input, NVTETensor output,
             cudaStream_t stream) {
  using namespace detail;
  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;

  quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, OP>(input, grad, nullptr, output, dbias,
                                                        workspace, stream);
}

template <typename ComputeType, typename Param, ComputeType (*ActOP)(ComputeType, const Param &)>
void gated_act_fn(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  using namespace detail;
  constexpr bool IS_DGATED = false;
  constexpr NVTETensor grad = nullptr;

  quantize_gated_helper<IS_DGATED, Param, ActOP, nullptr>(grad, input, output, stream);
}

template <typename ComputeType, typename Param, ComputeType (*ActOP)(ComputeType, const Param &),
          ComputeType (*DActOP)(ComputeType, const Param &)>
void dgated_act_fn(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                   cudaStream_t stream) {
  using namespace detail;
  constexpr bool IS_DGATED = true;

  quantize_gated_helper<IS_DGATED, Param, ActOP, DActOP>(grad, input, output, stream);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_
