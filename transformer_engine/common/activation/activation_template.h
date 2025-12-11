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

#include "../cast/dispatch/gated.cuh"
#include "../cast/dispatch/quantize.cuh"
#include "../common.h"

namespace transformer_engine {

template <typename ComputeType, typename Param, ComputeType (*OP)(ComputeType, const Param &)>
void act_fn(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  using namespace detail;
  constexpr bool IS_ACT = true;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, OP>(input, output, nullptr, stream);
}

template <typename ComputeType, typename Param, ComputeType (*OP)(ComputeType, const Param &)>
void dact_fn(const NVTETensor grad, const NVTETensor input, NVTETensor output,
             cudaStream_t stream) {
  using namespace detail;
  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, OP>(grad, input, output, dbias, workspace,
                                                              nullptr, stream);
}

template <typename ComputeType, typename Param, ComputeType (*ActOP)(ComputeType, const Param &)>
void gated_act_fn(const NVTETensor input, NVTETensor output, Param &p, cudaStream_t stream) {
  using namespace detail;
  dispatch::quantize_gated_fwd_helper<Param, ActOP>(input, output, p, stream);
}

template <typename ComputeType, typename Param, ComputeType (*ActOP)(ComputeType, const Param &),
          ComputeType (*DActOP)(ComputeType, const Param &)>
void dgated_act_fn(const NVTETensor grad, const NVTETensor input, NVTETensor output, Param &p,
                   cudaStream_t stream) {
  using namespace detail;
  dispatch::quantize_gated_bwd_helper<Param, ActOP, DActOP>(grad, input, output, p, stream);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_
