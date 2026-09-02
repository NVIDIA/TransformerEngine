/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_group_scaled_swiglu(const NVTEGroupedTensor input, const NVTETensor prob,
                              NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_scaled_swiglu);
  using namespace transformer_engine;
  // Scaled SwiGLU recompute: (silu(act) * gate) * prob -> columnwise MXFP8.
  // The kernel halves prob, so the activation is the doubled approximate SiLU.
  Empty e = {};
  dispatch::group_scaled_swiglu_fwd_helper<Empty, silu_approx_x2<fp32, fp32>>(input, prob, output,
                                                                              e, nullptr, stream);
}
