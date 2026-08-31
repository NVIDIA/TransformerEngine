/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_group_scaled_clamped_swiglu(const NVTEGroupedTensor input, const NVTETensor prob,
                                      NVTEGroupedTensor output, float limit, float alpha,
                                      float glu_linear_offset, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_scaled_clamped_swiglu);
  using namespace transformer_engine;
  // glu_linear_offset is a direct argument rather than hard-coded to 1.0f with a _v2
  // alongside: this entry point is new, so there is no prior behavior to preserve.
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  dispatch::group_scaled_swiglu_fwd_helper<ClampedSwiGLUParam, clamped_silu_approx_x2<fp32, fp32>>(
      input, prob, output, param, nullptr, stream);
}
