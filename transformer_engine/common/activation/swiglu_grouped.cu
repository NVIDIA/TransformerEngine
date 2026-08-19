/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_group_silu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_silu);
  using namespace transformer_engine;
  constexpr bool IS_ACT = true;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, silu<fp32, fp32>>(input, output, nullptr,
                                                                       stream);
}

void nvte_group_scaled_swiglu(const NVTEGroupedTensor input, const NVTETensor prob,
                              NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_scaled_swiglu);
  using namespace transformer_engine;
  // Scaled SwiGLU recompute: (silu(act) * gate) * prob -> columnwise MXFP8.
  Empty e = {};
  dispatch::group_scaled_swiglu_fwd_helper<Empty, silu<fp32, fp32>>(input, prob, output, e, nullptr,
                                                                    stream);
}

void nvte_group_scaled_clamped_swiglu(const NVTEGroupedTensor input, const NVTETensor prob,
                                      NVTEGroupedTensor output, float limit, float alpha,
                                      float glu_linear_offset, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_scaled_clamped_swiglu);
  using namespace transformer_engine;
  // glu_linear_offset is a direct argument rather than hard-coded to 1.0f with a _v2
  // alongside: this entry point is new, so there is no prior behavior to preserve.
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  dispatch::group_scaled_swiglu_fwd_helper<ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(
      input, prob, output, param, nullptr, stream);
}

void nvte_group_dsilu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                      NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_dsilu);
  using namespace transformer_engine;
  NVTEGroupedTensor dbias = nullptr;
  NVTETensor workspace = nullptr;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dsilu<fp32, fp32>>(
      grad, input, output, dbias, workspace, nullptr, stream);
}
