/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>

#include <cmath>

#include "../common.h"
#include "../util/math.h"
#include "./scaled_activation.h"

void nvte_scaled_swiglu(const NVTETensor input, const NVTETensor act_scales, NVTETensor output,
                        int64_t glu_interleave_size, cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_swiglu);
  using namespace transformer_engine;
  Empty param = {};
  launch_scaled_gated_forward<Empty, silu<fp32, fp32>>(
      input, act_scales, output, param, glu_interleave_size, stream, "nvte_scaled_swiglu");
}

void nvte_scaled_dswiglu(const NVTETensor grad, const NVTETensor input, const NVTETensor act_scales,
                         NVTETensor grad_input, NVTETensor grad_act_scales,
                         int64_t glu_interleave_size, cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_dswiglu);
  using namespace transformer_engine;
  Empty param = {};
  launch_scaled_gated_backward<Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(
      grad, input, act_scales, grad_input, grad_act_scales, param, glu_interleave_size, stream,
      "nvte_scaled_dswiglu");
}

void nvte_scaled_situglu(const NVTETensor input, const NVTETensor act_scales, NVTETensor output,
                         float beta1, float beta2, int64_t glu_interleave_size,
                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_situglu);
  using namespace transformer_engine;
  NVTE_CHECK(beta1 > 0.0f && std::isfinite(beta1), "beta1 must be finite and positive.");
  NVTE_CHECK(beta2 > 0.0f && std::isfinite(beta2), "beta2 must be finite and positive.");
  SiTUGLUParam param = {beta1, beta2};
  launch_scaled_gated_forward<SiTUGLUParam, situ_gate<fp32, fp32>>(
      input, act_scales, output, param, glu_interleave_size, stream, "nvte_scaled_situglu");
}

void nvte_scaled_dsituglu(const NVTETensor grad, const NVTETensor input,
                          const NVTETensor act_scales, NVTETensor grad_input,
                          NVTETensor grad_act_scales, float beta1, float beta2,
                          int64_t glu_interleave_size, cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_dsituglu);
  using namespace transformer_engine;
  NVTE_CHECK(beta1 > 0.0f && std::isfinite(beta1), "beta1 must be finite and positive.");
  NVTE_CHECK(beta2 > 0.0f && std::isfinite(beta2), "beta2 must be finite and positive.");
  SiTUGLUParam param = {beta1, beta2};
  launch_scaled_gated_backward<SiTUGLUParam, situ_gate<fp32, fp32>, dsitu_gate<fp32, fp32>>(
      grad, input, act_scales, grad_input, grad_act_scales, param, glu_interleave_size, stream,
      "nvte_scaled_dsituglu");
}

void nvte_scaled_clamped_swiglu(const NVTETensor input, const NVTETensor act_scales,
                                NVTETensor output, float limit, float alpha,
                                float glu_linear_offset, int64_t glu_interleave_size,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_clamped_swiglu);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  launch_scaled_gated_forward<ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(
      input, act_scales, output, param, glu_interleave_size, stream, "nvte_scaled_clamped_swiglu");
}

void nvte_scaled_clamped_dswiglu(const NVTETensor grad, const NVTETensor input,
                                 const NVTETensor act_scales, NVTETensor grad_input,
                                 NVTETensor grad_act_scales, float limit, float alpha,
                                 float glu_linear_offset, int64_t glu_interleave_size,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_clamped_dswiglu);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  launch_scaled_gated_backward<ClampedSwiGLUParam, clamped_silu<fp32, fp32>,
                               clamped_dsilu<fp32, fp32>>(
      grad, input, act_scales, grad_input, grad_act_scales, param, glu_interleave_size, stream,
      "nvte_scaled_clamped_dswiglu");
}
