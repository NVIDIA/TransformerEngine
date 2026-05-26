/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_silu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_silu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, silu<fp32, fp32>>(input, output, stream);
}

void nvte_dsilu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dsilu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dsilu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_swiglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swiglu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, silu<fp32, fp32>>(input, output, e, stream);
}

void nvte_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswiglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(grad, input, output, e, stream);
}

void nvte_clamped_swiglu(const NVTETensor input, NVTETensor output, float limit, float alpha,
                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_swiglu);
  using namespace transformer_engine;
  // Preserve original behavior: linear (gate) component offset is hard-coded to 1.0f.
  ClampedSwiGLUParam param = {limit, alpha, /*glu_linear_offset=*/1.0f};
  gated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(input, output, param, stream);
}

void nvte_clamped_swiglu_v2(const NVTETensor input, NVTETensor output, float limit, float alpha,
                            float glu_linear_offset, cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_swiglu_v2);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  gated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(input, output, param, stream);
}

void nvte_clamped_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                          float limit, float alpha, cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_dswiglu);
  using namespace transformer_engine;
  // Preserve original behavior: linear (gate) component offset is hard-coded to 1.0f.
  ClampedSwiGLUParam param = {limit, alpha, /*glu_linear_offset=*/1.0f};
  dgated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>, clamped_dsilu<fp32, fp32>>(
      grad, input, output, param, stream);
}

void nvte_clamped_dswiglu_v2(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                             float limit, float alpha, float glu_linear_offset,
                             cudaStream_t stream) {
  NVTE_API_CALL(nvte_clamped_dswiglu_v2);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  dgated_act_fn<fp32, ClampedSwiGLUParam, clamped_silu<fp32, fp32>, clamped_dsilu<fp32, fp32>>(
      grad, input, output, param, stream);
}
