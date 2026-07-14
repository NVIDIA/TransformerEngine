/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>

#include "../common.h"
#include "../util/math.h"
#include "./scaled_activation.h"

void nvte_scaled_srelu(const NVTETensor input, const NVTETensor act_scales, NVTETensor output,
                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_srelu);
  using namespace transformer_engine;
  Empty param = {};
  launch_scaled_unary_forward<Empty, srelu<fp32, fp32>>(input, act_scales, output, param, stream,
                                                        "nvte_scaled_srelu");
}

void nvte_scaled_dsrelu(const NVTETensor grad, const NVTETensor input, const NVTETensor act_scales,
                        NVTETensor grad_input, NVTETensor grad_act_scales, cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_dsrelu);
  using namespace transformer_engine;
  Empty param = {};
  launch_scaled_unary_backward<Empty, srelu<fp32, fp32>, dsrelu<fp32, fp32>>(
      grad, input, act_scales, grad_input, grad_act_scales, param, stream, "nvte_scaled_dsrelu");
}
