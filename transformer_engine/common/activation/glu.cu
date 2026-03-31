/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_glu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_glu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, sigmoid<fp32, fp32>>(input, output, e, stream);
}

void nvte_dglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
               cudaStream_t stream) {
  NVTE_API_CALL(nvte_dglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, sigmoid<fp32, fp32>, dsigmoid<fp32, fp32>>(grad, input, output, e,
                                                                        stream);
}
