/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_group_relu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_relu);
  using namespace transformer_engine;
  constexpr bool IS_ACT = true;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, relu<fp32, fp32>>(input, output, nullptr,
                                                                       stream);
}
