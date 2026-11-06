/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_group_gelu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_gelu);
  using namespace transformer_engine;
  constexpr bool IS_ACT = true;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, gelu<fp32, fp32>>(input, output, nullptr,
                                                                       stream);
}

void nvte_group_gelu_v2(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                        NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_gelu_v2);
  using namespace transformer_engine;
  convertNVTETensorCheck(workspace);
  dispatch::group_quantize_fwd_helper<true, Empty, gelu<fp32, fp32>>(input, output, nullptr, stream,
                                                                     workspace);
}
