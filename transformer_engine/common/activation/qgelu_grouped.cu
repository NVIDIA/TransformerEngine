/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_group_qgelu(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_qgelu);
  using namespace transformer_engine;
  constexpr bool IS_ACT = true;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, qgelu<fp32, fp32>>(input, output, nullptr,
                                                                        stream);
}

void nvte_group_qgelu_v2(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                         NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_qgelu_v2);
  using namespace transformer_engine;
  convertNVTETensorCheck(workspace);
  dispatch::group_quantize_fwd_helper<true, Empty, qgelu<fp32, fp32>>(input, output, nullptr,
                                                                      stream, workspace);
}
