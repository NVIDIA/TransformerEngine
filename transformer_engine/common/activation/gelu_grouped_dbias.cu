/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_group_quantize_dbias_dgelu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor activation_input,
                                     NVTEGroupedTensor output, NVTEGroupedTensor dbias,
                                     NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_quantize_dbias_dgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dgelu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_group_quantize_dbias_dqgelu(const NVTEGroupedTensor input,
                                      const NVTEGroupedTensor activation_input,
                                      NVTEGroupedTensor output, NVTEGroupedTensor dbias,
                                      NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_quantize_dbias_dqgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dqgelu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}
