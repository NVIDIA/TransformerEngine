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

void nvte_group_dgelu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                      NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_dgelu);
  using namespace transformer_engine;
  NVTEGroupedTensor dbias = nullptr;
  NVTETensor workspace = nullptr;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dgelu<fp32, fp32>>(
      grad, input, output, dbias, workspace, nullptr, stream);
}

void nvte_group_qgelu(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_qgelu);
  using namespace transformer_engine;
  constexpr bool IS_ACT = true;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, qgelu<fp32, fp32>>(input, output, nullptr,
                                                                        stream);
}

void nvte_group_dqgelu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                       NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_dqgelu);
  using namespace transformer_engine;
  NVTEGroupedTensor dbias = nullptr;
  NVTETensor workspace = nullptr;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dqgelu<fp32, fp32>>(
      grad, input, output, dbias, workspace, nullptr, stream);
}
