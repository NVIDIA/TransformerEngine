/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_relu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_relu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, relu<fp32, fp32>>(input, output, stream);
}

void nvte_group_relu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_relu);
  using namespace transformer_engine;
  constexpr bool IS_ACT = true;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, relu<fp32, fp32>>(input, output, nullptr,
                                                                       stream);
}

void nvte_drelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_drelu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, drelu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_group_drelu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                      NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_drelu);
  using namespace transformer_engine;
  NVTEGroupedTensor dbias = nullptr;
  NVTETensor workspace = nullptr;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, drelu<fp32, fp32>>(
      grad, input, output, dbias, workspace, nullptr, stream);
}

void nvte_quantize_dbias_drelu(const NVTETensor input, const NVTETensor activation_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_drelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, drelu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_group_quantize_dbias_drelu(const NVTEGroupedTensor input,
                                     const NVTEGroupedTensor activation_input,
                                     NVTEGroupedTensor output, NVTEGroupedTensor dbias,
                                     NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_quantize_dbias_drelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, drelu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_reglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_reglu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, relu<fp32, fp32>>(input, output, e, stream);
}

void nvte_dreglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_dreglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, relu<fp32, fp32>, drelu<fp32, fp32>>(grad, input, output, e, stream);
}

void nvte_srelu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_srelu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, srelu<fp32, fp32>>(input, output, stream);
}

void nvte_group_srelu(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_srelu);
  using namespace transformer_engine;
  constexpr bool IS_ACT = true;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, srelu<fp32, fp32>>(input, output, nullptr,
                                                                        stream);
}

void nvte_dsrelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_dsrelu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dsrelu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_group_dsrelu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                       NVTEGroupedTensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_dsrelu);
  using namespace transformer_engine;
  NVTEGroupedTensor dbias = nullptr;
  NVTETensor workspace = nullptr;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dsrelu<fp32, fp32>>(
      grad, input, output, dbias, workspace, nullptr, stream);
}

void nvte_quantize_dbias_dsrelu(const NVTETensor input, const NVTETensor activation_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dsrelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dsrelu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_group_quantize_dbias_dsrelu(const NVTEGroupedTensor input,
                                      const NVTEGroupedTensor activation_input,
                                      NVTEGroupedTensor output, NVTEGroupedTensor dbias,
                                      NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_quantize_dbias_dsrelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dsrelu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_sreglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_sreglu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, srelu<fp32, fp32>>(input, output, e, stream);
}

void nvte_dsreglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_dsreglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, srelu<fp32, fp32>, dsrelu<fp32, fp32>>(grad, input, output, e, stream);
}
