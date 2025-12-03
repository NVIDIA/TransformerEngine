/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_gelu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_gelu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, gelu<fp32, fp32>>(input, output, stream);
}

void nvte_dgelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dgelu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dgelu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_quantize_dbias_dgelu(const NVTETensor input, const NVTETensor activation_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dgelu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_geglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_geglu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, gelu<fp32, fp32>>(input, output, e, stream);
}

void nvte_dgeglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_dgeglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, gelu<fp32, fp32>, dgelu<fp32, fp32>>(grad, input, output, e, stream);
}

void nvte_qgelu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_qgelu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, qgelu<fp32, fp32>>(input, output, stream);
}

void nvte_dqgelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_dqgelu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dqgelu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_quantize_dbias_dqgelu(const NVTETensor input, const NVTETensor activation_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dqgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dqgelu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_qgeglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_qgeglu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, qgelu<fp32, fp32>>(input, output, e, stream);
}

void nvte_dqgeglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_dqgeglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, qgelu<fp32, fp32>, dqgelu<fp32, fp32>>(grad, input, output, e, stream);
}
