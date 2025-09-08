/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void nvte_gptoss_swiglu(const NVTETensor input, NVTETensor output, const float* const args,
                        int args_size, cudaStream_t stream) {
  NVTE_API_CALL(nvte_gptoss_swiglu);
  NVTE_CHECK(args_size==1);
  const float limit = *args;
  using namespace transformer_engine;
  GptOssParam param = {limit};
  gated_act_fn<fp32, GptOssParam, oss_silu<fp32, fp32>>(input, output, param, stream);
}

void nvte_gptoss_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                         const float* const args, int args_size, cudaStream_t stream) {
  NVTE_API_CALL(nvte_gptoss_dswiglu);
  NVTE_CHECK(args_size==1);
  const float limit = *args;
  using namespace transformer_engine;
  GptOssParam param = {limit};
  dgated_act_fn<fp32, GptOssParam, oss_silu<fp32, fp32>, oss_dsilu<fp32, fp32>>(grad, input, output,
                                                                                param, stream);
}
