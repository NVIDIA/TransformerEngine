/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "./activation_template.h"
#include "../util/math.h"


void nvte_swish(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream) {
  NVTE_API_CALL(nvte_swish);
  using namespace transformer_engine;
  act_fn<fp32, Empty, swish<fp32, fp32>>(*reinterpret_cast<const Tensor*>(input),
                                         reinterpret_cast<Tensor*>(output),
                                         stream);
}

void nvte_dswish(const NVTETensor grad,
                const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswish);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dswish<fp32, fp32>>(*reinterpret_cast<const Tensor*>(grad),
                                           *reinterpret_cast<const Tensor*>(input),
                                           reinterpret_cast<Tensor*>(output),
                                           stream);
}

void nvte_swiglu(const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_swiglu);
  using namespace transformer_engine;
  gated_act_fn<fp32, Empty, swish<fp32, fp32>>(*reinterpret_cast<const Tensor*>(input),
                                               reinterpret_cast<Tensor*>(output),
                                               stream);
}

void nvte_dswiglu(const NVTETensor grad,
                  const NVTETensor input,
                  NVTETensor output,
                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswiglu);
  using namespace transformer_engine;
  dgated_act_fn<fp32, Empty, swish<fp32, fp32>, dswish<fp32, fp32>>(
    *reinterpret_cast<const Tensor*>(grad),
    *reinterpret_cast<const Tensor*>(input),
    reinterpret_cast<Tensor*>(output),
    stream);
}
