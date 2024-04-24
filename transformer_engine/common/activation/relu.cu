/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "./activation_template.h"
#include "../util/math.h"


void nvte_relu(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream) {
  NVTE_API_CALL(nvte_relu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, relu<fp32, fp32>>(*reinterpret_cast<const Tensor*>(input),
                                        reinterpret_cast<Tensor*>(output),
                                        stream);
}

void nvte_drelu(const NVTETensor grad,
                const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_drelu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, drelu<fp32, fp32>>(*reinterpret_cast<const Tensor*>(grad),
                                         *reinterpret_cast<const Tensor*>(input),
                                         reinterpret_cast<Tensor*>(output),
                                         stream);
}

void nvte_reglu(const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_reglu);
  using namespace transformer_engine;
  gated_act_fn<fp32, Empty, relu<fp32, fp32>>(*reinterpret_cast<const Tensor*>(input),
        reinterpret_cast<Tensor*>(output),
        stream);
}

void nvte_dreglu(const NVTETensor grad,
                 const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_dreglu);
  using namespace transformer_engine;
  dgated_act_fn<fp32, Empty, relu<fp32, fp32>, drelu<fp32, fp32>>(
    *reinterpret_cast<const Tensor*>(grad),
    *reinterpret_cast<const Tensor*>(input),
    reinterpret_cast<Tensor*>(output),
    stream);
}
