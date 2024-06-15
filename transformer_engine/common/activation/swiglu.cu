/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_silu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_silu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, silu<fp32, fp32>>(*reinterpret_cast<const Tensor*>(input),
                                        reinterpret_cast<Tensor*>(output), stream);
}

void nvte_dsilu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dsilu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dsilu<fp32, fp32>>(*reinterpret_cast<const Tensor*>(grad),
                                          *reinterpret_cast<const Tensor*>(input),
                                          reinterpret_cast<Tensor*>(output), stream);
}

void nvte_swiglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swiglu);
  using namespace transformer_engine;
  gated_act_fn<fp32, Empty, silu<fp32, fp32>>(*reinterpret_cast<const Tensor*>(input),
                                              reinterpret_cast<Tensor*>(output), stream);
}

void nvte_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswiglu);
  using namespace transformer_engine;
  dgated_act_fn<fp32, Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(
      *reinterpret_cast<const Tensor*>(grad), *reinterpret_cast<const Tensor*>(input),
      reinterpret_cast<Tensor*>(output), stream);
}
