/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

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
