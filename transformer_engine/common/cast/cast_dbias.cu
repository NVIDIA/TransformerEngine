/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include "../common.h"
#include "dispatch/quantize.cuh"

void nvte_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                         NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr const NVTETensor activation_input = nullptr;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, nullptr>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}
