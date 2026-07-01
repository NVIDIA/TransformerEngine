/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor activation_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dsilu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, dsilu<fp32, fp32>>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}
