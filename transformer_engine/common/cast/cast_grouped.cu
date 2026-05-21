/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include "../common.h"
#include "dispatch/dequantize.cuh"
#include "dispatch/quantize.cuh"

void nvte_group_quantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, quant_config, stream);
}

void nvte_group_dequantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_dequantize);
  using namespace transformer_engine;
  dispatch::group_dequantize_helper(*convertNVTEGroupedTensorCheck(input),
                                    convertNVTEGroupedTensorCheck(output), stream);
}

// Group quantize assumes contiguous inputs and outputs in memory allocation.
// Note: this API assumes knowing split sections from the host. If split information
// comes from D2H copy, it will break cuda graph capture.
void nvte_group_nvfp4_quantize_with_amax(const NVTETensor input, NVTETensor *outputs,
                                         const size_t *split_sections, const size_t num_tensors,
                                         const NVTEQuantizationConfig quant_config,
                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_nvfp4_quantize_with_amax);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;

  dispatch::group_quantize_fwd_host_aware_helper<IS_ACT, Empty, nullptr>(
      input, outputs, split_sections, num_tensors, quant_config, stream);
}
