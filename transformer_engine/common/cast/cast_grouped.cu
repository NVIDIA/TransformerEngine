/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/recipe.h>

#include "../common.h"
#include "dispatch/dequantize.cuh"
#include "dispatch/quantize.cuh"
#include "fp8/group_amax_fp8.cuh"

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

void nvte_group_compute_scale_from_amax(NVTEGroupedTensor output,
                                        const NVTEQuantizationConfig config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_compute_scale_from_amax);
  using namespace transformer_engine;

  NVTE_CHECK(output != nullptr, "Invalid grouped output tensor (got NULL)");
  auto &out = *convertNVTEGroupedTensorCheck(output);
  const size_t num_tensors = out.num_tensors;
  if (num_tensors == 0) {
    return;
  }
  NVTE_CHECK(
      is_fp8_dtype(out.dtype()),
      "Grouped scale update requires an FP8 output tensor, but got dtype=", to_string(out.dtype()));

  // FP8 current scaling keeps a single amax/scale/scale_inv per group; the
  // scale_inv buffer is aliased by both directions, so pick whichever is set.
  float *amax_ptr = reinterpret_cast<float *>(out.amax.dptr != nullptr ? out.amax.dptr
                                                                       : out.columnwise_amax.dptr);
  NVTE_CHECK(amax_ptr != nullptr, "Grouped scale update requires an amax buffer.");
  float *scale_ptr = reinterpret_cast<float *>(out.scale.dptr);
  NVTE_CHECK(scale_ptr != nullptr, "Grouped scale update requires a scale buffer.");
  float *scale_inv_ptr = reinterpret_cast<float *>(
      out.scale_inv.dptr != nullptr ? out.scale_inv.dptr : out.columnwise_scale_inv.dptr);
  NVTE_CHECK(scale_inv_ptr != nullptr, "Grouped scale update requires a scale_inv buffer.");

  float max_fp8 = 0.f;
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      out.dtype(), DType,
      max_fp8 = Quantized_Limits<DType>::max_norm;);  // NOLINT(*)

  bool force_pow_2_scales = false;
  float epsilon = 0.f;
  float *noop_ptr = nullptr;
  if (config != nullptr) {
    const auto *config_cpp = reinterpret_cast<const QuantizationConfig *>(config);
    force_pow_2_scales = config_cpp->force_pow_2_scales;
    epsilon = config_cpp->amax_epsilon;
    const NVTETensor noop = config_cpp->noop_tensor;
    noop_ptr = reinterpret_cast<float *>(noop != nullptr ? convertNVTETensorCheck(noop)->data.dptr
                                                         : nullptr);
  }

  dispatch::fp8::launch_grouped_compute_scale_kernel(amax_ptr, scale_ptr, scale_inv_ptr,
                                                     num_tensors, max_fp8, force_pow_2_scales,
                                                     epsilon, noop_ptr, stream);
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
