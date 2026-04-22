/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/multi_stream.h>

#include "../common.h"
#include "../transpose/cast_transpose.h"
#include "../util/multi_stream.h"
#include "../utils.cuh"
#include "dispatch/dequantize.cuh"
#include "dispatch/quantize.cuh"
#include "nvfp4/quantize_pertoken_nvfp4.cuh"
#include "transformer_engine/transpose.h"

void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, nullptr, stream);
}

void nvte_group_quantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::group_quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, quant_config, stream);
}

void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_noop);
  using namespace transformer_engine;

  // Create config with noop tensor
  QuantizationConfig quant_config;
  quant_config.noop_tensor = noop;

  nvte_quantize_v2(input, output, reinterpret_cast<NVTEQuantizationConfig>(&quant_config), stream);
}

void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_v2);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, quant_config, stream);
}

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

void nvte_group_quantize_dbias(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                               NVTEGroupedTensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_quantize_dbias);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr const NVTEGroupedTensor activation_input = nullptr;

  dispatch::group_quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, nullptr>(
      input, activation_input, output, dbias, workspace, nullptr, stream);
}

void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dequantize);
  using namespace transformer_engine;
  dispatch::dequantize_helper(*convertNVTETensorCheck(input), convertNVTETensorCheck(output),
                              stream);
}

void nvte_group_dequantize(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_dequantize);
  using namespace transformer_engine;
  dispatch::group_dequantize_helper(*convertNVTEGroupedTensorCheck(input),
                                    convertNVTEGroupedTensorCheck(output), stream);
}

void nvte_multi_tensor_quantize(const NVTETensor *inputs, NVTETensor *outputs,
                                const NVTEQuantizationConfig quant_configs,
                                const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;

  const size_t num_streams = nvte_get_num_compute_streams();

  int num_stream_used = std::min(num_streams, num_tensors);
  // wait for current stream to finish
  NVTE_CHECK_CUDA(cudaEventRecord(detail::get_compute_stream_event(0), stream));
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(
        cudaStreamWaitEvent(detail::get_compute_stream(s), detail::get_compute_stream_event(0)));
  }

  for (int i = 0; i < num_tensors; i++) {
    dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(
        inputs[i], outputs[i], quant_configs, detail::get_compute_stream(i % num_streams));
  }

  // record events on compute streams
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(
        cudaEventRecord(detail::get_compute_stream_event(s), detail::get_compute_stream(s)));
  }
  // wait for all compute streams to finish
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream, detail::get_compute_stream_event(s)));
  }
}

// Group quantize assumes contiguous inputs and outputs in memory allocation
// Note: this API assumes knowing split sections from the host, if split information
// comes from D2H copy, it will break cuda graph capture
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

void nvte_quantize_nvfp4_pertoken(const NVTETensor input, NVTETensor output_data,
                                  NVTETensor output_scales, NVTETensor output_per_token_scales,
                                  size_t num_rows, size_t num_cols, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_nvfp4_pertoken);
  using namespace transformer_engine;

  const auto &input_tensor = *reinterpret_cast<const Tensor *>(input);
  auto *data_tensor = reinterpret_cast<Tensor *>(output_data);
  auto *scales_tensor = reinterpret_cast<Tensor *>(output_scales);
  auto *pertoken_tensor = reinterpret_cast<Tensor *>(output_per_token_scales);

  const auto itype = input_tensor.data.dtype;

  NVTE_CHECK(num_cols % 16 == 0,
             "num_cols must be a multiple of 16 for per-token NVFP4 quantization");

  if (itype == DType::kBFloat16) {
    dispatch::nvfp4::quantize_pertoken_kernel::launch_quantize_pertoken_nvfp4<__nv_bfloat16>(
        num_rows, num_cols, reinterpret_cast<const __nv_bfloat16 *>(input_tensor.data.dptr),
        nullptr,  // row_offsets
        reinterpret_cast<uint8_t *>(data_tensor->data.dptr),
        reinterpret_cast<fp8e4m3 *>(scales_tensor->data.dptr),
        reinterpret_cast<float *>(pertoken_tensor->data.dptr), stream);
  } else if (itype == DType::kFloat16) {
    dispatch::nvfp4::quantize_pertoken_kernel::launch_quantize_pertoken_nvfp4<half>(
        num_rows, num_cols, reinterpret_cast<const half *>(input_tensor.data.dptr),
        nullptr,  // row_offsets
        reinterpret_cast<uint8_t *>(data_tensor->data.dptr),
        reinterpret_cast<fp8e4m3 *>(scales_tensor->data.dptr),
        reinterpret_cast<float *>(pertoken_tensor->data.dptr), stream);
  } else {
    NVTE_ERROR(
        "Unsupported input dtype for per-token NVFP4 quantization. "
        "Expected BFloat16 or Float16.");
  }
}
