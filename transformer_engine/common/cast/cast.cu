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
#include "mxfp4/fake_quantize_mxfp4.cuh"
#include "transformer_engine/transpose.h"

void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, nullptr, stream);
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

void nvte_mxfp4_fake_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mxfp4_fake_quantize);
  using namespace transformer_engine;
  const Tensor &input_tensor = *convertNVTETensorCheck(input);
  Tensor &output_tensor = *convertNVTETensorCheck(output);
  NVTE_CHECK(input_tensor.dtype() == DType::kBFloat16 || input_tensor.dtype() == DType::kFloat32,
             "MXFP4 fake-quantize supports BF16/FP32 inputs, got ",
             to_string(input_tensor.dtype()));
  NVTE_CHECK(output_tensor.dtype() == input_tensor.dtype(),
             "MXFP4 fake-quantize output dtype must match the input dtype.");
  NVTE_CHECK(input_tensor.data.shape == output_tensor.data.shape,
             "MXFP4 fake-quantize output shape must match the input shape.");
  const size_t numel = input_tensor.numel();
  NVTE_CHECK(numel % 32 == 0, "MXFP4 fake-quantize needs the element count divisible by 32.");
  NVTE_CHECK(!input_tensor.data.shape.empty() && input_tensor.data.shape.back() % 32 == 0,
             "MXFP4 fake-quantize needs the innermost dimension divisible by 32 "
             "(1x32 blocks must not cross rows).");
  if (numel == 0) {
    return;
  }
  if (input_tensor.dtype() == DType::kBFloat16) {
    dispatch::mxfp4::fake_quantize_mxfp4_launch<bf16>(
        reinterpret_cast<const bf16 *>(input_tensor.data.dptr),
        reinterpret_cast<bf16 *>(output_tensor.data.dptr), numel, stream);
  } else {
    dispatch::mxfp4::fake_quantize_mxfp4_launch<fp32>(
        reinterpret_cast<const fp32 *>(input_tensor.data.dptr),
        reinterpret_cast<fp32 *>(output_tensor.data.dptr), numel, stream);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dequantize);
  using namespace transformer_engine;
  dispatch::dequantize_helper(*convertNVTETensorCheck(input), convertNVTETensorCheck(output),
                              stream);
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
