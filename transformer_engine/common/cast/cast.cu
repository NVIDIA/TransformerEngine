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

void nvte_nvfp4_quantize_4over6_multi(const NVTETensor *inputs, NVTETensor *outputs,
                                      const NVTEQuantizationConfig quant_config,
                                      const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_quantize_4over6_multi);
  using namespace transformer_engine;
  using namespace transformer_engine::dispatch::nvfp4;

  NVTE_CHECK(inputs != nullptr && outputs != nullptr,
             "Multi 4over6 quantization requires non-null tensor lists.");
  NVTE_CHECK(num_tensors > 0, "Multi 4over6 quantization requires a non-empty tensor list.");

  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<const QuantizationConfig *>(quant_config);
  }
  NVTE_CHECK(quant_config_cpp.nvfp4_4over6_mode != kNVTENVFP44Over6Disabled,
             "Multi 4over6 quantization requires a non-disabled 4over6 mode.");
  NVTE_CHECK(!quant_config_cpp.stochastic_rounding,
             "Multi 4over6 quantization does not support stochastic rounding.");

  std::vector<Tensor> in_list;
  std::vector<Tensor> out_list;
  in_list.reserve(num_tensors);
  out_list.reserve(num_tensors);
  size_t rows = 0, cols = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    in_list.push_back(*convertNVTETensorCheck(inputs[i]));
    out_list.push_back(*convertNVTETensorCheck(outputs[i]));
    const auto &in = in_list.back();
    const auto &out = out_list.back();
    NVTE_CHECK(out.has_data() && !out.has_columnwise_data(),
               "Multi 4over6 supports rowwise-only outputs.");
    NVTE_CHECK(out.scale_inv.dptr != nullptr, "Multi 4over6 requires allocated scaling tensors.");
    NVTE_CHECK(out.amax.dptr != nullptr, "Multi 4over6 requires allocated amax tensors.");
    NVTE_CHECK(is_fp4_dtype(out.data.dtype), "Multi 4over6 output must have FP4 type.");
    NVTE_CHECK(!out.row_scaled_nvfp4,
               "Multi 4over6 targets per-tensor-scaled tensors, not row-scaled ones.");
    NVTE_CHECK(!out.with_gemm_swizzled_scales, "Multi 4over6 requires compact scale layout.");
    if (i == 0) {
      rows = in.flat_first_dim();
      cols = in.flat_last_dim();
    } else {
      NVTE_CHECK(in.flat_first_dim() == rows && in.flat_last_dim() == cols,
                 "Multi 4over6 requires same-shaped input tensors.");
      NVTE_CHECK(in.dtype() == in_list[0].dtype(), "Multi 4over6 requires the same input dtype.");
      NVTE_CHECK(out.scale_inv.shape == out_list[0].scale_inv.shape,
                 "Multi 4over6 requires the same scaling tensor shape.");
    }
  }
  NVTE_CHECK(cols % quantize_4over6_kernel::kGroupSize == 0,
             "Multi 4over6 quantization requires columns divisible by ",
             quantize_4over6_kernel::kGroupSize, ".");
  const size_t scale_stride = out_list[0].scale_inv.shape[1];

  using quantize_4over6_kernel::BatchedQuantizeParams;
  std::vector<BatchedQuantizeParams> host_params(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    host_params[i].input = in_list[i].data.dptr;
    host_params[i].output = out_list[i].data.dptr;
    host_params[i].scales = out_list[i].scale_inv.dptr;
    host_params[i].amax = static_cast<float *>(out_list[i].amax.dptr);
  }

  void *params_dev = nullptr;
  float *temp_amax_dev = nullptr;
  NVTE_CHECK_CUDA(
      cudaMallocAsync(&params_dev, num_tensors * sizeof(BatchedQuantizeParams), stream));
  NVTE_CHECK_CUDA(cudaMallocAsync(&temp_amax_dev, num_tensors * sizeof(float), stream));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(params_dev, host_params.data(),
                                  num_tensors * sizeof(BatchedQuantizeParams),
                                  cudaMemcpyHostToDevice, stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(temp_amax_dev, 0, num_tensors * sizeof(float), stream));

  const float *noop_ptr = nullptr;
  TRANSFORMER_ENGINE_NVFP4_4OVER6_E4M3_MAX_SWITCH(
      out_list[0].nvfp4_e4m3_max, E4M3_MAX,
      TRANSFORMER_ENGINE_NVFP4_4OVER6_MODE_SWITCH(
          quant_config_cpp.nvfp4_4over6_mode, MODE,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              quant_config_cpp.nvfp4_4over6_err_use_fast_math, ERR_USE_FAST_MATH, {
                using Cfg = quantize_4over6_kernel::Config<MODE, ERR_USE_FAST_MATH>;
                TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
                    in_list[0].dtype(), IType,
                    quantize_4over6_kernel::launch_quantize_4over6_batched<Cfg, E4M3_MAX, IType>(
                        params_dev, temp_amax_dev, static_cast<int>(num_tensors), rows, cols,
                        scale_stride, noop_ptr, stream););
              });););

  NVTE_CHECK_CUDA(cudaFreeAsync(params_dev, stream));
  NVTE_CHECK_CUDA(cudaFreeAsync(temp_amax_dev, stream));
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
