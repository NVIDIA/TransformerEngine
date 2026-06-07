/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename InputType, typename OutputType>
void test_cast_fp8_grouped_impl(const std::vector<std::vector<size_t>>& shapes,
                                DType input_dtype, DType output_dtype) {
  const size_t num_tensors = shapes.size();

  // Create standard Tensor objects
  std::vector<Tensor> in_tensors;
  std::vector<Tensor> out_tensors;
  std::vector<Tensor*> in_tensor_ptrs;
  std::vector<Tensor*> out_tensor_ptrs;

  in_tensors.reserve(num_tensors);
  out_tensors.reserve(num_tensors);
  in_tensor_ptrs.reserve(num_tensors);
  out_tensor_ptrs.reserve(num_tensors);

  for (size_t t = 0; t < num_tensors; ++t) {
    in_tensors.emplace_back("in_" + std::to_string(t), shapes[t], input_dtype);
    out_tensors.emplace_back("out_" + std::to_string(t), shapes[t], output_dtype,
                             true, false, NVTE_DELAYED_TENSOR_SCALING);

    // Initialize inputs with random uniform values
    fillUniform(&in_tensors[t]);
    in_tensors[t].from_cpu();

    // Initialize scales with random scaling factors
    float random_scale = 1.5f + static_cast<float>(t) * 0.5f;
    out_tensors[t].set_scale(random_scale);
    out_tensors[t].set_scale_inv(0.0f); // Clear to ensure it's written
    out_tensors[t].set_amax(0.0f);      // Clear amax

    in_tensor_ptrs.push_back(&in_tensors[t]);
    out_tensor_ptrs.push_back(&out_tensors[t]);
  }

  // Build grouped tensors
  GroupedBuffers in_group = build_grouped_tensor(in_tensor_ptrs, NVTE_DELAYED_TENSOR_SCALING);
  GroupedBuffers out_group = build_grouped_tensor(out_tensor_ptrs, NVTE_DELAYED_TENSOR_SCALING);

  // CPU reference computation
  std::vector<std::vector<float>> ref_outputs(num_tensors);
  std::vector<float> ref_amaxs(num_tensors, 0.0f);
  std::vector<float> ref_scale_invs(num_tensors, 0.0f);

  for (size_t t = 0; t < num_tensors; ++t) {
    size_t size = product(shapes[t]);
    ref_outputs[t].resize(size);
    float scale = out_tensors[t].scale();
    float cur_amax = 0.0f;

    InputType* in_cpu = in_tensors[t].rowwise_cpu_dptr<InputType>();
    for (size_t i = 0; i < size; ++i) {
      float val = static_cast<float>(in_cpu[i]);
      cur_amax = std::max(cur_amax, std::abs(val));
      float scaled_val = val * scale;
      ref_outputs[t][i] = scaled_val;
    }
    ref_amaxs[t] = cur_amax;
    ref_scale_invs[t] = 1.0f / scale;
  }

  // Run GPU grouped quantization
  QuantizationConfigWrapper quant_config;
  nvte_group_quantize(in_group.get_handle(), out_group.get_handle(), quant_config, 0);
  cudaDeviceSynchronize();

  // Copy results back from grouped buffer to individual output tensors
  for (size_t t = 0; t < num_tensors; ++t) {
    // 1. Copy output data
    size_t offset_bytes = (out_group.offsets_host[t] * typeToNumBits(out_group.dtype)) / 8;
    NVTE_CHECK_CUDA(cudaMemcpy(out_tensors[t].rowwise_dptr(),
                               static_cast<char*>(out_group.get_data()) + offset_bytes,
                               out_group.tensor_bytes[t],
                               cudaMemcpyDeviceToDevice));

    // 2. Copy scale_inv
    NVTEBasicTensor scale_inv_bt = nvte_get_tensor_param(out_tensors[t].data(), kNVTERowwiseScaleInv);
    if (scale_inv_bt.data_ptr != nullptr) {
      NVTE_CHECK_CUDA(cudaMemcpy(scale_inv_bt.data_ptr,
                                 static_cast<float*>(out_group.scale_inv.get()) + t,
                                 sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }

    // 3. Copy amax
    NVTEBasicTensor amax_bt = nvte_get_tensor_param(out_tensors[t].data(), kNVTEAmax);
    if (amax_bt.data_ptr != nullptr) {
      NVTE_CHECK_CUDA(cudaMemcpy(amax_bt.data_ptr,
                                 static_cast<float*>(out_group.amax_dev.get()) + t,
                                 sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }
  }

  // Validate results
  for (size_t t = 0; t < num_tensors; ++t) {
    size_t size = product(shapes[t]);
    out_tensors[t].to_cpu();

    // 1. Compare scale inverse
    float scale_inv_gpu = out_tensors[t].rowwise_scale_inv();
    EXPECT_NEAR(scale_inv_gpu, ref_scale_invs[t], 1e-4);

    // 2. Compare amax
    float amax_gpu = out_tensors[t].amax();
    EXPECT_NEAR(amax_gpu, ref_amaxs[t], 1e-4);

    // 3. Compare outputs
    OutputType* out_cpu = out_tensors[t].rowwise_cpu_dptr<OutputType>();
    for (size_t i = 0; i < size; ++i) {
      float gpu_val = static_cast<float>(out_cpu[i]);
      float ref_val = static_cast<float>(OutputType(ref_outputs[t][i]));
      EXPECT_NEAR(gpu_val, ref_val, 1e-4);
    }
  }
}

class CastFP8GroupedTestSuite : public ::testing::Test {};

TEST_F(CastFP8GroupedTestSuite, BF16_to_E4M3_Uniform) {
  test_cast_fp8_grouped_impl<bf16, fp8e4m3>(
      {{32, 64}, {32, 64}, {32, 64}},
      DType::kBFloat16, DType::kFloat8E4M3);
}

TEST_F(CastFP8GroupedTestSuite, BF16_to_E4M3_Varying) {
  test_cast_fp8_grouped_impl<bf16, fp8e4m3>(
      {{16, 32}, {64, 128}, {32, 64}},
      DType::kBFloat16, DType::kFloat8E4M3);
}

TEST_F(CastFP8GroupedTestSuite, FP16_to_E4M3_Varying) {
  test_cast_fp8_grouped_impl<fp16, fp8e4m3>(
      {{8, 16}, {128, 64}, {64, 32}},
      DType::kFloat16, DType::kFloat8E4M3);
}

TEST_F(CastFP8GroupedTestSuite, FP32_to_E5M2_Varying) {
  test_cast_fp8_grouped_impl<float, fp8e5m2>(
      {{32, 32}, {16, 64}, {128, 32}},
      DType::kFloat32, DType::kFloat8E5M2);
}

} // namespace
