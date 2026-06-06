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
void test_dequantize_fp8_grouped_impl(const std::vector<std::vector<size_t>>& shapes,
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
    // Input is FP8 (with scale_inv)
    in_tensors.emplace_back("in_" + std::to_string(t), shapes[t], input_dtype,
                            true, false, NVTE_DELAYED_TENSOR_SCALING);
    // Output is higher precision
    out_tensors.emplace_back("out_" + std::to_string(t), shapes[t], output_dtype);

    // Initialize inputs with random uniform FP8 values
    fillUniform(&in_tensors[t]);
    in_tensors[t].from_cpu();

    // Set scale_inv
    float random_scale_inv = 0.5f + static_cast<float>(t) * 0.25f;
    in_tensors[t].set_scale_inv(random_scale_inv);

    // Clear output
    fillUniform(&out_tensors[t]); // Initialize to some random values
    out_tensors[t].from_cpu();

    in_tensor_ptrs.push_back(&in_tensors[t]);
    out_tensor_ptrs.push_back(&out_tensors[t]);
  }

  // Build grouped tensors
  GroupedBuffers in_group = build_grouped_tensor(in_tensor_ptrs, NVTE_DELAYED_TENSOR_SCALING);
  GroupedBuffers out_group = build_grouped_tensor(out_tensor_ptrs, NVTE_DELAYED_TENSOR_SCALING);

  // CPU reference computation
  std::vector<std::vector<float>> ref_outputs(num_tensors);
  for (size_t t = 0; t < num_tensors; ++t) {
    size_t size = product(shapes[t]);
    ref_outputs[t].resize(size);
    float scale_inv = in_tensors[t].rowwise_scale_inv();

    InputType* in_cpu = in_tensors[t].rowwise_cpu_dptr<InputType>();
    for (size_t i = 0; i < size; ++i) {
      float val = static_cast<float>(in_cpu[i]);
      ref_outputs[t][i] = val * scale_inv;
    }
  }

  // Run GPU grouped dequantization
  nvte_group_dequantize(in_group.get_handle(), out_group.get_handle(), 0);
  cudaDeviceSynchronize();

  // Copy results back from grouped buffer to individual output tensors
  for (size_t t = 0; t < num_tensors; ++t) {
    size_t offset_bytes = (out_group.offsets_host[t] * typeToNumBits(out_group.dtype)) / 8;
    NVTE_CHECK_CUDA(cudaMemcpy(out_tensors[t].rowwise_dptr(),
                               static_cast<char*>(out_group.get_data()) + offset_bytes,
                               out_group.tensor_bytes[t],
                               cudaMemcpyDeviceToDevice));
  }

  // Validate results
  for (size_t t = 0; t < num_tensors; ++t) {
    size_t size = product(shapes[t]);
    out_tensors[t].to_cpu();

    OutputType* out_cpu = out_tensors[t].rowwise_cpu_dptr<OutputType>();
    for (size_t i = 0; i < size; ++i) {
      float gpu_val = static_cast<float>(out_cpu[i]);
      float ref_val = ref_outputs[t][i];
      EXPECT_NEAR(gpu_val, ref_val, 1e-4);
    }
  }
}

class DequantizeFP8GroupedTestSuite : public ::testing::Test {};

TEST_F(DequantizeFP8GroupedTestSuite, E4M3_to_BF16_Uniform) {
  test_dequantize_fp8_grouped_impl<fp8e4m3, bf16>(
      {{32, 64}, {32, 64}, {32, 64}},
      DType::kFloat8E4M3, DType::kBFloat16);
}

TEST_F(DequantizeFP8GroupedTestSuite, E4M3_to_BF16_Varying) {
  test_dequantize_fp8_grouped_impl<fp8e4m3, bf16>(
      {{16, 32}, {64, 128}, {32, 64}},
      DType::kFloat8E4M3, DType::kBFloat16);
}

TEST_F(DequantizeFP8GroupedTestSuite, E4M3_to_FP16_Varying) {
  test_dequantize_fp8_grouped_impl<fp8e4m3, fp16>(
      {{8, 16}, {128, 64}, {64, 32}},
      DType::kFloat8E4M3, DType::kFloat16);
}

TEST_F(DequantizeFP8GroupedTestSuite, E5M2_to_FP32_Varying) {
  test_dequantize_fp8_grouped_impl<fp8e5m2, float>(
      {{32, 32}, {16, 64}, {128, 32}},
      DType::kFloat8E5M2, DType::kFloat32);
}

} // namespace
