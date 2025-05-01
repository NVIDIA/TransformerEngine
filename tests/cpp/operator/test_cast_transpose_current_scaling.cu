/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/recipe.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename InputType, typename OutputType>
void compute_ref(const InputType *data, OutputType *output_c, OutputType *output_t,
                 const size_t N, const size_t H,
                 float *amax, float scale) {
  using compute_t = float;
  compute_t current_max = -1e100;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      current_max = fmaxf(current_max, fabsf(current));
      output_c[i * H + j] = OutputType(scale * current);
      output_t[j * N + i] = OutputType(scale * current);
    }
  }
}

template <typename InputType, typename OutputType>
void compute_amax_scale_ref(const InputType *data,
                 const size_t N, const size_t H,
                 float *amax_ptr, float *scale_ptr, float* scale_inv_ptr,
                 float max_fp8, float epsilon) {
  using compute_t = float;
  compute_t current_max = -1e100;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      current_max = fmaxf(current_max, fabsf(current));
    }
  }
  *amax_ptr = current_max;

  // compute scale from amax
  float clamp_amax = current_max;
  if (current_max <= epsilon){
      clamp_amax = epsilon;
  }

  float scale = 1.f;
  float scale_inv = 1.f;

  if (isinf(clamp_amax) || clamp_amax == 0.f) {
      *scale_ptr = scale;
      *scale_inv_ptr = scale_inv;
      return;
  }

  // use ieee_div in CPU
  scale = max_fp8 / clamp_amax;

  // The amax is too small that the scale becoming infinite in FP32. In other word,
  // the scale is not representable in FP32.
  if (isinf(scale)) {
    scale = std::numeric_limits<float>::max();
  }

  if (isnan(scale)) {
    scale = 1.f;
  }

  scale_inv = 1.0f / scale;

  *scale_ptr = scale;
  *scale_inv_ptr = scale_inv;
}

// current tensor scaling test
template <typename InputType, typename OutputType>
void performTest(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  bool is_out_fp8 = isFp8Type(otype);

  // find out max fp8 value
  float max_fp8;
  if (is_out_fp8){
    switch (otype) {
      case DType::kFloat8E5M2: {
          max_fp8 = Quantized_Limits<fp8e5m2>::max();
      } break;
      case DType::kFloat8E4M3: {
          max_fp8 = Quantized_Limits<fp8e4m3>::max();
      } break;
      default:
        NVTE_ERROR("Invalid type.");
    }
  }

  Tensor input("input", std::vector<size_t>{ N, H }, itype);
  Tensor output("output", std::vector<size_t>{ N, H }, otype, true, true);

  std::unique_ptr<OutputType[]> ref_output_c = std::make_unique<OutputType[]>(N * H);
  std::unique_ptr<OutputType[]> ref_output_t = std::make_unique<OutputType[]>(N * H);

  fillUniform(&input);

  // compute amax
  float amax_to_check = 0.0f;
  if (is_out_fp8){
    nvte_compute_amax(input.data(), output.data(), 0);
    QuantizationConfigWrapper config;
    nvte_compute_scale_from_amax(output.data(), config, 0);
    // avoid atomic amax update in cuda cast kernels because of current per-tensor scaling
    amax_to_check = output.amax();
    output.set_tensor_amax_nullptr();
  }
  nvte_quantize(input.data(), output.data(), 0);

  float ref_amax;
  float ref_scale;
  float ref_scale_inv;
  if (is_out_fp8){
    compute_amax_scale_ref<InputType, OutputType>(input.rowwise_cpu_dptr<InputType>(),
                                      N, H, &ref_amax, &ref_scale, &ref_scale_inv, max_fp8, 0.0f);
  }

  compute_ref<InputType, OutputType>(input.rowwise_cpu_dptr<InputType>(), ref_output_c.get(),
                                     ref_output_t.get(), N, H, nullptr,
                                     is_out_fp8 ? output.scale() : 1.0f );

  cudaDeviceSynchronize();

  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  if (isFp8Type(otype)) {
    auto [atol_fp32, rtol_fp32] = getTolerances(DType::kFloat32);
    compareResults("amax", amax_to_check, ref_amax, 0.0f, rtol_fp32);
    compareResults("scale", output.scale(), ref_scale, 0.0f, rtol_fp32);
    compareResults("scale_inv", output.rowwise_scale_inv(), ref_scale_inv, 0.0f, rtol_fp32);
    compareResults("scale_inv_columnwise", output.columnwise_cpu_scale_inv_ptr<float>()[0], ref_scale_inv, 0.0f, rtol_fp32);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output, ref_output_c.get(), true, 0.0f, rtol);
  compareResults("output_t", output, ref_output_t.get(), false, 0.0f, rtol);
}

std::vector<std::pair<size_t, size_t>> test_cases = {{2048, 12288},
                                                     {768, 1024},
                                                     {256, 65536},
                                                     {65536, 128},
                                                     {256, 256},
                                                     {120, 2080},
                                                     {8, 8},
                                                     {1, 3221},       // Prime 456
                                                     {2333, 1},       // Prime 345
                                                     {1481, 677}};    // Primes 234, 123
}  // namespace

class CTCSTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                               transformer_engine::DType,
                                                               std::pair<size_t, size_t>>> {};

TEST_P(CTCSTestSuite, TestCastTransposeCS) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      // current tensor scaling
      performTest<InputType, OutputType>(size.first, size.second);
    );
  );
}



INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  CTCSTestSuite,
  ::testing::Combine(
      ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
      ::testing::ValuesIn(test::all_fp_types),
      ::testing::ValuesIn(test_cases)),
  [](const testing::TestParamInfo<CTCSTestSuite::ParamType>& info) {
    std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                       test::typeName(std::get<1>(info.param)) + "X" +
                       std::to_string(std::get<2>(info.param).first) + "X" +
                       std::to_string(std::get<2>(info.param).second);
    return name;
  });
