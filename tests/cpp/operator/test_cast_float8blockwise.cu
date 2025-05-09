/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

struct QuantizationOptions {
  bool force_pow_2_scales = false;
  float amax_epsilon = 0.0;
  size_t block_scaling_dim = 2u;
};

constexpr size_t kBlockLen = 128;

enum ProcessingMethod {
  CAST_ONLY,
  // CAST_DBIAS,
  // CAST_DBIAS_DACT,
  // CAST_DACT,
  // CAST_ACT
};

enum ActivationType {
  Identity,
  // GeLU,
  // SiLU,
  // ReLU,
  // QGeLU,
  // SReLU
};

template <typename InputType, typename OutputType>
void scales_from_amax(float amax, const QuantizationOptions& opts, float* qscale_out,
                      float* qscale_inv_out) {
  float input_type_max_val = Quantized_Limits<InputType>::max();
  float quant_type_max_val = Quantized_Limits<OutputType>::max();
  float eps = opts.amax_epsilon;
  amax = std::max(amax, eps);
  float qscale = quant_type_max_val / amax;
  if (std::isinf(qscale)) {
    qscale = input_type_max_val;
  }
  if (std::isnan(qscale) || amax == 0) {
    qscale = 1.0;
  }

  if (opts.force_pow_2_scales && qscale != 0.0) {
    uint32_t scale_bits = *reinterpret_cast<uint32_t*>(&qscale);
    // Scale must be positive, shift it
    uint8_t exp = scale_bits >> 23;
    ASSERT_FALSE(exp == 0) << "Subnormals in this path is a logic error.";
    qscale = ldexpf(1.0f, static_cast<int32_t>(exp) - 127);
  }

  float qscale_inv = 1.0 / qscale;
  *qscale_out = qscale;
  *qscale_inv_out = qscale_inv;
}

template <typename InputType, typename OutputType>
void ref_quantize(const ProcessingMethod processing_method, const InputType* input,
                  const std::pair<size_t, size_t>& input_hw, OutputType* output, float* scale_inv,
                  OutputType* output_t, float* scale_inv_t, const QuantizationOptions& opts) {
  constexpr size_t kBlockLenX = kBlockLen;
  constexpr size_t kBlockLenY = kBlockLen;

  auto quantize_element = [](InputType element, float qscale) -> OutputType {
    // Scale in FP32 and cast result to nearest FP8.
    return static_cast<OutputType>(float(element) * qscale);
  };

  size_t height = input_hw.first;
  size_t width = input_hw.second;
  size_t blocks_x = (width + kBlockLenX - 1) / kBlockLenX;
  size_t blocks_y = (height + kBlockLenY - 1) / kBlockLenY;
  // Find the absolute maximum value in the block
  for (size_t block_x = 0; block_x < blocks_x; ++block_x) {
    for (size_t block_y = 0; block_y < blocks_y; ++block_y) {
      float amax = 0.0f;
      // Calculate amax for a tile.
      for (size_t i = 0; i < kBlockLenX; ++i) {
        for (size_t j = 0; j < kBlockLenY; ++j) {
          size_t x_pos = i + block_x * kBlockLenX;
          size_t y_pos = j + block_y * kBlockLenY;
          if (y_pos >= height || x_pos >= width) {
            continue;
          }
          float val = static_cast<float>(input[y_pos * width + x_pos]);
          amax = std::max(amax, std::abs(val));
        }
      }

      // We've calculated amax for a tile. Calculate scale and
      // scale_inv and populate outputs.
      float qscale, qscale_inv;
      scales_from_amax<InputType, OutputType>(amax, opts, &qscale, &qscale_inv);

      // NOTE: This reference function outputs contigous scale tensors.
      // It calculates a naive scale data format. Strides are handled
      // in comparison.
      if (scale_inv != nullptr) {
        scale_inv[block_y * blocks_x + block_x] = qscale_inv;
      }
      if (scale_inv_t != nullptr) {
        scale_inv_t[block_x * blocks_y + block_y] = qscale_inv;
      }

      for (size_t i = 0; i < kBlockLenX; ++i) {
        for (size_t j = 0; j < kBlockLenY; ++j) {
          size_t x_pos = i + block_x * kBlockLenX;
          size_t y_pos = j + block_y * kBlockLenY;
          if (y_pos >= height || x_pos >= width) {
            continue;
          }
          if (output != nullptr) {
            output[y_pos * width + x_pos] = quantize_element(input[y_pos * width + x_pos], qscale);
          }
          if (output_t != nullptr) {
            output_t[x_pos * height + y_pos] =
                quantize_element(input[y_pos * width + x_pos], qscale);
          }
        }
      }
    }
  }
}

template <typename InputType, typename OutputType>
void ref_quantize_onedimensional_blocks(const ProcessingMethod processing_method,
                                        const InputType* input,
                                        const std::pair<size_t, size_t>& input_hw,
                                        OutputType* output, float* scale_inv, OutputType* output_t,
                                        float* scale_inv_t, const QuantizationOptions& opts) {
  float input_type_max_val = Quantized_Limits<InputType>::max();
  float quant_type_max_val = Quantized_Limits<OutputType>::max();

  constexpr size_t kBlockLenX = kBlockLen;

  auto quantize_element = [](InputType element, float qscale) -> OutputType {
    // Scale in FP32 and cast result to nearest FP8.
    return static_cast<OutputType>(float(element) * qscale);
  };

  size_t height = input_hw.first;
  size_t width = input_hw.second;
  size_t blocks_x = (width + kBlockLenX - 1) / kBlockLenX;
  size_t blocks_x_t = (height + kBlockLenX - 1) / kBlockLenX;
  if (output != nullptr && scale_inv != nullptr) {
    // Find the absolute maximum value in the block
    for (size_t block_x = 0; block_x < blocks_x; ++block_x) {
      for (size_t y = 0; y < height; ++y) {
        float amax = 0.0f;
        // Calculate amax for a tile.
        for (size_t i = 0; i < kBlockLenX; ++i) {
          size_t x_pos = i + block_x * kBlockLenX;
          if (x_pos >= width) {
            continue;
          }
          float val = static_cast<float>(input[y * width + x_pos]);
          amax = std::max(amax, std::abs(val));
        }

        // We've calculated amax for a tile. Calculate scale and
        // scale_inv and populate outputs.
        float qscale, qscale_inv;
        scales_from_amax<InputType, OutputType>(amax, opts, &qscale, &qscale_inv);

        scale_inv[y + height * block_x] = qscale_inv;

        for (size_t i = 0; i < kBlockLenX; ++i) {
          size_t x_pos = i + block_x * kBlockLenX;
          if (x_pos >= width) {
            continue;
          }
          output[y * width + x_pos] = quantize_element(input[y * width + x_pos], qscale);
        }
      }
    }
  }
  if (output_t != nullptr && scale_inv_t != nullptr) {
    // Find the absolute maximum value in the block
    for (size_t block_x_t = 0; block_x_t < blocks_x_t; ++block_x_t) {
      for (size_t x = 0; x < width; ++x) {
        float amax = 0.0f;
        // Calculate amax for a tile.
        for (size_t i = 0; i < kBlockLenX; ++i) {
          size_t y_pos = i + block_x_t * kBlockLenX;
          if (y_pos >= height) {
            continue;
          }
          float val = static_cast<float>(input[x + y_pos * width]);
          amax = std::max(amax, std::abs(val));
        }

        // We've calculated amax for a tile. Calculate scale and
        // scale_inv and populate outputs.
        float qscale, qscale_inv;
        scales_from_amax<InputType, OutputType>(amax, opts, &qscale, &qscale_inv);

        scale_inv_t[x + width * block_x_t] = qscale_inv;

        for (size_t i = 0; i < kBlockLenX; ++i) {
          size_t y_pos = i + block_x_t * kBlockLenX;
          if (y_pos >= height) {
            continue;
          }
          output_t[x * height + y_pos] = quantize_element(input[y_pos * width + x], qscale);
        }
      }
    }
  }
}

inline size_t scale_align_stride(size_t inner_elements) {
  return ((inner_elements + 4u - 1u) / 4u) * 4u;
};

void compare_scaling_factors(const std::string& name, const float* test, const float* ref,
                             const size_t row_blocks, const size_t col_blocks,
                             const size_t test_stride, const size_t ref_stride) {
  for (int i = 0; i < row_blocks; ++i) {
    for (int j = 0; j < col_blocks; ++j) {
      const int test_idx = i * test_stride + j;
      const int ref_idx = i * ref_stride + j;
      ASSERT_FALSE(test[test_idx] != ref[ref_idx])
          << "Error in " << name << std::endl
          << "Mismatch: " << test[test_idx] << " vs " << ref[ref_idx] << " at index " << test_idx
          << "," << ref_idx;
    }
  }
}

void compare_scaling_factors_one_dimensional_blocks(const std::string& name, const float* test,
                                                    const float* ref, const size_t rows,
                                                    const size_t col_blocks) {
  const size_t test_stride = scale_align_stride(rows);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < col_blocks; ++j) {
      const int test_idx = i + test_stride * j;
      const int ref_idx = i + rows * j;
      ASSERT_FALSE(test[test_idx] != ref[ref_idx])
          << "Error in " << name << std::endl
          << "Mismatch: " << test[test_idx] << " vs " << ref[ref_idx] << " at index " << test_idx
          << "," << ref_idx;
    }
  }
}

template <typename InputType, typename OutputType>
void runTestCase(const ProcessingMethod processing_method, const std::vector<size_t>& shape,
                 const bool rowwise, const bool colwise, InputsFillCase fill_case,
                 const QuantizationOptions& opts) {
  using namespace test;
  using EncodingType = fp32;
  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  const size_t rows = first_dimension(shape);
  const size_t cols = last_dimension(shape);

  size_t blocks_x = (cols + kBlockLen - 1) / kBlockLen;
  size_t blocks_y = (rows + kBlockLen - 1) / kBlockLen;

  Tensor input("input", shape, itype);
  Tensor grad("grad", shape, itype);
  Tensor output_c("output_c", shape, otype, rowwise, colwise,
                  opts.block_scaling_dim == 2 ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);
  Tensor output_dbias("output_dbias", std::vector<size_t>{cols}, itype);

  std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(rows * cols);
  std::unique_ptr<OutputType[]> ref_output_t = std::make_unique<OutputType[]>(rows * cols);
  std::unique_ptr<float[]> ref_scale_inv = std::make_unique<float[]>(blocks_y * blocks_x);
  std::unique_ptr<float[]> ref_scale_inv_t = std::make_unique<float[]>(blocks_y * blocks_x);

  if (!rowwise) {
    ref_output = nullptr;
    ref_scale_inv = nullptr;
  }
  if (!colwise) {
    ref_output_t = nullptr;
    ref_scale_inv_t = nullptr;
  }

  fillCase<EncodingType>(&input, fill_case);
  fillUniform(&grad);

  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(opts.force_pow_2_scales);
  quant_config.set_amax_epsilon(opts.amax_epsilon);
  Tensor workspace;
  switch (processing_method) {
    case ProcessingMethod::CAST_ONLY: {
      nvte_quantize_v2(input.data(), output_c.data(), quant_config, nullptr);
      break;
    }
  }

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  ref_quantize<InputType, OutputType>(processing_method, input.rowwise_cpu_dptr<InputType>(),
                                      {rows, cols}, ref_output.get(), ref_scale_inv.get(),
                                      ref_output_t.get(), ref_scale_inv_t.get(), opts);

  float atol = 0.0;
  float rtol = 0.0;

  if (rowwise) {
    compareResults("output_c", output_c, ref_output.get(), true, atol, rtol);
    compare_scaling_factors("scale_inv", output_c.rowwise_cpu_scale_inv_ptr<float>(),
                            ref_scale_inv.get(), blocks_y, blocks_x, scale_align_stride(blocks_x),
                            blocks_x);
  }
  if (colwise) {
    compareResults("output_c_t", output_c, ref_output_t.get(), false, atol, rtol);
    compare_scaling_factors("scale_inv_t", output_c.columnwise_cpu_scale_inv_ptr<float>(),
                            ref_scale_inv_t.get(), blocks_x, blocks_y, scale_align_stride(blocks_y),
                            blocks_y);
  }
}

template <typename InputType, typename OutputType>
void runTestCaseOneDimensionalBlocks(const ProcessingMethod processing_method,
                                     const std::vector<size_t>& shape, const bool rowwise,
                                     const bool colwise, InputsFillCase fill_case,
                                     const QuantizationOptions& opts) {
  using namespace test;
  using EncodingType = fp32;
  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  const size_t rows = first_dimension(shape);
  const size_t cols = last_dimension(shape);

  size_t blocks_x = (cols + kBlockLen - 1) / kBlockLen;
  size_t blocks_x_t = (rows + kBlockLen - 1) / kBlockLen;

  Tensor input("input", shape, itype);
  Tensor grad("grad", shape, itype);
  Tensor output_c("output_c", shape, otype, rowwise, colwise,
                  opts.block_scaling_dim == 2 ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);
  Tensor output_dbias("output_dbias", std::vector<size_t>{cols}, itype);

  std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(rows * cols);
  std::unique_ptr<OutputType[]> ref_output_t = std::make_unique<OutputType[]>(rows * cols);
  std::unique_ptr<float[]> ref_scale_inv = std::make_unique<float[]>(rows * blocks_x);
  std::unique_ptr<float[]> ref_scale_inv_t = std::make_unique<float[]>(cols * blocks_x_t);

  if (!rowwise) {
    ref_output = nullptr;
    ref_scale_inv = nullptr;
  }
  if (!colwise) {
    ref_output_t = nullptr;
    ref_scale_inv_t = nullptr;
  }

  fillCase<EncodingType>(&input, fill_case);
  fillUniform(&grad);

  Tensor workspace;
  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(opts.force_pow_2_scales);
  quant_config.set_amax_epsilon(opts.amax_epsilon);
  switch (processing_method) {
    case ProcessingMethod::CAST_ONLY: {
      nvte_quantize_v2(input.data(), output_c.data(), quant_config, nullptr);
      break;
    }
  }

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  ref_quantize_onedimensional_blocks<InputType, OutputType>(
      processing_method, input.rowwise_cpu_dptr<InputType>(), {rows, cols}, ref_output.get(),
      ref_scale_inv.get(), ref_output_t.get(), ref_scale_inv_t.get(), opts);

  float atol = 0.0;
  float rtol = 0.0;

  if (rowwise) {
    compareResults("output_c", output_c, ref_output.get(), true, atol, rtol);
    compare_scaling_factors_one_dimensional_blocks("scale_inv",
                                                   output_c.rowwise_cpu_scale_inv_ptr<float>(),
                                                   ref_scale_inv.get(), rows, blocks_x);
  }
  if (colwise) {
    compareResults("output_c_t", output_c, ref_output_t.get(), false, atol, rtol);
    compare_scaling_factors_one_dimensional_blocks("scale_inv_t",
                                                   output_c.columnwise_cpu_scale_inv_ptr<float>(),
                                                   ref_scale_inv_t.get(), cols, blocks_x_t);
  }
}

std::vector<std::vector<size_t>> matrix_sizes = {
    {1, 16}, {65, 96}, {256, 256}, {993, 512},
    {256, 65536}, {4096, 1632}, {1024, 1},
    {16, 512}, {1024}, {8, 32, 1024}, {16, 8, 4, 512},
};

std::vector<InputsFillCase> input_scenarios = {
    InputsFillCase::uniform,
};

std::vector<ProcessingMethod> processing_methods = {
    ProcessingMethod::CAST_ONLY,
    // ProcessingMethod::CAST_DBIAS,
    // ProcessingMethod::CAST_DBIAS_DACT,
    // ProcessingMethod::CAST_DACT,
    // ProcessingMethod::CAST_ACT,
};

// Only GeLU activation tests are supported
std::vector<ActivationType> Activation_types = {
    ActivationType::Identity,
    // ActivationType::GeLU,
    // ActivationType::SiLU,
    // ActivationType::ReLU,
    // ActivationType::QGeLU,
    // ActivationType::SReLU,
};


std::vector<float> amax_epsilons = {
    0.0f,
    1.0f, // Make large to be observable.

};

}  // namespace

class FusedCastFloat8BlockwiseTestSuite
    : public ::testing::TestWithParam<std::tuple<
          ProcessingMethod, ActivationType, std::vector<size_t>, transformer_engine::DType,
          transformer_engine::DType, InputsFillCase, bool, float, bool>> {};

class FusedCastFloat8VectorwiseTestSuite
    : public ::testing::TestWithParam<std::tuple<
          ProcessingMethod, ActivationType, std::vector<size_t>, transformer_engine::DType,
          transformer_engine::DType, InputsFillCase, bool, float, bool>> {};

#define DACT_FUNC_SWITCH(OP_FUNC_TYPE, OP, ...) \
  switch (OP_FUNC_TYPE) {                       \
    case ActivationType::Identity: {            \
      constexpr auto OP = &identity;            \
      {                                         \
        __VA_ARGS__                             \
      }                                         \
    } break;                                    \
  }

#define ACT_FUNC_SWITCH(OP_FUNC_TYPE, OP, ...) \
  switch (OP_FUNC_TYPE) {                      \
    case ActivationType::Identity: {           \
      constexpr auto OP = &identity;           \
      {                                        \
        __VA_ARGS__                            \
      }                                        \
    } break;                                   \
  }

TEST_P(FusedCastFloat8BlockwiseTestSuite, TestFusedCastFloat8Blockwise) {
  if (getDeviceComputeCapability() < hopperComputeCapability) {
    GTEST_SKIP();
  }

  using namespace transformer_engine;
  using namespace test;

  const ProcessingMethod processing_method = std::get<0>(GetParam());
  const ActivationType Act_type = std::get<1>(GetParam());
  const auto matrix_size = std::get<2>(GetParam());
  const DType input_type = std::get<3>(GetParam());
  const DType output_type = std::get<4>(GetParam());
  const InputsFillCase fill_case = std::get<5>(GetParam());
  const bool colwise = std::get<6>(GetParam());
  const bool rowwise = true;
  const float eps = std::get<7>(GetParam());
  const bool force_pow_2 = std::get<8>(GetParam());

  QuantizationOptions q_opts;
  q_opts.force_pow_2_scales = force_pow_2;
  q_opts.amax_epsilon = eps;
  q_opts.block_scaling_dim = 2u;

  if (colwise && matrix_size.size() < 2) {
    // test_common Tensor initialization code does not
    // handle this case.
    GTEST_SKIP();
  }
  // Skips non Act tests if the Activation type is not an identity
  if (  // (processing_method == ProcessingMethod::CAST_ONLY || processing_method == ProcessingMethod::CAST_DBIAS)
      (processing_method == ProcessingMethod::CAST_ONLY) && Act_type != ActivationType::Identity) {
    GTEST_SKIP();
  }
  // Skips Act tests if the Activation is an identity
  // if ((processing_method == ProcessingMethod::CAST_DBIAS_DACT
  //     || processing_method == ProcessingMethod::CAST_DACT
  //     || processing_method == ProcessingMethod::CAST_ACT) && (Act_type == ActivationType::Identity)) {
  //     GTEST_SKIP();
  // }

  DACT_FUNC_SWITCH(
      Act_type, OP,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
          input_type, InputType,
          TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(
              output_type, OutputType,
              runTestCase<InputType, OutputType>(processing_method, matrix_size, rowwise, colwise,
                                                 fill_case, q_opts););););
}

TEST_P(FusedCastFloat8VectorwiseTestSuite, TestFusedCastFloat8Vectorwise) {
  if (getDeviceComputeCapability() < hopperComputeCapability) {
    GTEST_SKIP();
  }

  using namespace transformer_engine;
  using namespace test;

  const ProcessingMethod processing_method = std::get<0>(GetParam());
  const ActivationType Act_type = std::get<1>(GetParam());
  const auto matrix_size = std::get<2>(GetParam());
  const DType input_type = std::get<3>(GetParam());
  const DType output_type = std::get<4>(GetParam());
  const InputsFillCase fill_case = std::get<5>(GetParam());
  const bool colwise = std::get<6>(GetParam());
  const bool rowwise = true;
  const float eps = std::get<7>(GetParam());
  const bool force_pow_2 = std::get<8>(GetParam());

  QuantizationOptions q_opts;
  q_opts.force_pow_2_scales = force_pow_2;
  q_opts.amax_epsilon = eps;
  q_opts.block_scaling_dim = 1u;

  if (colwise && matrix_size.size() < 2) {
    // test_common Tensor initialization code does not
    // handle this case.
    GTEST_SKIP();
  }
  // Skips non Act tests if the Activation type is not an identity
  if (  // (processing_method == ProcessingMethod::CAST_ONLY || processing_method == ProcessingMethod::CAST_DBIAS)
      (processing_method == ProcessingMethod::CAST_ONLY) && Act_type != ActivationType::Identity) {
    GTEST_SKIP();
  }
  // Skips Act tests if the Activation is an identity
  // if ((processing_method == ProcessingMethod::CAST_DBIAS_DACT
  //     || processing_method == ProcessingMethod::CAST_DACT
  //     || processing_method == ProcessingMethod::CAST_ACT) && (Act_type == ActivationType::Identity)) {
  //     GTEST_SKIP();
  // }

  DACT_FUNC_SWITCH(
      Act_type, OP,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
          input_type, InputType,
          TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(
              output_type, OutputType,
              runTestCaseOneDimensionalBlocks<InputType, OutputType>(
                  processing_method, matrix_size, rowwise, colwise, fill_case, q_opts););););
}

std::string to_string(const ProcessingMethod method) {
  switch (method) {
    case ProcessingMethod::CAST_ONLY:
      return "CAST_ONLY";
    // case ProcessingMethod::CAST_DBIAS:      return "CAST_DBIAS";
    // case ProcessingMethod::CAST_DBIAS_DACT: return "CAST_DBIAS_DACT";
    // case ProcessingMethod::CAST_DACT:       return "CAST_DACT";
    // case ProcessingMethod::CAST_ACT:        return "CAST_ACT";
    default:
      return "";
  }
}

std::string to_string(const ActivationType Act_type) {
  switch (Act_type) {
    case ActivationType::Identity:
      return "Identity";
    // case ActivationType::GeLU:      return "GeLU";
    // case ActivationType::SiLU:      return "SiLU";
    // case ActivationType::ReLU:      return "ReLU";
    // case ActivationType::QGeLU:     return "QGeLU";
    // case ActivationType::SReLU:     return "SReLU";
    default:
      return "";
  }
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, FusedCastFloat8BlockwiseTestSuite,
    ::testing::Combine(::testing::ValuesIn(processing_methods),
                       ::testing::ValuesIn(Activation_types), ::testing::ValuesIn(matrix_sizes),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
                       ::testing::ValuesIn(input_scenarios), ::testing::Values(true, false),
                       ::testing::ValuesIn(amax_epsilons), ::testing::Values(true, false)),
    [](const testing::TestParamInfo<FusedCastFloat8BlockwiseTestSuite::ParamType>& info) {
      std::string name =
          to_string(std::get<0>(info.param)) + "X" + to_string(std::get<1>(info.param));
      const auto& shape = std::get<2>(info.param);
      for (const auto& s : shape) {
        name += "X" + std::to_string(s);
      }
      name += "X" + test::typeName(std::get<3>(info.param)) + "X" +
              test::typeName(std::get<4>(info.param)) + "X" +
              test::caseName(std::get<5>(info.param)) + "X" +
              std::to_string(std::get<6>(info.param)) + "X" +
              std::to_string(std::get<7>(info.param) != 0.0f) + "X" +
              std::to_string(std::get<8>(info.param));
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, FusedCastFloat8VectorwiseTestSuite,
    ::testing::Combine(::testing::ValuesIn(processing_methods),
                       ::testing::ValuesIn(Activation_types), ::testing::ValuesIn(matrix_sizes),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
                       ::testing::ValuesIn(input_scenarios), ::testing::Values(true, false),
                       ::testing::ValuesIn(amax_epsilons), ::testing::Values(true, false)),
    [](const testing::TestParamInfo<FusedCastFloat8VectorwiseTestSuite::ParamType>& info) {
      std::string name =
          to_string(std::get<0>(info.param)) + "X" + to_string(std::get<1>(info.param));
      const auto& shape = std::get<2>(info.param);
      for (const auto& s : shape) {
        name += "X" + std::to_string(s);
      }
      name += "X" + test::typeName(std::get<3>(info.param)) + "X" +
              test::typeName(std::get<4>(info.param)) + "X" +
              test::caseName(std::get<5>(info.param)) + "X" +
              std::to_string(std::get<6>(info.param)) + "X" +
              std::to_string(std::get<7>(info.param) != 0.0f) + "X" +
              std::to_string(std::get<8>(info.param));
      return name;
    });
