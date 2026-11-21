/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

constexpr size_t kWorkspaceBytes = 32ull * 1024 * 1024;

/* Random multiples of 1/8 in [-1, 1]
 *
 * These are exact in FP8 E4M3 and all higher-precision dtypes, and
 * partial sums of their products are exact in FP32. With power-of-two
 * scaling factors, the GEMM result is exact before it is cast to the
 * output dtype, regardless of accumulation order.
 */
std::vector<float> random_values(size_t size, std::mt19937 *gen) {
  std::uniform_int_distribution<int> dist(-8, 8);
  std::vector<float> values(size);
  for (auto &v : values) {
    v = dist(*gen) / 8.f;
  }
  return values;
}

/* Cast values to a dtype, after applying a scale */
std::vector<unsigned char> cast_values(const std::vector<float> &values, DType dtype,
                                       float scale = 1.f) {
  std::vector<unsigned char> ret(values.size() * typeToNumBits(dtype) / 8);
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, T,
    T *data = reinterpret_cast<T *>(ret.data());
    for (size_t i = 0; i < values.size(); ++i) {
      const double x = scale * values[i];
      data[i] = static_cast<T>(x);
    }
  );
  return ret;
}

/* Set row-wise data of a tensor on GPU */
void set_data(Tensor *t, const std::vector<float> &values) {
  const auto data = cast_values(values, t->dtype());
  NVTE_CHECK_CUDA(cudaMemcpy(t->rowwise_dptr(), data.data(), data.size(),
                             cudaMemcpyHostToDevice));
}

/* Reference for D = alpha * op(A) * op(B) + beta * C
 *
 * Matrices are row-major. A is [m, k] if transposed and [k, m]
 * otherwise, B is [k, n] if transposed and [n, k] otherwise, and C and
 * D are [n, m].
 */
std::vector<float> reference_gemm(const std::vector<float> &A, bool transa,
                                  const std::vector<float> &B, bool transb,
                                  const std::vector<float> &C, float alpha, float beta,
                                  size_t m, size_t n, size_t k) {
  std::vector<float> D(n * m);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      float acc = 0;
      for (size_t l = 0; l < k; ++l) {
        const float a = transa ? A[j * k + l] : A[l * m + j];
        const float b = transb ? B[l * n + i] : B[i * k + l];
        acc += a * b;
      }
      D[i * m + j] = alpha * acc + beta * C[i * m + j];
    }
  }
  return D;
}

using GemmTestParams = std::tuple<std::pair<DType, DType>,               // Input, output dtype
                                  std::pair<bool, bool>,                 // transa, transb
                                  std::tuple<size_t, size_t, size_t>>;   // m, n, k

// Dims are multiples of 16 to satisfy FP8 GEMM alignment requirements.
// k <= 128 keeps FP8 outputs below the E4M3 max.
const std::vector<std::tuple<size_t, size_t, size_t>> gemm_shapes = {
    {32, 64, 128}, {256, 128, 64}, {48, 80, 96}};

std::string gemm_test_name(const testing::TestParamInfo<GemmTestParams> &info) {
  const auto [input_dtype, output_dtype] = std::get<0>(info.param);
  const auto [transa, transb] = std::get<1>(info.param);
  const auto [m, n, k] = std::get<2>(info.param);
  return typeName(input_dtype) + "X" + typeName(output_dtype) + "X" + (transa ? "T" : "N") +
         (transb ? "T" : "N") + "X" + std::to_string(m) + "X" + std::to_string(n) + "X" +
         std::to_string(k);
}

/* Check D = alpha * op(A) * op(B) + beta * C
 *
 * If with_C is true, C and D are in different buffers. Otherwise, C
 * is NULL and treated as zero.
 */
void run_gemm_test(const GemmTestParams &params, bool with_C) {
  const auto [input_dtype, output_dtype] = std::get<0>(params);
  const auto [transa, transb] = std::get<1>(params);
  const auto [m, n, k] = std::get<2>(params);
  const bool fp8_input = isFp8Type(input_dtype);
  const bool fp8_output = isFp8Type(output_dtype);
  if (fp8_input && getDeviceComputeCapability() < 89) {
    GTEST_SKIP() << "FP8 GEMM requires compute capability 8.9+";
  }

  // FP8 output requires FP16 or BF16 C
  const DType C_dtype = fp8_output ? DType::kBFloat16 : output_dtype;

  // Tensors
  const auto A_shape = transa ? std::vector<size_t>{m, k} : std::vector<size_t>{k, m};
  const auto B_shape = transb ? std::vector<size_t>{k, n} : std::vector<size_t>{n, k};
  Tensor A("A", A_shape, input_dtype);
  Tensor B("B", B_shape, input_dtype);
  Tensor C("C", std::vector<size_t>{n, m}, C_dtype);
  Tensor D("D", std::vector<size_t>{n, m}, output_dtype);
  Tensor workspace("workspace", std::vector<size_t>{kWorkspaceBytes}, DType::kByte);

  // Initialize data. D is filled with values that would corrupt the
  // result if it were read as the accumulation input.
  std::mt19937 gen(1234);
  const auto A_values = random_values(m * k, &gen);
  const auto B_values = random_values(n * k, &gen);
  const auto C_values = random_values(n * m, &gen);
  set_data(&A, A_values);
  set_data(&B, B_values);
  set_data(&C, C_values);
  set_data(&D, random_values(n * m, &gen));

  // Scaling factors are powers of two so the result is exact
  const float alpha = 2.f;
  const float beta = 0.5f;
  float A_scale_inv = 1.f, B_scale_inv = 1.f, D_scale = 1.f;
  if (fp8_input) {
    A_scale_inv = 2.f;
    B_scale_inv = 0.25f;
    A.set_scale_inv(A_scale_inv);
    B.set_scale_inv(B_scale_inv);
  }
  if (fp8_output) {
    D_scale = 2.f;
    D.set_scale(D_scale);
  }

  // Launch GEMM
  MatmulConfigWrapper config;
  config.set_use_split_accumulator(true);
  nvte_cublas_gemm_v2(transa, transb, &alpha, A.data(), B.data(), &beta,
                      with_C ? C.data() : nullptr, D.data(), workspace.data(), config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  // Check output
  const auto D_ref = reference_gemm(A_values, transa, B_values, transb, C_values,
                                    alpha * A_scale_inv * B_scale_inv, with_C ? beta : 0.f,
                                    m, n, k);
  const auto D_ref_cast = cast_values(D_ref, output_dtype, D_scale);
  const auto [atol, rtol] = getTolerances(output_dtype);
  compareResults("D", D, D_ref_cast.data(), true, atol, rtol);
  if (fp8_output) {
    float D_amax_ref = 0;
    for (const float x : D_ref) {
      D_amax_ref = std::max(D_amax_ref, std::fabs(x));
    }
    const auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("D_amax", D.amax(), D_amax_ref, atol_amax, rtol_amax);
  }

  // Check that C is unchanged
  const auto C_ref = cast_values(C_values, C_dtype);
  compareResults("C", C, C_ref.data(), true, 0, 0);
}

/* Random values that are exact in an MX format
 *
 * Each 32-element block is grid values times a power of two, with the
 * largest grid value present so the E8M0 scale is exact. With the grids
 * below, FP32 accumulation is exact for k <= 512.
 */
std::vector<float> exact_mx_values(size_t rows, size_t cols, bool rowwise_blocks,
                                   const std::vector<float> &grid, std::mt19937 *gen) {
  constexpr size_t block_size = 32;
  std::uniform_int_distribution<size_t> magnitude_dist(0, grid.size() - 1);
  std::uniform_int_distribution<int> exponent_dist(-1, 1);
  std::uniform_int_distribution<size_t> position_dist(0, block_size - 1);
  std::bernoulli_distribution sign_dist;
  std::vector<float> values(rows * cols);
  const size_t num_blocks = rows * cols / block_size;
  for (size_t block = 0; block < num_blocks; ++block) {
    const float block_scale = std::ldexp(1.f, exponent_dist(*gen));
    const size_t max_position = position_dist(*gen);
    for (size_t pos = 0; pos < block_size; ++pos) {
      const float magnitude = pos == max_position ? grid.back() : grid[magnitude_dist(*gen)];
      const float value = (sign_dist(*gen) ? -1.f : 1.f) * magnitude * block_scale;
      const size_t idx = rowwise_blocks ? block * block_size + pos
                                        : ((block / cols) * block_size + pos) * cols + block % cols;
      values[idx] = value;
    }
  }
  return values;
}

const std::vector<float> e2m1_grid = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
const std::vector<float> mxfp8_grid = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

enum class Accumulation { None, OutOfPlace, InPlace };

using MXFP4GemmTestParams = std::tuple<DType,                                // Output dtype
                                       bool,                                 // transa
                                       std::tuple<size_t, size_t, size_t>>;  // m, n, k

// MXFP8 quantization with GEMM-swizzled scales needs n and k to be
// multiples of 128, and the NN layout needs m to be a multiple of 128.
const std::vector<std::tuple<size_t, size_t, size_t>> mxfp4_gemm_shapes = {
    {128, 128, 128}, {256, 128, 512}, {384, 256, 256}};

std::string mxfp4_gemm_test_name(const testing::TestParamInfo<MXFP4GemmTestParams> &info) {
  const auto [m, n, k] = std::get<2>(info.param);
  return typeName(std::get<0>(info.param)) + "X" + (std::get<1>(info.param) ? "TN" : "NN") + "X" +
         std::to_string(m) + "X" + std::to_string(n) + "X" + std::to_string(k);
}

/* Check D = alpha * op(A) * B + beta * C with MXFP4 A and MXFP8 B
 *
 * The TN layout uses row-wise MXFP4 data (fprop) and the NN layout uses
 * column-wise MXFP4 data (dgrad). Both operands are quantized exactly, so
 * the result matches an FP32 reference on the original values.
 */
void run_mxfp4_mxfp8_gemm_test(const MXFP4GemmTestParams &params, Accumulation accumulation) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP() << "MXFP4 x MXFP8 GEMM requires compute capability 10.0+";
  }
  const DType output_dtype = std::get<0>(params);
  const bool transa = std::get<1>(params);
  const auto [m, n, k] = std::get<2>(params);

  // Quantize operands
  const auto A_shape = transa ? std::vector<size_t>{m, k} : std::vector<size_t>{k, m};
  const std::vector<size_t> B_shape{n, k};
  std::mt19937 gen(1234);
  const auto A_values = exact_mx_values(A_shape[0], A_shape[1], transa, e2m1_grid, &gen);
  const auto B_values = exact_mx_values(n, k, true, mxfp8_grid, &gen);
  Tensor A_fp32("A_fp32", A_shape, DType::kFloat32);
  Tensor B_fp32("B_fp32", B_shape, DType::kFloat32);
  set_data(&A_fp32, A_values);
  set_data(&B_fp32, B_values);
  Tensor A("A", A_shape, DType::kFloat4E2M1, transa, !transa, NVTE_MXFP4_1D_SCALING);
  Tensor B("B", B_shape, DType::kFloat8E4M3, true, false, NVTE_MXFP8_1D_SCALING);
  A.set_with_gemm_swizzled_scales(true);
  B.set_with_gemm_swizzled_scales(true);
  nvte_quantize(A_fp32.data(), A.data(), 0);
  nvte_quantize(B_fp32.data(), B.data(), 0);

  // Output and accumulation input
  Tensor C("C", std::vector<size_t>{n, m}, output_dtype);
  Tensor D("D", std::vector<size_t>{n, m}, output_dtype);
  Tensor workspace("workspace", std::vector<size_t>{kWorkspaceBytes}, DType::kByte);
  const auto C_values = random_values(n * m, &gen);
  set_data(&C, C_values);
  set_data(&D, accumulation == Accumulation::InPlace ? C_values : random_values(n * m, &gen));
  NVTETensor C_arg = nullptr;
  if (accumulation == Accumulation::OutOfPlace) {
    C_arg = C.data();
  } else if (accumulation == Accumulation::InPlace) {
    C_arg = D.data();
  }

  // Launch GEMM
  const float alpha = 2.f;
  const float beta = 0.5f;
  MatmulConfigWrapper config;
  nvte_cublas_gemm_v2(transa, false, &alpha, A.data(), B.data(), &beta, C_arg, D.data(),
                      workspace.data(), config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  // Check output
  const auto D_ref = reference_gemm(A_values, transa, B_values, false, C_values, alpha,
                                    accumulation == Accumulation::None ? 0.f : beta, m, n, k);
  const auto D_ref_cast = cast_values(D_ref, output_dtype);
  const auto [atol, rtol] = getTolerances(output_dtype);
  compareResults("D", D, D_ref_cast.data(), true, atol, rtol);

  // Check that C is unchanged
  const auto C_ref = cast_values(C_values, output_dtype);
  compareResults("C", C, C_ref.data(), true, 0, 0);
}

}  // namespace

class GemmTestSuite : public ::testing::TestWithParam<GemmTestParams> {};

/* GEMM with C and D in different buffers */
TEST_P(GemmTestSuite, OutOfPlaceAccumulation) { run_gemm_test(GetParam(), true); }

/* GEMM without accumulation input */
TEST_P(GemmTestSuite, NoAccumulationInput) { run_gemm_test(GetParam(), false); }

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GemmTestSuite,
    ::testing::Combine(::testing::Values(std::make_pair(DType::kFloat32, DType::kFloat32),
                                         std::make_pair(DType::kFloat16, DType::kFloat16),
                                         std::make_pair(DType::kBFloat16, DType::kBFloat16)),
                       ::testing::Values(std::make_pair(true, false), std::make_pair(false, false),
                                         std::make_pair(false, true)),
                       ::testing::ValuesIn(gemm_shapes)),
    gemm_test_name);

// FP8 GEMMs use the TN layout, which only needs row-wise FP8 data on
// all supported architectures
INSTANTIATE_TEST_SUITE_P(
    OperatorTestFP8, GemmTestSuite,
    ::testing::Combine(::testing::Values(std::make_pair(DType::kFloat8E4M3, DType::kBFloat16),
                                         std::make_pair(DType::kFloat8E4M3, DType::kFloat8E4M3)),
                       ::testing::Values(std::make_pair(true, false)),
                       ::testing::ValuesIn(gemm_shapes)),
    gemm_test_name);

class MXFP4MXFP8GemmTestSuite : public ::testing::TestWithParam<MXFP4GemmTestParams> {};

TEST_P(MXFP4MXFP8GemmTestSuite, NoAccumulationInput) {
  run_mxfp4_mxfp8_gemm_test(GetParam(), Accumulation::None);
}

TEST_P(MXFP4MXFP8GemmTestSuite, OutOfPlaceAccumulation) {
  run_mxfp4_mxfp8_gemm_test(GetParam(), Accumulation::OutOfPlace);
}

TEST_P(MXFP4MXFP8GemmTestSuite, InPlaceAccumulation) {
  run_mxfp4_mxfp8_gemm_test(GetParam(), Accumulation::InPlace);
}

INSTANTIATE_TEST_SUITE_P(OperatorTest, MXFP4MXFP8GemmTestSuite,
                         ::testing::Combine(::testing::Values(DType::kBFloat16, DType::kFloat32),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(mxfp4_gemm_shapes)),
                         mxfp4_gemm_test_name);

/* GEMM with C that is incompatible with D */
TEST(GemmTest, InvalidAccumulationInput) {
  const size_t m = 32, n = 64, k = 16;
  Tensor A("A", std::vector<size_t>{m, k}, DType::kBFloat16);
  Tensor B("B", std::vector<size_t>{n, k}, DType::kBFloat16);
  Tensor D("D", std::vector<size_t>{n, m}, DType::kBFloat16);
  Tensor C_wrong_dtype("C_wrong_dtype", std::vector<size_t>{n, m}, DType::kFloat32);
  Tensor C_wrong_shape("C_wrong_shape", std::vector<size_t>{m, n}, DType::kBFloat16);
  Tensor workspace("workspace", std::vector<size_t>{kWorkspaceBytes}, DType::kByte);
  const float alpha = 1.f;
  const float beta = 1.f;
  EXPECT_THROW(nvte_cublas_gemm_v2(true, false, &alpha, A.data(), B.data(), &beta,
                                   C_wrong_dtype.data(), D.data(), workspace.data(), nullptr, 0),
               std::runtime_error);
  EXPECT_THROW(nvte_cublas_gemm_v2(true, false, &alpha, A.data(), B.data(), &beta,
                                   C_wrong_shape.data(), D.data(), workspace.data(), nullptr, 0),
               std::runtime_error);
}

/* GEMM with FP8 output and C that is incompatible with D */
TEST(GemmTest, InvalidAccumulationInputWithFP8Output) {
  if (getDeviceComputeCapability() < 89) {
    GTEST_SKIP() << "FP8 GEMM requires compute capability 8.9+";
  }
  const size_t m = 32, n = 64, k = 16;
  Tensor A("A", std::vector<size_t>{m, k}, DType::kFloat8E4M3);
  Tensor B("B", std::vector<size_t>{n, k}, DType::kFloat8E4M3);
  Tensor D("D", std::vector<size_t>{n, m}, DType::kFloat8E4M3);
  Tensor C_fp32("C_fp32", std::vector<size_t>{n, m}, DType::kFloat32);
  Tensor workspace("workspace", std::vector<size_t>{kWorkspaceBytes}, DType::kByte);
  const float alpha = 1.f;
  const float zero = 0.f;
  const float beta = 1.f;

  // FP8 output cannot be accumulated in-place, even if beta is zero
  EXPECT_THROW(nvte_cublas_gemm_v2(true, false, &alpha, A.data(), B.data(), &zero, D.data(),
                                   D.data(), workspace.data(), nullptr, 0),
               std::runtime_error);
  EXPECT_THROW(nvte_cublas_gemm_v2(true, false, &alpha, A.data(), B.data(), &beta, D.data(),
                                   D.data(), workspace.data(), nullptr, 0),
               std::runtime_error);

  // FP8 output requires FP16 or BF16 C
  EXPECT_THROW(nvte_cublas_gemm_v2(true, false, &alpha, A.data(), B.data(), &beta,
                                   C_fp32.data(), D.data(), workspace.data(), nullptr, 0),
               std::runtime_error);
}
