/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The register-tiled SwiGLU / dSwiGLU MXFP8 kernel must write exactly the bytes of the generic
// gated kernel. Each case runs both (NVTE_MXFP8_REGTILE=0 keeps the generic kernel) into outputs
// pre-filled with a marker byte, so they must also write the same set of bytes. The AllBF16 fill
// puts every BF16 value in the activated half.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class GatedFill { Normal, Wide, Specials, AllBF16 };

class ScopedEnv {
 public:
  ScopedEnv(const char *name, const char *value) : name_(name) {
    const char *old = std::getenv(name);
    if (old != nullptr) old_ = old;
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (old_) {
      setenv(name_, old_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char *name_;
  std::optional<std::string> old_;
};

void fill(Tensor *t, const size_t rows, const size_t cols, const GatedFill kind, const int seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal(0.0f, 3.0f);
  std::uniform_int_distribution<int> exponent(-10, 9);
  const float specials[] = {std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(), 0.0f, -0.0f, 1e-30f, -1e30f};
  bf16 *p = t->rowwise_cpu_dptr<bf16>();
  for (size_t r = 0; r < rows; ++r) {
    // Wide: every 32x32 block gets its own power-of-two scale.
    for (size_t c = 0; c < cols; ++c) {
      float v = normal(gen);
      if (kind == GatedFill::AllBF16 && c < cols / 2) {
        // The activated half (the first cols / 2 columns) walks through every BF16 bit pattern.
        const uint16_t bits = static_cast<uint16_t>(r * (cols / 2) + c);
        std::memcpy(&p[r * cols + c], &bits, sizeof(bits));
        continue;
      }
      if (kind == GatedFill::Wide) {
        std::mt19937 block_gen(seed * 7919 + (r / 32) * 104729 + c / 32);
        v *= std::ldexp(1.0f, exponent(block_gen));
      } else if (kind == GatedFill::Specials && gen() % 61 == 0) {
        v = specials[gen() % 7];
      }
      p[r * cols + c] = static_cast<bf16>(v);
    }
  }
  t->from_cpu();
}

size_t count_byte_mismatches(const uint8_t *a, const uint8_t *b, const size_t n) {
  size_t m = 0;
  for (size_t i = 0; i < n; ++i) m += (a[i] != b[i]);
  return m;
}

void run_case(const bool bwd, const GatedFill kind, const size_t rows, const size_t cols,
              const bool swizzled, const DType otype) {
  Tensor input("input", std::vector<size_t>{rows, 2 * cols}, DType::kBFloat16);
  Tensor grad("grad", std::vector<size_t>{rows, cols}, DType::kBFloat16);
  fill(&input, rows, 2 * cols, kind, 1);
  fill(&grad, rows, cols,
       kind == GatedFill::Specials || kind == GatedFill::AllBF16 ? GatedFill::Normal : kind, 2);
  const size_t out_cols = bwd ? 2 * cols : cols;
  const std::vector<size_t> out_shape{rows, out_cols};
  Tensor outs[2] = {
      Tensor("regtile", out_shape, otype, true, true, NVTE_MXFP8_1D_SCALING),
      Tensor("generic", out_shape, otype, true, true, NVTE_MXFP8_1D_SCALING)};
  for (int i = 0; i < 2; ++i) {
    outs[i].set_with_gemm_swizzled_scales(swizzled);
    for (const auto param : {kNVTERowwiseData, kNVTEColumnwiseData, kNVTERowwiseScaleInv,
                             kNVTEColumnwiseScaleInv}) {
      const NVTEBasicTensor b = nvte_get_tensor_param(outs[i].data(), param);
      size_t bytes = b.data_ptr == nullptr ? 0 : 1;
      for (size_t d = 0; d < b.shape.ndim; ++d) bytes *= b.shape.data[d];
      if (bytes != 0) ASSERT_EQ(cudaMemset(b.data_ptr, 0xA5, bytes), cudaSuccess);
    }
    const ScopedEnv regtile("NVTE_MXFP8_REGTILE", i == 0 ? "1" : "0");
    if (bwd) {
      nvte_dswiglu(grad.data(), input.data(), outs[i].data(), 0);
    } else {
      nvte_swiglu(input.data(), outs[i].data(), 0);
    }
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  for (auto &out : outs) out.to_cpu();
  const size_t n = rows * out_cols;
  // The FP8 data as raw bytes, whichever FP8 type the output holds.
  const auto rowwise_bytes = [&](Tensor &t) {
    return otype == DType::kFloat8E5M2
               ? reinterpret_cast<const uint8_t *>(t.rowwise_cpu_dptr<fp8e5m2>())
               : reinterpret_cast<const uint8_t *>(t.rowwise_cpu_dptr<fp8e4m3>());
  };
  const auto columnwise_bytes = [&](Tensor &t) {
    return otype == DType::kFloat8E5M2
               ? reinterpret_cast<const uint8_t *>(t.columnwise_cpu_dptr<fp8e5m2>())
               : reinterpret_cast<const uint8_t *>(t.columnwise_cpu_dptr<fp8e4m3>());
  };
  EXPECT_EQ(count_byte_mismatches(rowwise_bytes(outs[0]), rowwise_bytes(outs[1]), n), 0u)
      << "rowwise data";
  EXPECT_EQ(count_byte_mismatches(columnwise_bytes(outs[0]), columnwise_bytes(outs[1]), n), 0u)
      << "colwise data";
  EXPECT_EQ(count_byte_mismatches(outs[0].rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                  outs[1].rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                  product(outs[0].rowwise_scale_inv_shape())),
            0u)
      << "rowwise scales";
  EXPECT_EQ(count_byte_mismatches(outs[0].columnwise_cpu_scale_inv_ptr<uint8_t>(),
                                  outs[1].columnwise_cpu_scale_inv_ptr<uint8_t>(),
                                  product(outs[0].columnwise_scale_inv_shape())),
            0u)
      << "colwise scales";
}

using Params = std::tuple<bool /*bwd*/, GatedFill, bool /*swizzled*/, std::pair<size_t, size_t>,
                          DType /*output*/>;

class GatedRegtileTestSuite : public ::testing::TestWithParam<Params> {};

TEST_P(GatedRegtileTestSuite, MatchesGenericKernel) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 10) {
    GTEST_SKIP() << "This MXFP8 quantize kernel requires compute capability 10.0 or newer";
  }
  const auto [bwd, kind, swizzled, dims, otype] = GetParam();
  run_case(bwd, kind, dims.first, dims.second, swizzled, otype);
}

std::string case_name(const testing::TestParamInfo<Params> &info) {
  const auto [bwd, kind, swizzled, dims, otype] = info.param;
  const char *fills[] = {"Normal", "Wide", "Specials", "AllBF16"};
  return std::string(bwd ? "DSwiGLU" : "SwiGLU") + "X" + fills[static_cast<int>(kind)] + "X" +
         (swizzled ? "Swizzled" : "Compact") + "X" + std::to_string(dims.first) + "X" +
         std::to_string(dims.second) + (otype == DType::kFloat8E5M2 ? "XE5M2" : "XE4M3");
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GatedRegtileTestSuite,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(GatedFill::Normal, GatedFill::Wide, GatedFill::Specials),
                       ::testing::Bool(),
                       ::testing::Values(std::pair<size_t, size_t>{1024, 2048},
                                         std::pair<size_t, size_t>{4160, 2048},
                                         std::pair<size_t, size_t>{128, 256},
                                         // Row and column tails (K % 32 == 0).
                                         std::pair<size_t, size_t>{993, 7200},
                                         std::pair<size_t, size_t>{1, 32},
                                         std::pair<size_t, size_t>{33, 288},
                                         std::pair<size_t, size_t>{100, 2080}),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2)),
    case_name);

// Every BF16 value in the activated half: 64 rows x 1024 columns.
INSTANTIATE_TEST_SUITE_P(AllBF16, GatedRegtileTestSuite,
                         ::testing::Combine(::testing::Bool(), ::testing::Values(GatedFill::AllBF16),
                                            ::testing::Bool(),
                                            ::testing::Values(std::pair<size_t, size_t>{64, 1024}),
                                            ::testing::Values(DType::kFloat8E4M3,
                                                              DType::kFloat8E5M2)),
                         case_name);

}  // namespace
