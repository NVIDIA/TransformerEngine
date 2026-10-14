/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Every BF16 input read from each activation table, against the specification written out
// here independently of util/activation_table.cuh:
//   - forward: util/math.h rounded to BF16, except that -inf gives -0;
//   - derivative: util/math.h, except below the table window, where it is the linear Taylor
//     term rounded once.
// The derivative is also checked against util/math.h with the ulp bound stated in
// util/activation_table.cuh. The tables are the ones TE builds and caches per device.
// With NVTE_ACTIVATION_TABLE_EXACT_DSILU=1 the dSiLU window starts at 2^-25 and dSiLU must match
// util/math.h exactly.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../test_common.h"
#include "util/activation_table.cuh"
#include "util/math.h"

namespace {

using transformer_engine::Empty;
using transformer_engine::activation_table::kDgelu;
using transformer_engine::activation_table::kDsilu;
using transformer_engine::activation_table::kGelu;
using transformer_engine::activation_table::Kind;
using transformer_engine::activation_table::kSilu;

constexpr int kNumInputs = 1 << 16;

// Below-window bounds and linear forms of the specification.
constexpr uint32_t kDgeluWindowLow = 0x3980;       // 2^-12
constexpr uint32_t kDsiluWindowLow = 0x3B00;       // 2^-9
constexpr uint32_t kExactDsiluWindowLow = 0x3300;  // 2^-25

bool exact_dsilu() {
  const char *env = std::getenv("NVTE_ACTIVATION_TABLE_EXACT_DSILU");
  return env != nullptr && std::string(env) == "1";
}

__device__ float spec(const Kind kind, const uint32_t bits,
                      const uint32_t dsilu_window_low) {
  const float x = __uint_as_float(bits << 16);
  const uint32_t magnitude = bits & 0x7fffu;
  switch (kind) {
    case kGelu:
      if (bits == 0xff80u) return -0.0f;
      return __bfloat162float(__float2bfloat16_rn(transformer_engine::gelu<float, float>(x, {})));
    case kSilu:
      if (bits == 0xff80u) return -0.0f;
      return __bfloat162float(__float2bfloat16_rn(transformer_engine::silu<float, float>(x, {})));
    case kDgelu:
      if (magnitude < kDgeluWindowLow) return fmaf(0.79788456f, x, 0.5f);
      return transformer_engine::dgelu<float, float>(x, {});
    default:
      if (magnitude < dsilu_window_low) return fmaf(0.5f, x, 0.5f);
      return transformer_engine::dsilu<float, float>(x, {});
  }
}

__device__ float scalar_math(const Kind kind, const float x) {
  switch (kind) {
    case kDgelu:
      return transformer_engine::dgelu<float, float>(x, {});
    default:
      return transformer_engine::dsilu<float, float>(x, {});
  }
}

template <Kind kKind>
struct Op;
template <>
struct Op<kGelu> {
  static constexpr float (*value)(float, const Empty &) = transformer_engine::gelu<float, float>;
};
template <>
struct Op<kSilu> {
  static constexpr float (*value)(float, const Empty &) = transformer_engine::silu<float, float>;
};
template <>
struct Op<kDgelu> {
  static constexpr float (*value)(float, const Empty &) = transformer_engine::dgelu<float, float>;
};
template <>
struct Op<kDsilu> {
  static constexpr float (*value)(float, const Empty &) = transformer_engine::dsilu<float, float>;
};

template <Kind kKind, int kUse>
__global__ void lookup_kernel(const void *table, const uint32_t dsilu_window_low, float *looked_up,
                              float *expected, float *math_h) {
  const uint32_t bits = blockIdx.x * blockDim.x + threadIdx.x;
  if (bits >= kNumInputs) return;
  namespace at = transformer_engine::activation_table;
  const float x = __uint_as_float(bits << 16);
  looked_up[bits] = at::evaluate<kUse, Empty, Op<kKind>::value>(x, table);
  expected[bits] = spec(kKind, bits, dsilu_window_low);
  math_h[bits] = (kKind == kDgelu || kKind == kDsilu)
                     ? scalar_math(kKind, x)
                     : 0.0f;
}

uint32_t float_bits(const float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

bool is_nan(const float f) { return (float_bits(f) & 0x7fffffffu) > 0x7f800000u; }

// Monotone integer order of floats, for ulp distances.
int64_t ordered(const float f) {
  const int32_t i = static_cast<int32_t>(float_bits(f));
  return i < 0 ? -static_cast<int64_t>(i & 0x7fffffff) : i;
}

template <Kind kKind>
void check_table(const int max_ulp_below_window) {
  namespace at = transformer_engine::activation_table;
  const at::TableRef table = at::device_table(kKind);
  const bool exact = kKind == kDsilu && exact_dsilu();
  ASSERT_NE(table.ptr, nullptr);
  ASSERT_EQ(table.use, exact ? at::kExactDsiluTable : at::kDefaultTable);
  float *d = nullptr;
  ASSERT_EQ(cudaMalloc(&d, 3 * kNumInputs * sizeof(float)), cudaSuccess);
  const uint32_t dsilu_window_low = exact ? kExactDsiluWindowLow : kDsiluWindowLow;
  if (exact) {
    if constexpr (kKind == kDsilu) {
      lookup_kernel<kKind, at::kExactDsiluTable><<<kNumInputs / 256, 256>>>(
          table.ptr, dsilu_window_low, d, d + kNumInputs, d + 2 * kNumInputs);
    }
  } else {
    lookup_kernel<kKind, at::kDefaultTable><<<kNumInputs / 256, 256>>>(
        table.ptr, dsilu_window_low, d, d + kNumInputs, d + 2 * kNumInputs);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> h(3 * kNumInputs);
  ASSERT_EQ(cudaMemcpy(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d);
  const float *looked_up = h.data(), *expected = h.data() + kNumInputs,
              *math_h = h.data() + 2 * kNumInputs;

  int mismatches = 0;
  int64_t worst_ulp = 0;
  for (uint32_t bits = 0; bits < kNumInputs; ++bits) {
    const float got = looked_up[bits], want = expected[bits];
    const bool same = float_bits(got) == float_bits(want) || (is_nan(got) && is_nan(want));
    if (!same && mismatches++ < 8) {
      ADD_FAILURE() << "input 0x" << std::hex << bits << ": table 0x" << float_bits(got)
                    << ", specification 0x" << float_bits(want);
    }
    if (max_ulp_below_window >= 0 && !is_nan(math_h[bits])) {
      worst_ulp = std::max<int64_t>(worst_ulp, std::llabs(ordered(got) - ordered(math_h[bits])));
    }
  }
  EXPECT_EQ(mismatches, 0);
  if (max_ulp_below_window >= 0) {
    EXPECT_LE(worst_ulp, max_ulp_below_window) << "distance from util/math.h in ulp";
  }
}

}  // namespace

TEST(ActivationTableTest, GeluMatchesSpecification) {
  check_table<kGelu>(-1);
}
TEST(ActivationTableTest, SiluMatchesSpecification) {
  check_table<kSilu>(-1);
}
TEST(ActivationTableTest, DgeluMatchesSpecification) {
  check_table<kDgelu>(1);
}
TEST(ActivationTableTest, DsiluMatchesSpecification) {
  check_table<kDsilu>(exact_dsilu() ? 0 : 2);
}

TEST(ActivationTableTest, OneTablePerDevice) {
  namespace at = transformer_engine::activation_table;
  int num_devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);
  int current = 0;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  for (const Kind kind : {kGelu, kSilu, kDgelu, kDsilu}) {
    std::vector<const void *> tables;
    for (int device = 0; device < num_devices; ++device) {
      ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
      const void *first = at::device_table(kind).ptr;
      EXPECT_EQ(at::device_table(kind).ptr, first) << "kind " << kind << ", device " << device;
      for (const void *other : tables) EXPECT_NE(first, other);
      tables.push_back(first);
    }
  }
  ASSERT_EQ(cudaSetDevice(current), cudaSuccess);
}

// The first use of a table, inside CUDA graph capture, must build the table outside the capture,
// and the replayed graph must read it. This runs in a fresh process (a death test), so that no
// earlier test has built the table.
TEST(ActivationTableDeathTest, FirstUseInsideGraphCapture) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto body = [] {
    const std::vector<size_t> shape{256, 256};
    test::Tensor x("x", shape, transformer_engine::DType::kBFloat16);
    test::fillUniform(&x);
    test::Tensor captured("captured", shape, transformer_engine::DType::kFloat8E4M3, true, true,
                          NVTE_MXFP8_1D_SCALING);
    test::Tensor eager("eager", shape, transformer_engine::DType::kFloat8E4M3, true, true,
                       NVTE_MXFP8_1D_SCALING);
    cudaStream_t stream;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) std::exit(2);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess) std::exit(3);
    nvte_gelu(x.data(), captured.data(), stream);
    if (cudaStreamEndCapture(stream, &graph) != cudaSuccess) std::exit(4);
    if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) std::exit(5);
    if (cudaGraphLaunch(exec, stream) != cudaSuccess) std::exit(6);
    nvte_gelu(x.data(), eager.data(), stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) std::exit(7);
    captured.to_cpu();
    eager.to_cpu();
    const size_t n = shape[0] * shape[1];
    const bool same =
        std::memcmp(captured.rowwise_cpu_dptr<test::fp8e4m3>(),
                    eager.rowwise_cpu_dptr<test::fp8e4m3>(), n) == 0 &&
        std::memcmp(captured.columnwise_cpu_dptr<test::fp8e4m3>(),
                    eager.columnwise_cpu_dptr<test::fp8e4m3>(), n) == 0;
    std::exit(same ? 0 : 1);
  };
  EXPECT_EXIT(body(), testing::ExitedWithCode(0), "");
}
