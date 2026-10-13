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
// The derivative is also checked against util/math.h with the ulp bound stated in the API.
// With NVTE_ACTIVATION_TABLE_EXACT_DSILU=1 the dSiLU window starts at 2^-25 and dSiLU must match
// util/math.h exactly.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/activation_table.cuh"
#include "util/math.h"

namespace {

using transformer_engine::Empty;

constexpr int kNumInputs = 1 << 16;

// Below-window bounds and linear forms of the specification.
constexpr uint32_t kDgeluWindowLow = 0x3980;       // 2^-12
constexpr uint32_t kDsiluWindowLow = 0x3B00;       // 2^-9
constexpr uint32_t kExactDsiluWindowLow = 0x3300;  // 2^-25

bool exact_dsilu() {
  const char *env = std::getenv("NVTE_ACTIVATION_TABLE_EXACT_DSILU");
  return env != nullptr && std::string(env) == "1";
}

__device__ float spec(const NVTEActivationTableKind kind, const uint32_t bits,
                      const uint32_t dsilu_window_low) {
  const float x = __uint_as_float(bits << 16);
  const uint32_t magnitude = bits & 0x7fffu;
  switch (kind) {
    case kNVTEActivationTableGelu:
      if (bits == 0xff80u) return -0.0f;
      return __bfloat162float(__float2bfloat16_rn(transformer_engine::gelu<float, float>(x, {})));
    case kNVTEActivationTableSilu:
      if (bits == 0xff80u) return -0.0f;
      return __bfloat162float(__float2bfloat16_rn(transformer_engine::silu<float, float>(x, {})));
    case kNVTEActivationTableDgelu:
      if (magnitude < kDgeluWindowLow) return fmaf(0.79788456f, x, 0.5f);
      return transformer_engine::dgelu<float, float>(x, {});
    default:
      if (magnitude < dsilu_window_low) return fmaf(0.5f, x, 0.5f);
      return transformer_engine::dsilu<float, float>(x, {});
  }
}

__device__ float scalar_math(const NVTEActivationTableKind kind, const float x) {
  switch (kind) {
    case kNVTEActivationTableDgelu:
      return transformer_engine::dgelu<float, float>(x, {});
    default:
      return transformer_engine::dsilu<float, float>(x, {});
  }
}

template <NVTEActivationTableKind kKind>
struct Op;
template <>
struct Op<kNVTEActivationTableGelu> {
  static constexpr float (*value)(float, const Empty &) = transformer_engine::gelu<float, float>;
};
template <>
struct Op<kNVTEActivationTableSilu> {
  static constexpr float (*value)(float, const Empty &) = transformer_engine::silu<float, float>;
};
template <>
struct Op<kNVTEActivationTableDgelu> {
  static constexpr float (*value)(float, const Empty &) = transformer_engine::dgelu<float, float>;
};
template <>
struct Op<kNVTEActivationTableDsilu> {
  static constexpr float (*value)(float, const Empty &) = transformer_engine::dsilu<float, float>;
};

template <NVTEActivationTableKind kKind, int kUse>
__global__ void lookup_kernel(const void *table, const uint32_t dsilu_window_low, float *looked_up,
                              float *expected, float *math_h) {
  const uint32_t bits = blockIdx.x * blockDim.x + threadIdx.x;
  if (bits >= kNumInputs) return;
  namespace at = transformer_engine::activation_table;
  const float x = __uint_as_float(bits << 16);
  looked_up[bits] = at::evaluate<kUse, Empty, Op<kKind>::value>(x, table);
  expected[bits] = spec(kKind, bits, dsilu_window_low);
  math_h[bits] = (kKind == kNVTEActivationTableDgelu || kKind == kNVTEActivationTableDsilu)
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

struct DeviceTable {
  void *ptr = nullptr;
  size_t bytes = 0;
  explicit DeviceTable(const NVTEActivationTableKind kind) {
    bytes = nvte_activation_table_size(kind);
    EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
    transformer_engine::TensorWrapper table(ptr, std::vector<size_t>{bytes},
                                            transformer_engine::DType::kByte);
    nvte_build_activation_table(kind, table.data(), 0);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  ~DeviceTable() { cudaFree(ptr); }
};

template <NVTEActivationTableKind kKind>
void check_table(const int max_ulp_below_window) {
  DeviceTable table(kKind);
  float *d = nullptr;
  ASSERT_EQ(cudaMalloc(&d, 3 * kNumInputs * sizeof(float)), cudaSuccess);
  namespace at = transformer_engine::activation_table;
  const bool exact = kKind == kNVTEActivationTableDsilu && exact_dsilu();
  const uint32_t dsilu_window_low = exact ? kExactDsiluWindowLow : kDsiluWindowLow;
  if (exact) {
    if constexpr (kKind == kNVTEActivationTableDsilu) {
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
  check_table<kNVTEActivationTableGelu>(-1);
}
TEST(ActivationTableTest, SiluMatchesSpecification) {
  check_table<kNVTEActivationTableSilu>(-1);
}
TEST(ActivationTableTest, DgeluMatchesSpecification) {
  check_table<kNVTEActivationTableDgelu>(1);
}
TEST(ActivationTableTest, DsiluMatchesSpecification) {
  check_table<kNVTEActivationTableDsilu>(exact_dsilu() ? 0 : 2);
}

TEST(ActivationTableTest, SizesIdentifyKind) {
  const size_t sizes[] = {nvte_activation_table_size(kNVTEActivationTableGelu),
                          nvte_activation_table_size(kNVTEActivationTableSilu),
                          nvte_activation_table_size(kNVTEActivationTableDgelu),
                          nvte_activation_table_size(kNVTEActivationTableDsilu)};
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) EXPECT_NE(sizes[i], sizes[j]);
  }
}

TEST(ActivationTableTest, RejectsTableOfWrongSize) {
  const size_t bytes = nvte_activation_table_size(kNVTEActivationTableDgelu);
  void *ptr = nullptr;
  ASSERT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
  transformer_engine::TensorWrapper table(ptr, std::vector<size_t>{bytes},
                                          transformer_engine::DType::kByte);
  EXPECT_THROW(nvte_build_activation_table(kNVTEActivationTableDsilu, table.data(), 0),
               std::runtime_error);
  cudaFree(ptr);
}
