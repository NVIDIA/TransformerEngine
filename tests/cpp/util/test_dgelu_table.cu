/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "util/dgelu_table.cuh"

namespace te = transformer_engine;

namespace {

constexpr int kNumBf16Values = 1 << 16;

// For every BF16 bit pattern x: dgelu from the table, with x in the low and in the high half of
// the pair, and from activation_2x.
__global__ void dgelu_table_kernel(float *ref, float *table_lo, float *table_hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kNumBf16Values) return;
  const unsigned other = (i * 40503u) & 0xffffu;
  const float x = __uint_as_float(i << 16);
  ref[i] = te::activation_2x<te::Empty, te::dgelu<float, float>>({x, x}, {}).x;
  table_lo[i] = te::dgelu_table::lookup((other << 16) | i, te::dgelu_table::d_table).x;
  table_hi[i] = te::dgelu_table::lookup((i << 16) | other, te::dgelu_table::d_table).y;
#endif
}

// Bit-identical, or both NaN.
bool same_value(float a, float b) {
  uint32_t ua, ub;
  std::memcpy(&ua, &a, 4);
  std::memcpy(&ub, &b, 4);
  return ua == ub || (a != a && b != b);
}

}  // namespace

// Kernels read dGeLU from the table in place of computing it, so the two must agree for every
// BF16 input, inside and outside the tabulated window.
TEST(UtilTest, DgeluTableMatchesActivation) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 10) {
    GTEST_SKIP() << "Packed FP32x2 arithmetic requires compute capability 10.0 or newer";
  }

  te::dgelu_table::ensure_table(0);
  constexpr int kNumOutputs = 3;
  float *device_buffer = nullptr;
  ASSERT_EQ(cudaMalloc(&device_buffer, kNumOutputs * kNumBf16Values * sizeof(float)), cudaSuccess);
  dgelu_table_kernel<<<kNumBf16Values / 256, 256>>>(device_buffer, device_buffer + kNumBf16Values,
                                                    device_buffer + 2 * kNumBf16Values);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> host(kNumOutputs * kNumBf16Values);
  ASSERT_EQ(
      cudaMemcpy(host.data(), device_buffer, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
      cudaSuccess);
  cudaFree(device_buffer);

  const float *ref = host.data();
  const char *names[2] = {"low half", "high half"};
  for (int k = 0; k < 2; ++k) {
    const float *table = ref + (k + 1) * kNumBf16Values;
    int mismatches = 0;
    for (int i = 0; i < kNumBf16Values; ++i) {
      if (!same_value(ref[i], table[i])) {
        if (mismatches == 0) {
          ADD_FAILURE() << names[k] << " differs for BF16 input 0x" << std::hex << i << std::dec
                        << ": computed " << ref[i] << ", table " << table[i];
        }
        ++mismatches;
      }
    }
    EXPECT_EQ(mismatches, 0) << names[k];
  }
}
