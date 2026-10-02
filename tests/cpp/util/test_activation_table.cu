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

#include "util/activation_table.cuh"

namespace te = transformer_engine;
namespace at = te::activation_table;

namespace {

constexpr int kNumBf16Values = 1 << 16;

// For every BF16 bit pattern x: OP from activation_2x, and from the table with x in the low and
// in the high half of the pair (the other half holds a different value).
// out is laid out as [computed, low half, high half][input].
template <float (*OP)(float, const te::Empty &)>
__global__ void activation_table_kernel(float *out, int *ran) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) *ran = 1;
  if (i >= kNumBf16Values) return;
  const unsigned other = (i * 40503u) & 0xffffu;
  const float x = __uint_as_float(i << 16);
  out[i] = te::activation_2x<te::Empty, OP>({x, x}, {}).x;
  out[kNumBf16Values + i] = at::lookup<OP>((other << 16) | i, at::d_table<OP>).x;
  out[2 * kNumBf16Values + i] = at::lookup<OP>((i << 16) | other, at::d_table<OP>).y;
#endif
}

// Bit-identical, or both NaN.
bool same_value(float a, float b) {
  uint32_t ua, ub;
  std::memcpy(&ua, &a, 4);
  std::memcpy(&ub, &b, 4);
  return ua == ub || (a != a && b != b);
}

template <float (*OP)(float, const te::Empty &)>
void check_table(const char *name) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 10) {
    GTEST_SKIP() << "Packed FP32x2 arithmetic requires compute capability 10.0 or newer";
  }

  at::ensure_table<OP>(0);
  constexpr int kNumOutputs = 3;
  float *out = nullptr;
  int *ran = nullptr;
  ASSERT_EQ(cudaMalloc(&out, kNumOutputs * kNumBf16Values * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&ran, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(ran, 0, sizeof(int)), cudaSuccess);
  activation_table_kernel<OP><<<kNumBf16Values / 256, 256>>>(out, ran);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int ran_host = 0;
  std::vector<float> host(kNumOutputs * kNumBf16Values);
  ASSERT_EQ(cudaMemcpy(&ran_host, ran, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(host.data(), out, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(out);
  cudaFree(ran);
  if (ran_host == 0) {
    GTEST_SKIP() << "Test kernel was not compiled for compute capability 10.0 or newer";
  }

  const float *ref = host.data();
  for (int half = 0; half < 2; ++half) {
    const float *table = ref + (half + 1) * kNumBf16Values;
    int mismatches = 0;
    for (int i = 0; i < kNumBf16Values; ++i) {
      if (!same_value(ref[i], table[i])) {
        if (mismatches == 0) {
          ADD_FAILURE() << name << (half == 0 ? " (low half)" : " (high half)")
                        << " differs for BF16 input 0x" << std::hex << i << std::dec
                        << ": computed " << ref[i] << ", table " << table[i];
        }
        ++mismatches;
      }
    }
    EXPECT_EQ(mismatches, 0) << name << (half == 0 ? " low half" : " high half");
  }
}

}  // namespace

// Kernels read these activations from the tables in place of computing them, so the two must
// agree for every BF16 input, inside and outside the tabulated window.
TEST(UtilTest, ActivationTableMatchesActivation) {
  check_table<te::dgelu<float, float>>("dgelu");
  check_table<te::dsilu<float, float>>("dsilu");
}
