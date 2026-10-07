/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "util/packed_activation.cuh"

namespace {

// Inputs: every BF16 and FP16 bit pattern, then FP32 prefixes with low bits 0 and 0xffff.
constexpr int kNumInputs = 4 << 16;
constexpr int kNumOps = 4;
const char *const kOpNames[kNumOps] = {"gelu", "dgelu", "silu", "dsilu"};

__device__ float input_value(unsigned i) {
  if (i < (1u << 16)) return __uint_as_float(i << 16);
  if (i < (2u << 16)) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(i - (1u << 16))));
  }
  const unsigned prefix = (i - (2u << 16)) & 0xffffu;
  const unsigned suffix = i < (3u << 16) ? 0u : 0xffffu;
  return __uint_as_float((prefix << 16) | suffix);
}

// For every input x and every op: util/math.h's scalar result, and the packed form with x in
// the low lane and in the high lane of the pair (the other lane holds a different value).
// out is laid out as [op][ref, low lane, high lane][input].
__global__ void packed_activation_kernel(float *out, int *ran) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  namespace te = transformer_engine;
  namespace pa = te::packed_activation;
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) *ran = 1;
  if (i >= kNumInputs) return;
  const float x = input_value(i);
  const float other = input_value((i * 40503u) % kNumInputs);
  float *o = out + i;
  const te::Empty e{};
  o[0 * kNumInputs] = te::gelu<float, float>(x, e);
  o[1 * kNumInputs] = pa::gelu_2x({x, other}).x;
  o[2 * kNumInputs] = pa::gelu_2x({other, x}).y;
  o[3 * kNumInputs] = te::dgelu<float, float>(x, e);
  o[4 * kNumInputs] = pa::dgelu_2x({x, other}).x;
  o[5 * kNumInputs] = pa::dgelu_2x({other, x}).y;
  o[6 * kNumInputs] = te::silu<float, float>(x, e);
  o[7 * kNumInputs] = pa::silu_2x({x, other}).x;
  o[8 * kNumInputs] = pa::silu_2x({other, x}).y;
  o[9 * kNumInputs] = te::dsilu<float, float>(x, e);
  o[10 * kNumInputs] = pa::dsilu_2x({x, other}).x;
  o[11 * kNumInputs] = pa::dsilu_2x({other, x}).y;
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

// The quantize kernels use the packed forms in place of util/math.h's activations, so they must
// round identically for every BF16 and FP16 input and sampled FP32 mantissas.
TEST(UtilTest, PackedActivationMatchesScalar) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 10) {
    GTEST_SKIP() << "Packed FP32x2 arithmetic requires compute capability 10.0 or newer";
  }

  constexpr int kNumOutputs = 3 * kNumOps;
  float *out = nullptr;
  int *ran = nullptr;
  ASSERT_EQ(cudaMalloc(&out, kNumOutputs * kNumInputs * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&ran, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(ran, 0, sizeof(int)), cudaSuccess);
  packed_activation_kernel<<<kNumInputs / 256, 256>>>(out, ran);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int ran_host = 0;
  std::vector<float> host(kNumOutputs * kNumInputs);
  ASSERT_EQ(cudaMemcpy(&ran_host, ran, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(host.data(), out, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(out);
  cudaFree(ran);
  if (ran_host == 0) {
    GTEST_SKIP() << "Test kernel was not compiled for compute capability 10.0 or newer";
  }

  for (int op = 0; op < kNumOps; ++op) {
    const float *ref = host.data() + (3 * op) * kNumInputs;
    for (int lane = 0; lane < 2; ++lane) {
      const float *packed = ref + (1 + lane) * kNumInputs;
      int mismatches = 0;
      for (int i = 0; i < kNumInputs; ++i) {
        if (!same_value(ref[i], packed[i])) {
          if (mismatches == 0) {
            ADD_FAILURE() << kOpNames[op] << (lane == 0 ? " (low lane)" : " (high lane)")
                          << " differs for "
                          << (i < (1 << 16) ? "BF16" : i < (2 << 16) ? "FP16" : "FP32")
                          << " input 0x" << std::hex << (i & 0xffff) << std::dec << ": scalar "
                          << ref[i]
                          << ", packed " << packed[i];
          }
          ++mismatches;
        }
      }
      EXPECT_EQ(mismatches, 0) << kOpNames[op] << (lane == 0 ? " low lane" : " high lane");
    }
  }
}
