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

#include "util/packed_activation.cuh"

using namespace transformer_engine;

namespace {

constexpr int kNumBf16Values = 1 << 16;

// For every BF16 bit pattern x: util/math.h's scalar GeLU/dGeLU, and the packed
// forms with x in the low lane and in the high lane of the pair (the other lane
// holds a different value).
__global__ void packed_activation_kernel(float *ref_gelu, float *ref_dgelu, float *packed_gelu_lo,
                                         float *packed_gelu_hi, float *packed_dgelu_lo,
                                         float *packed_dgelu_hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kNumBf16Values) return;
  const float x = __uint_as_float(i << 16);
  const float other = __uint_as_float(((i * 40503u) & 0xffffu) << 16);
  ref_gelu[i] = gelu<float, float>(x, Empty{});
  ref_dgelu[i] = dgelu<float, float>(x, Empty{});
  float lo, hi;
  packed_act::unpack_f32x2(packed_act::gelu_f32x2(packed_act::make_f32x2(x, other)), lo, hi);
  packed_gelu_lo[i] = lo;
  packed_act::unpack_f32x2(packed_act::gelu_f32x2(packed_act::make_f32x2(other, x)), lo, hi);
  packed_gelu_hi[i] = hi;
  packed_act::unpack_f32x2(packed_act::dgelu_f32x2(packed_act::make_f32x2(x, other)), lo, hi);
  packed_dgelu_lo[i] = lo;
  packed_act::unpack_f32x2(packed_act::dgelu_f32x2(packed_act::make_f32x2(other, x)), lo, hi);
  packed_dgelu_hi[i] = hi;
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

// The quantize kernels use the packed forms in place of util/math.h's gelu and
// dgelu, so they must round identically for every BF16 input.
TEST(UtilTest, PackedActivationMatchesScalar) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 10) {
    GTEST_SKIP() << "Packed FP32x2 arithmetic requires compute capability 10.0 or newer";
  }

  constexpr int kNumOutputs = 6;
  float *device_buffer = nullptr;
  ASSERT_EQ(cudaMalloc(&device_buffer, kNumOutputs * kNumBf16Values * sizeof(float)), cudaSuccess);
  float *out[kNumOutputs];
  for (int k = 0; k < kNumOutputs; ++k) out[k] = device_buffer + k * kNumBf16Values;
  packed_activation_kernel<<<kNumBf16Values / 256, 256>>>(out[0], out[1], out[2], out[3], out[4],
                                                          out[5]);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> host(kNumOutputs * kNumBf16Values);
  ASSERT_EQ(
      cudaMemcpy(host.data(), device_buffer, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
      cudaSuccess);
  cudaFree(device_buffer);

  const float *ref_gelu = host.data();
  const float *ref_dgelu = ref_gelu + kNumBf16Values;
  const float *packed[4] = {ref_gelu + 2 * kNumBf16Values, ref_gelu + 3 * kNumBf16Values,
                            ref_gelu + 4 * kNumBf16Values, ref_gelu + 5 * kNumBf16Values};
  const char *names[4] = {"gelu (low lane)", "gelu (high lane)", "dgelu (low lane)",
                          "dgelu (high lane)"};
  for (int k = 0; k < 4; ++k) {
    const float *ref = k < 2 ? ref_gelu : ref_dgelu;
    int mismatches = 0;
    for (int i = 0; i < kNumBf16Values; ++i) {
      if (!same_value(ref[i], packed[k][i])) {
        if (mismatches == 0) {
          ADD_FAILURE() << names[k] << " differs for BF16 input 0x" << std::hex << i << std::dec
                        << ": scalar " << ref[i] << ", packed " << packed[k][i];
        }
        ++mismatches;
      }
    }
    EXPECT_EQ(mismatches, 0) << names[k];
  }
}
