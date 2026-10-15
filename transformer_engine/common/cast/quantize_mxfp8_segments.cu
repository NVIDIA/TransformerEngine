/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <transformer_engine/cast.h>

#include <cstdint>
#include <limits>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/ptx.cuh"

namespace transformer_engine {
namespace segmented_mxfp8 {

struct Segments {
  const __nv_bfloat16 *q, *k, *v;
  int dq, dk, dv, qs, ks, vs;
};

__device__ __forceinline__ const __nv_bfloat16 *base(Segments s, int col, int &offset,
                                                     int &stride) {
  if (col < s.dq) {
    offset = col;
    stride = s.qs;
    return s.q;
  }
  if (col < s.dq + s.dk) {
    offset = col - s.dq;
    stride = s.ks;
    return s.k;
  }
  offset = col - s.dq - s.dk;
  stride = s.vs;
  return s.v;
}

__device__ __forceinline__ unsigned char scale_exponent(float amax) {
  unsigned int packed;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float normalized = amax * (1.0f / 448.0f);
  asm volatile(
      "{ .reg .b16 pair,zero; mov.b16 zero,0; cvt.rp.satfinite.ue8m0x2.f32 pair,%1,%1; mov.b32 "
      "%0,{pair,zero}; }"
      : "=r"(packed)
      : "f"(normalized));
#else
  packed = 0;
#endif
  return packed & 255;
}

__device__ __forceinline__ float inverse(unsigned char e) {
  unsigned int bits = e == 254 ? 0x00400000u : (254u - e) << 23;
  if (e == 255) bits = 0x7fffffffu;
  return __uint_as_float(bits);
}

__device__ __forceinline__ int64_t sf(int i, int j, int tiles) {
  return (static_cast<int64_t>(i / 128) * tiles + j / 4) * 512 + (i % 32) * 16 +
         ((i % 128) / 32) * 4 + j % 4;
}

__device__ __forceinline__ uint16_t pair_cast(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint16_t result;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;" : "=h"(result) : "f"(a), "f"(b));
  return result;
#else
  NVTE_DEVICE_ERROR("FP8 pair conversion is only supported on SM 8.9+.");
  return 0;
#endif
}

__global__ void row_kernel(Segments s, unsigned char *out, unsigned char *scales, int n, int cols,
                           bool swizzle) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= n * (cols / 32)) return;
  int row = group / (cols / 32), col = (group % (cols / 32)) * 32, offset, stride;
  const __nv_bfloat16 *p = base(s, col, offset, stride);
  p += static_cast<int64_t>(row) * stride + offset;
  uint4 input[4];
#pragma unroll
  for (int i = 0; i < 4; i++) input[i] = reinterpret_cast<const uint4 *>(p)[i];
  const __nv_bfloat16 *values = reinterpret_cast<const __nv_bfloat16 *>(input);
  float x[32], amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    x[i] = __bfloat162float(values[i]);
    amax = fmaxf(amax, fabsf(x[i]));
  }
  unsigned char e = scale_exponent(amax);
  alignas(16) unsigned int packed[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    ptx::floatx4 input_group;
#pragma unroll
    for (int j = 0; j < 4; ++j) reinterpret_cast<float *>(&input_group)[j] = x[i * 4 + j];
    ptx::floatx2 scale;
    reinterpret_cast<float *>(&scale)[0] = inverse(e);
    reinterpret_cast<float *>(&scale)[1] = inverse(e);
    ptx::fp8e4m3x4 converted;
    ptx::mul_cvt_4x(converted, input_group, scale);
    const unsigned int word = reinterpret_cast<const unsigned int &>(converted);
    packed[i] = word;
  }
  auto dst = reinterpret_cast<unsigned int *>(out + static_cast<int64_t>(row) * cols + col);
#pragma unroll
  for (int i = 0; i < 2; i++)
    reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<uint4 *>(packed)[i];
  scales[swizzle ? sf(row, col / 32, cols / 128)
                 : static_cast<int64_t>(row) * (cols / 32) + col / 32] = e;
}

__global__ void column_kernel(Segments s, unsigned char *out, unsigned char *scales, int n,
                              int cols, bool swizzle, bool canonical_zero) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= (n / 32) * cols) return;
  int row = (group / cols) * 32, col = group % cols, offset, stride;
  const __nv_bfloat16 *p = base(s, col, offset, stride);
  p += static_cast<int64_t>(row) * stride + offset;
  float x[32], amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    x[i] = __bfloat162float(p[static_cast<int64_t>(i) * stride]);
    amax = i == 0 ? fabsf(x[i]) : fmaxf(amax, fabsf(x[i]));
  }
  unsigned char e = scale_exponent(amax);
  float inv = inverse(e);
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint16_t packed = pair_cast(__fmaf_rn(x[i * 2], inv, 0.0f), __fmaf_rn(x[i * 2 + 1], inv, 0.0f));
    unsigned char a = packed & 255, b = (packed >> 8) & 255;
    if (canonical_zero && x[i * 2] == 0.0f) a = 0;
    if (canonical_zero && x[i * 2 + 1] == 0.0f) b = 0;
    out[static_cast<int64_t>(row + i * 2) * cols + col] = a;
    out[static_cast<int64_t>(row + i * 2 + 1) * cols + col] = b;
  }
  scales[swizzle ? sf(col, row / 32, n / 128) : static_cast<int64_t>(row / 32) * cols + col] = e;
}

}  // namespace segmented_mxfp8
}  // namespace transformer_engine

void nvte_quantize_mxfp8_segments(const NVTETensor *inputs, size_t num_inputs,
                                  const int64_t *row_strides, NVTETensor output,
                                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_mxfp8_segments);
  using namespace transformer_engine;
  using namespace segmented_mxfp8;
  NVTE_CHECK(cuda::sm_arch() == 103, "Segmented MXFP8 is qualified for SM103.");
  NVTE_CHECK(num_inputs == 1 || num_inputs == 3, "Expected one or three column segments.");
  NVTE_CHECK(inputs != nullptr && row_strides != nullptr, "Missing input segments or strides.");
  auto *out = convertNVTETensorCheck(output);
  NVTE_CHECK(out->scaling_mode == NVTE_MXFP8_1D_SCALING && out->dtype() == DType::kFloat8E4M3,
             "Expected 1D E4M3 MXFP8 output.");
  Segments segments{};
  const __nv_bfloat16 **pointers[] = {&segments.q, &segments.k, &segments.v};
  int *widths[] = {&segments.dq, &segments.dk, &segments.dv};
  int *strides[] = {&segments.qs, &segments.ks, &segments.vs};
  int64_t rows = -1, columns = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto *input = convertNVTETensorCheck(inputs[i]);
    NVTE_CHECK(input->data.dtype == DType::kBFloat16 && input->data.shape.size() == 2,
               "Segments must be BF16 matrices.");
    const int64_t n = input->data.shape[0], width = input->data.shape[1];
    NVTE_CHECK(n > 0 && n % 128 == 0 && width > 0 && width % 128 == 0,
               "Rows and each segment width must be positive multiples of 128.");
    NVTE_CHECK(rows == -1 || rows == n, "Segment row counts must match.");
    NVTE_CHECK(row_strides[i] >= width && row_strides[i] % 8 == 0 &&
                   row_strides[i] <= std::numeric_limits<int>::max(),
               "Invalid row stride.");
    NVTE_CHECK(reinterpret_cast<uintptr_t>(input->data.dptr) % 16 == 0,
               "Segment addresses must be 16-byte aligned.");
    rows = n;
    columns += width;
    NVTE_CHECK(width <= std::numeric_limits<int>::max(), "Segment is too wide.");
    *pointers[i] = static_cast<const __nv_bfloat16 *>(input->data.dptr);
    *widths[i] = static_cast<int>(width);
    *strides[i] = static_cast<int>(row_strides[i]);
  }
  if (num_inputs == 1) {
    segments.k = segments.v = segments.q;
    segments.ks = segments.vs = segments.qs;
  }
  NVTE_CHECK(rows <= std::numeric_limits<int>::max() &&
                 columns <= std::numeric_limits<int>::max() &&
                 rows * columns / 32 <= std::numeric_limits<int>::max() - 128,
             "Matrix is too large.");
  NVTE_CHECK(out->numel() == static_cast<size_t>(rows * columns), "Output shape mismatch.");
  const bool rw = out->data.dptr != nullptr, cw = out->columnwise_data.dptr != nullptr;
  NVTE_CHECK(rw && cw && out->with_gemm_swizzled_scales,
             "Segmented MXFP8 requires both orientations and GEMM-swizzled scales.");
  const size_t scales = rows * columns / 32;
  if (rw) {
    NVTE_CHECK(out->scale_inv.dptr != nullptr && out->scale_inv.numel() >= scales,
               "Missing rowwise scale storage.");
  }
  if (cw) {
    NVTE_CHECK(
        out->columnwise_scale_inv.dptr != nullptr && out->columnwise_scale_inv.numel() >= scales,
        "Missing columnwise scale storage.");
  }
  constexpr int threads = 128;
  const int groups = rows * columns / 32;
  if (rw) {
    row_kernel<<<(groups + threads - 1) / threads, threads, 0, stream>>>(
        segments, static_cast<unsigned char *>(out->data.dptr),
        static_cast<unsigned char *>(out->scale_inv.dptr), rows, columns,
        out->with_gemm_swizzled_scales);
  }
  if (cw) {
    column_kernel<<<(groups + threads - 1) / threads, threads, 0, stream>>>(
        segments, static_cast<unsigned char *>(out->columnwise_data.dptr),
        static_cast<unsigned char *>(out->columnwise_scale_inv.dptr), rows, columns,
        out->with_gemm_swizzled_scales, rw);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}
