/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/cast.h>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace {
#if FP4_TYPE_SUPPORTED
// Shared block processing for both outer-scale modes. Both rounded encodings
// stay in registers; only the decoded high-precision values are written.
template <typename T, bool ROW_SCALED_NVFP4>
__device__ __forceinline__ void qdq_block(const T *input, T *output, float outer_amax,
                                          float encode) {
  Vec<T, 16> values;
  values.load_from(input);
  float local_amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    local_amax = fmaxf(local_amax, fabsf(static_cast<float>(values.data.elt[i])));
  }
  using namespace dispatch::nvfp4;
  if constexpr (!ROW_SCALED_NVFP4) {
    // Keep input loads ahead of the encode-scale division, as in the original
    // tensor-scaled kernel. Row scaling reuses its CTA's precomputed encode.
    encode = core::compute_global_encode_scaling_factor_FP4<fp8e4m3>(outer_amax);
  }
  const fp8e4m3 scale = core::compute_decoding_scaling_factor<fp8e4m3>(local_amax, encode);
  const float inverse =
      fminf(1.0f / (static_cast<float>(scale) * (1.0f / encode)), detail::TypeExtrema<float>::max);
  // Match native dequantize multiplication order, including zero signs.
  const float decode = static_cast<float>(scale) * outer_amax * (1.0f / (6.0f * 448.0f));
#pragma unroll
  for (int i = 0; i < 16; i += 4) {
    fp4e2m1x4 packed;
    ptx::mul_cvt_4x(packed,
                    make_float2(static_cast<float>(values.data.elt[i]),
                                static_cast<float>(values.data.elt[i + 1])),
                    make_float2(static_cast<float>(values.data.elt[i + 2]),
                                static_cast<float>(values.data.elt[i + 3])),
                    inverse);
    const float4 decoded = static_cast<float4>(packed);
    values.data.elt[i] = static_cast<T>(decoded.x * decode);
    values.data.elt[i + 1] = static_cast<T>(decoded.y * decode);
    values.data.elt[i + 2] = static_cast<T>(decoded.z * decode);
    values.data.elt[i + 3] = static_cast<T>(decoded.w * decode);
  }
  values.store_to(output);
}

// Tensor scaling distributes blocks across CTAs and reads a precomputed amax.
// Row scaling gives each CTA a complete row and fuses its outer amax reduction.
template <typename T, int THREADS, bool ROW_SCALED_NVFP4>
__global__ void qdq_kernel(const T *input, T *output, const float *amax, const float *noop,
                           size_t blocks) {
  if (noop != nullptr && *noop == 1.0f) return;
  using namespace dispatch::nvfp4;
  if constexpr (ROW_SCALED_NVFP4) {
    const size_t row_offset = static_cast<size_t>(blockIdx.x) * blocks * 16;
    float thread_amax = 0.0f;
    for (size_t block = threadIdx.x; block < blocks; block += THREADS) {
      Vec<T, 16> values;
      values.load_from(input + row_offset + block * 16);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        thread_amax = fmaxf(thread_amax, fabsf(static_cast<float>(values.data.elt[i])));
      }
    }
    const float maximum = reduce_max<THREADS / 32>(thread_amax, threadIdx.x / 32);
    __shared__ float row_amax;
    if (threadIdx.x == 0) row_amax = maximum;
    __syncthreads();
    const float encode = core::compute_global_encode_scaling_factor_FP4<fp8e4m3>(row_amax);
    for (size_t block = threadIdx.x; block < blocks; block += THREADS) {
      qdq_block<T, ROW_SCALED_NVFP4>(input + row_offset + block * 16,
                                     output + row_offset + block * 16, row_amax, encode);
    }
  } else {
    const size_t block = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block >= blocks) return;
    const float outer_amax = *amax;
    qdq_block<T, ROW_SCALED_NVFP4>(input + block * 16, output + block * 16, outer_amax, 0.0f);
  }
}
#endif

template <bool ROW_SCALED_NVFP4>
void launch_qdq(const NVTETensor input, NVTETensor output, const NVTETensor amax,
                const NVTETensor noop, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_CHECK(cuda::sm_arch() == 100, "NVFP4 QDQ currently supports SM100");
  const auto &in = *convertNVTETensorCheck(input);
  auto &out = *convertNVTETensorCheck(output);
  CheckInputTensor(in, "input");
  CheckOutputTensor(out, "output");
  NVTE_CHECK(in.data.dtype == out.data.dtype && in.data.shape == out.data.shape,
             "QDQ input and output must have matching dtype and shape");
  NVTE_CHECK(in.data.dtype == DType::kBFloat16 || in.data.dtype == DType::kFloat16,
             "QDQ supports BF16 and FP16");
  NVTE_CHECK(in.data.shape.size() >= 2 && in.numel() > 0 && in.data.shape.back() % 16 == 0 &&
                 (in.numel() / in.data.shape.back()) % 16 == 0,
             "QDQ requires nonempty block-aligned rowwise input");
  const float *amax_ptr = nullptr;
  if constexpr (!ROW_SCALED_NVFP4) {
    const auto &amax_tensor = *convertNVTETensorCheck(amax);
    CheckInputTensor(amax_tensor, "amax");
    NVTE_CHECK(amax_tensor.data.dtype == DType::kFloat32 && amax_tensor.numel() == 1,
               "QDQ requires a scalar FP32 amax");
    amax_ptr = static_cast<const float *>(amax_tensor.data.dptr);
  }
  NVTE_CHECK(reinterpret_cast<uintptr_t>(in.data.dptr) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(out.data.dptr) % 16 == 0,
             "QDQ requires 16-byte aligned buffers");
  const float *noop_ptr = nullptr;
  if (noop != nullptr) {
    const auto &flag = *convertNVTETensorCheck(noop);
    CheckInputTensor(flag, "noop");
    NVTE_CHECK(flag.data.dtype == DType::kFloat32 && flag.numel() == 1,
               "QDQ noop must be a scalar FP32 tensor");
    noop_ptr = static_cast<const float *>(flag.data.dptr);
  }
  constexpr int threads = 128;
  const size_t blocks = ROW_SCALED_NVFP4 ? in.data.shape.back() / 16 : in.numel() / 16;
  const size_t ctas =
      ROW_SCALED_NVFP4 ? in.numel() / in.data.shape.back() : (blocks + threads - 1) / threads;
  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      in.data.dtype, T,
      qdq_kernel<T, threads, ROW_SCALED_NVFP4>
      <<<ctas, threads, 0, stream>>>(static_cast<const T *>(in.data.dptr),
                                     static_cast<T *>(out.data.dptr), amax_ptr, noop_ptr, blocks););
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("NVFP4 QDQ requires a CUDA toolkit with FP4 support");
#endif
}
}  // namespace
}  // namespace transformer_engine

void nvte_nvfp4_qdq(const NVTETensor input, NVTETensor output, const NVTETensor amax,
                    const NVTETensor noop, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_qdq);
  transformer_engine::launch_qdq<false>(input, output, amax, noop, stream);
}

void nvte_nvfp4_qdq_row_scaled(const NVTETensor input, NVTETensor output, const NVTETensor noop,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_qdq_row_scaled);
  transformer_engine::launch_qdq<true>(input, output, nullptr, noop, stream);
}
