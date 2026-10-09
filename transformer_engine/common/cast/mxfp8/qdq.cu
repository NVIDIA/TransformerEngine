/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/cast.h>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx.cuh"
#include "../../util/ptx_arch_spec.cuh"

namespace transformer_engine {
namespace {

// Four adjacent threads own one block of 32. Each thread reads/writes 16B.
template <typename T>
__global__ void mxfp8_qdq_kernel(const T *input, T *output, const float *noop, size_t blocks) {
  if (noop != nullptr && *noop == 1.0f) return;
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t block = thread / 4;
  if (block >= blocks) return;
  const size_t offset = block * 32 + (thread % 4) * 8;
  Vec<T, 8> values;
  values.load_from(input + offset);
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    amax = fmaxf(amax, fabsf(static_cast<float>(values.data.elt[i])));
  }
  const unsigned mask = __activemask();
  amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1, 4));
  amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 2, 4));
  const e8m0_t exponent = ptx::float_to_e8m0(amax * (1.0f / 448.0f));
  const float encode = ptx::exp2f_rcp<float>(exponent);
  const float decode = ptx::exp2f(exponent);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const fp8e4m3 encoded = static_cast<fp8e4m3>(static_cast<float>(values.data.elt[i]) * encode);
    values.data.elt[i] = static_cast<T>(static_cast<float>(encoded) * decode);
  }
  values.store_to(output + offset);
}
}  // namespace
}  // namespace transformer_engine

void nvte_mxfp8_qdq(const NVTETensor input, NVTETensor output, const NVTETensor noop,
                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_mxfp8_qdq);
  using namespace transformer_engine;
  NVTE_CHECK(cuda::sm_arch() == 100, "MXFP8 QDQ currently supports SM100");
  const auto &in = *convertNVTETensorCheck(input);
  auto &out = *convertNVTETensorCheck(output);
  CheckInputTensor(in, "input");
  CheckOutputTensor(out, "output");
  NVTE_CHECK(in.data.dtype == out.data.dtype && in.data.shape == out.data.shape,
             "QDQ input and output must have matching dtype and shape");
  NVTE_CHECK(in.data.dtype == DType::kBFloat16 || in.data.dtype == DType::kFloat16,
             "QDQ supports BF16 and FP16");
  NVTE_CHECK(in.data.shape.size() >= 2 && in.numel() > 0 && in.data.shape.back() % 32 == 0,
             "QDQ requires nonempty block-32 aligned rows");
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
  const size_t blocks = in.numel() / 32;
  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      in.data.dtype, T,
      mxfp8_qdq_kernel<T><<<(blocks * 4 + threads - 1) / threads, threads, 0, stream>>>(
          static_cast<const T *>(in.data.dptr), static_cast<T *>(out.data.dptr), noop_ptr,
          blocks););
  NVTE_CHECK_CUDA(cudaGetLastError());
}
