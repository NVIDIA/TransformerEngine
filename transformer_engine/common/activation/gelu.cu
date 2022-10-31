/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include "../utils.cuh"
#include "../common.h"
#include <cstdlib>
#include <../util/vectorized_pointwise.h>

namespace transformer_engine {

namespace detail {

struct GELUParam {};

__device__ inline fp32 gelu(fp32 value, const GELUParam &) {
  return value * (0.5F + 0.5F * tanhf(value * (0.79788456F + 0.03567741F * value * value)));
}

}

void gelu_cast(const Tensor &input,
               const Tensor &scale,
               Tensor *output,
               Tensor *amax,
               Tensor *scale_inv,
               cudaStream_t stream) {
  NVTE_CHECK(input.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.shape == output->shape, "Input and output shapes must match.");
  const size_t tot_elts = input.shape[1] * input.shape[0];

  NVTE_CHECK(amax->shape == std::vector<size_t>{ 1 }, "AMAX tensor must have 1 element.");
  NVTE_CHECK(amax->dtype == DType::kFloat32, "AMAX tensor must have Float32 type.");
  NVTE_CHECK(scale.shape == std::vector<size_t>{ 1 }, "Scale tensor must have 1 element.");
  NVTE_CHECK(scale.dtype == DType::kFloat32, "Scale tensor must have Float32 type.");
  NVTE_CHECK(scale_inv->shape == std::vector<size_t>{ 1 },
      "scale_inv tensor must have 1 element.");
  NVTE_CHECK(scale_inv->dtype == DType::kFloat32, "scale_inv tensor must have Float32 type.");

  NVTE_CHECK(input.dptr != nullptr, "Input is not allocated.");
  NVTE_CHECK(scale.dptr != nullptr, "Scale is not allocated.");
  NVTE_CHECK(output->dptr != nullptr, "Output is not allocated.");
  NVTE_CHECK(amax->dptr != nullptr, "AMAX tensor is not allocated.");
  NVTE_CHECK(scale_inv->dptr != nullptr, "scale_inv tensor is not allocated.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      VectorizedUnaryKernelLauncher<nvec, detail::GELUParam, detail::gelu>(
        reinterpret_cast<const IType*>(input.dptr),
        reinterpret_cast<OType*>(output->dptr),
        reinterpret_cast<const fp32*>(scale.dptr),
        reinterpret_cast<fp32*>(scale_inv->dptr),
        reinterpret_cast<fp32*>(amax->dptr),
        tot_elts,
        {},
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

template <int nvec, bool aligned,
          typename ComputeType,
          typename Param,
          ComputeType (*OP)(ComputeType, const Param&),
          typename InputType,
          typename OutputType>
__launch_bounds__(unary_kernel_threads)
__global__ void gated_gelu_kernel(const InputType *input,
                                  OutputType *output,
                                  const ComputeType *scale,
                                  ComputeType *scale_inv,
                                  ComputeType *amax,
                                  Param p,
                                  const size_t m,
                                  const size_t n,
                                  const size_t num_aligned_elements) {
  const size_t M = num_aligned_elements * m;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      tid < M;
      tid += gridDim.x * blockDim.x) {
    const size_t id_x = tid % num_aligned_elements;
    const size_t id_y = tid / num_aligned_elements;
    VectorizedLoader<InputType, nvec, aligned> loader(input + id_y * n * 2, n);
    VectorizedLoader<InputType, nvec, aligned> loader2(input + id_y * n * 2 + n, n);
    VectorizedStorer<OutputType, nvec, aligned> storer(output + id_y * n, n);
    ComputeType max = 0;
    ComputeType s = 0;
    if constexpr (is_fp8<OutputType>::value) {
        if (scale != nullptr) s = *scale;
        if (blockIdx.x == 0 && threadIdx.x == 0 && scale_inv != nullptr) {
          reciprocal<ComputeType>(scale_inv, s);
        }
    }
    const int warp_id = threadIdx.x / THREADS_PER_WARP;

    loader.load(id_x, n);
    loader2.load(id_x, n);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const ComputeType val = static_cast<ComputeType>(loader.separate()[i]);
      const InputType val2 = loader2.separate()[i];
      InputType temp = static_cast<InputType>(OP(val, p)) * val2;
      ComputeType tempC = static_cast<ComputeType>(temp);
      if constexpr (is_fp8<OutputType>::value) {
        __builtin_assume(max >= 0);
        max = fmaxf(fabsf(tempC), max);

        temp = tempC * s;
      }

      storer.separate()[i] = static_cast<OutputType>(static_cast<ComputeType>(temp));
    }
    storer.store(id_x, n);

    if constexpr (is_fp8<OutputType>::value) {
      /* warp tile amax reduce*/
      max = reduce_max<unary_kernel_threads / THREADS_PER_WARP>(max, warp_id);

      if (threadIdx.x == 0 && amax != nullptr) {
          static_assert(std::is_same<ComputeType, float>::value);
          atomicMaxFloat(amax, max);
      }
    }
  }
}

template <int nvec, typename Param,
          fp32 (*OP)(fp32, const Param&),
          typename InputType,
          typename OutputType>
void GatedGeluKernelLauncher(const InputType *input,
                             OutputType *output,
                             const fp32 *scale,
                             fp32 *scale_inv,
                             fp32 *amax,
                             const size_t m,
                             const size_t n,
                             const Param params,
                             cudaStream_t stream) {
  if (m != 0 && n != 0) {
    auto align = CheckAlignment(n, nvec, input, output);

    size_t num_aligned_elements = get_num_aligned_elements(input, n, nvec,
                                                           sizeof(InputType));
    constexpr size_t threads = unary_kernel_threads;
    size_t num_blocks = DIVUP(num_aligned_elements * m, threads);
    constexpr size_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    switch (align) {
      case Alignment::SAME_ALIGNED:
        gated_gelu_kernel<nvec, true, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, m, n, num_aligned_elements);
        break;
      case Alignment::SAME_UNALIGNED:
        gated_gelu_kernel<nvec, false, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, m, n, num_aligned_elements);
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        gated_gelu_kernel<1, true, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, m, n, n);
        break;
      }
    }
  }
}

void gated_gelu_cast(const Tensor &input,
                     const Tensor &scale,
                     Tensor *output,
                     Tensor *amax,
                     Tensor *scale_inv,
                     cudaStream_t stream) {
  NVTE_CHECK(input.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.shape[0] == output->shape[0] && input.shape[1] == output->shape[1] * 2,
      "Input and output shapes must match.");

  NVTE_CHECK(amax->shape == std::vector<size_t>{ 1 }, "AMAX tensor must have 1 element.");
  NVTE_CHECK(amax->dtype == DType::kFloat32, "AMAX tensor must have Float32 type.");
  NVTE_CHECK(scale.shape == std::vector<size_t>{ 1 }, "Scale tensor must have 1 element.");
  NVTE_CHECK(scale.dtype == DType::kFloat32, "Scale tensor must have Float32 type.");
  NVTE_CHECK(scale_inv->shape == std::vector<size_t>{ 1 },
      "scale_inv tensor must have 1 element.");
  NVTE_CHECK(scale_inv->dtype == DType::kFloat32, "scale_inv tensor must have Float32 type.");

  NVTE_CHECK(input.dptr != nullptr, "Input is not allocated.");
  NVTE_CHECK(scale.dptr != nullptr, "Scale is not allocated.");
  NVTE_CHECK(output->dptr != nullptr, "Output is not allocated.");
  NVTE_CHECK(amax->dptr != nullptr, "AMAX tensor is not allocated.");
  NVTE_CHECK(scale_inv->dptr != nullptr, "scale_inv tensor is not allocated.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      GatedGeluKernelLauncher<nvec, detail::GELUParam, detail::gelu>(
        reinterpret_cast<const IType*>(input.dptr),
        reinterpret_cast<OType*>(output->dptr),
        reinterpret_cast<const fp32*>(scale.dptr),
        reinterpret_cast<fp32*>(scale_inv->dptr),
        reinterpret_cast<fp32*>(amax->dptr),
        output->shape[0],
        output->shape[1],
        {},
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}
}  // namespace transformer_engine

void nvte_gelu(const NVTETensor input,
               NVTETensor output,
               const NVTETensor scale,
               NVTETensor amax,
               NVTETensor scale_inv,
               cudaStream_t stream) {
  using namespace transformer_engine;
  gelu_cast(*reinterpret_cast<const Tensor*>(input),
            *reinterpret_cast<const Tensor*>(scale),
            reinterpret_cast<Tensor*>(output),
            reinterpret_cast<Tensor*>(amax),
            reinterpret_cast<Tensor*>(scale_inv),
            stream);
}

void nvte_gated_gelu(const NVTETensor input,
                     NVTETensor output,
                     const NVTETensor scale,
                     NVTETensor amax,
                     NVTETensor scale_inv,
                     cudaStream_t stream) {
  using namespace transformer_engine;
  gated_gelu_cast(*reinterpret_cast<const Tensor*>(input),
                  *reinterpret_cast<const Tensor*>(scale),
                  reinterpret_cast<Tensor*>(output),
                  reinterpret_cast<Tensor*>(amax),
                  reinterpret_cast<Tensor*>(scale_inv),
                  stream);
}
