/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_per_token_nvfp4.cuh
 *  \brief CUDA kernels to cast to NVFP4 with per-token (per-row) global scaling.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <type_traits>

#include "../../common.h"
#include "../../transpose/cast_transpose.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"
#include "quantize_transpose_nvfp4.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace quantize_per_token_kernel {

using namespace core;
using namespace ptx;

constexpr int PERTOKEN_BLOCK_SIZE = 256;
constexpr int PERTOKEN_SF_VEC_SIZE = 16;

template <typename IType>
__device__ __forceinline__ void abs_max_2x_update(ptx::FPx2<IType> &dst,
                                                  const ptx::FPx2<IType> &val) {
  if constexpr (std::is_same_v<IType, float>) {
    dst.x = fmaxf(fabsf(dst.x), fabsf(val.x));
    dst.y = fmaxf(fabsf(dst.y), fabsf(val.y));
  } else {
    ptx::abs_max_2x(dst, dst, val);
  }
}

template <typename IType>
__device__ __forceinline__ float abs_max_2x_to_float(const ptx::FPx2<IType> &val) {
  if constexpr (std::is_same_v<IType, float>) {
    return fmaxf(fabsf(val.x), fabsf(val.y));
  } else {
    return static_cast<float>(__hmax(__habs(val.x), __habs(val.y)));
  }
}

template <typename IType, int BLOCK_SIZE>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(BLOCK_SIZE)
#endif
    compute_per_token_amax_kernel(const int num_rows, const int num_cols,
                                  const IType *__restrict__ input,
                                  float *__restrict__ output_per_token_amax,
                                  const float *__restrict__ noop) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  using IType2 = typename ptx::FPx2<IType>;

  const int row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  const int num_vec2 = num_cols / 2;
  const IType2 *input_row = reinterpret_cast<const IType2 *>(input + row_idx * num_cols);

  IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
  for (int i = threadIdx.x; i < num_vec2; i += BLOCK_SIZE) {
    const IType2 val = input_row[i];
    abs_max_2x_update(thread_amax_2x, val);
  }
  const float thread_max = abs_max_2x_to_float(thread_amax_2x);

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float row_amax =
      BlockReduce(temp_storage).Reduce(thread_max, [](float a, float b) { return fmaxf(a, b); });

  if (threadIdx.x == 0) {
    output_per_token_amax[row_idx] = row_amax;
  }
#endif
}

template <typename IType>
void launch_compute_per_token_amax(const int num_rows, const int num_cols, const IType *input,
                                   float *output_per_token_amax, cudaStream_t stream,
                                   const float *noop = nullptr) {
#if FP4_TYPE_SUPPORTED
  if (num_rows == 0 || num_cols == 0) return;

  NVTE_CHECK(num_cols % 2 == 0, "num_cols must be even for per-token amax computation, got ",
             num_cols);
  dim3 grid(num_rows);
  dim3 block(PERTOKEN_BLOCK_SIZE);

  compute_per_token_amax_kernel<IType, PERTOKEN_BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(num_rows, num_cols, input, output_per_token_amax, noop);
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif
}

}  // namespace quantize_per_token_kernel

inline void quantize_per_token(const Tensor &input, const Tensor *noop, Tensor *output,
                               const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace detail;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->has_data(), "Per-token NVFP4 quantization requires rowwise output.");
  NVTE_CHECK(!output->has_columnwise_data(),
             "Per-token NVFP4 quantization does not produce columnwise output.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales, "Output must have scales in compact format.");

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  NVTE_CHECK(cols % quantize_per_token_kernel::PERTOKEN_SF_VEC_SIZE == 0,
             "Per-token NVFP4 quantization requires last dim divisible by ",
             quantize_per_token_kernel::PERTOKEN_SF_VEC_SIZE, ".");

  const auto *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  auto *amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  NVTE_CHECK(amax_ptr != nullptr, "Per-token rowwise amax tensor must be allocated.");
  NVTE_CHECK(output->amax.numel() == rows, "Per-token rowwise amax must have ", rows,
             " entries, got ", output->amax.shape, ".");

  if (input.dtype() == DType::kBFloat16) {
    const auto *input_ptr = reinterpret_cast<const __nv_bfloat16 *>(input.data.dptr);
    quantize_per_token_kernel::launch_compute_per_token_amax<__nv_bfloat16>(
        static_cast<int>(rows), static_cast<int>(cols), input_ptr, amax_ptr, stream, noop_ptr);
  } else if (input.dtype() == DType::kFloat16) {
    const auto *input_ptr = reinterpret_cast<const half *>(input.data.dptr);
    quantize_per_token_kernel::launch_compute_per_token_amax<half>(
        static_cast<int>(rows), static_cast<int>(cols), input_ptr, amax_ptr, stream, noop_ptr);
  } else if (input.dtype() == DType::kFloat32) {
    const auto *input_ptr = reinterpret_cast<const float *>(input.data.dptr);
    quantize_per_token_kernel::launch_compute_per_token_amax<float>(
        static_cast<int>(rows), static_cast<int>(cols), input_ptr, amax_ptr, stream, noop_ptr);
  } else {
    NVTE_ERROR(
        "Unsupported input dtype for per-token NVFP4 quantization. "
        "Expected BFloat16, Float16, or Float32.");
  }

  if (output->has_data()) {
    NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Rowwise output must have FP4 type.");
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Rowwise scaling tensor must be allocated.");
    NVTE_CHECK(output->amax.dptr != nullptr, "Rowwise per-token amax tensor must be allocated.");

    QuantizationConfig per_token_quant_config;
    if (quant_config != nullptr) {
      per_token_quant_config = *quant_config;
    }
    per_token_quant_config.nvfp4_per_token_activation = true;
    per_token_quant_config.nvfp4_2d_quantization = false;

    const bool use_optimized_kernel =
        (input.dtype() == DType::kBFloat16) && (rows % 32 == 0) && (cols % 32 == 0);
    if (use_optimized_kernel) {
      quantize_transpose</*use_2d_quantization=*/false>(input, noop, output,
                                                        &per_token_quant_config, stream);
    } else {
      quantize_transpose_vector_blockwise_fp4(
          /*input=*/input.data, /*global_amax=*/output->amax,
          /*scale_inv=*/output->scale_inv, /*scale_inv_t=*/output->columnwise_scale_inv,
          /*output=*/output->data, /*output_t=*/output->columnwise_data,
          /*epsilon=*/0.0f, /*return_identity=*/true, /*return_transpose=*/false,
          /*pow2_scale=*/false, /*swizzled_scale=*/false,
          /*use_stochastic_rounding=*/per_token_quant_config.stochastic_rounding,
          /*rng_state=*/per_token_quant_config.rng_state, /*use_2d_quantization=*/false,
          /*per_token_rowwise_scaling=*/true, /*noop_tensor=*/noop->data, /*stream=*/stream);
    }
  }
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_
