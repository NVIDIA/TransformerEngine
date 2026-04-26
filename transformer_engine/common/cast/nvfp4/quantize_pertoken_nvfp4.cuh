/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_pertoken_nvfp4.cuh
 *  \brief CUDA kernels to cast to NVFP4 with per-token (per-row) global scaling.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <type_traits>

#include "../../common.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace quantize_pertoken_kernel {

using namespace core;
using namespace ptx;

constexpr int PERTOKEN_BLOCK_SIZE = 256;
constexpr int PERTOKEN_SF_VEC_SIZE = 16;

template <typename IType, int BLOCK_SIZE>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(BLOCK_SIZE)
#endif
    quantize_pertoken_nvfp4_kernel(
        const int num_rows, const int num_cols, const IType *__restrict__ input,
        const int *__restrict__ row_offsets, uint8_t *__restrict__ output_data,
        fp8e4m3 *__restrict__ output_scales, float *__restrict__ output_per_token_amax,
        const int scale_stride, const float *__restrict__ noop) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using namespace detail;
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  using IType2 = typename ptx::FPx2<IType>;

  const int row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  const int actual_row = (row_offsets != nullptr) ? row_offsets[row_idx] : row_idx;
  if (actual_row < 0) return;

  const int num_vec2 = num_cols / 2;
  const IType2 *input_row = reinterpret_cast<const IType2 *>(input + actual_row * num_cols);

  IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
  for (int i = threadIdx.x; i < num_vec2; i += BLOCK_SIZE) {
    const IType2 val = input_row[i];
    ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, val);
  }
  const float thread_max =
      static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float row_amax =
      BlockReduce(temp_storage).Reduce(thread_max, [](float a, float b) { return fmaxf(a, b); });

  __shared__ float shared_s_enc;
  if (threadIdx.x == 0) {
    const float s_enc = compute_global_encode_scaling_factor_FP4(row_amax);
    output_per_token_amax[row_idx] = row_amax;
    shared_s_enc = s_enc;
  }
  __syncthreads();
  const float S_enc = shared_s_enc;
  const float S_dec_rowwise = 1.0 / S_enc;
  constexpr float fp4_max_inv = 1.0f / detail::TypeExtrema<fp4e2m1>::max;
  const float global_encode_scale_multiplier = S_enc * fp4_max_inv;

  const int num_sf_blocks = num_cols / PERTOKEN_SF_VEC_SIZE;
  for (int sf_idx = threadIdx.x; sf_idx < num_sf_blocks; sf_idx += BLOCK_SIZE) {
    const int col_start = sf_idx * PERTOKEN_SF_VEC_SIZE;

    IType2 block_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
    alignas(8) IType2 vals[PERTOKEN_SF_VEC_SIZE / 2];
    const IType2 *input_block =
        reinterpret_cast<const IType2 *>(input + actual_row * num_cols + col_start);
    for (int j = 0; j < PERTOKEN_SF_VEC_SIZE / 2; ++j) {
      vals[j] = input_block[j];
      ptx::abs_max_2x(block_amax_2x, block_amax_2x, vals[j]);
    }
    const float block_max =
        static_cast<float>(__hmax(__habs(block_amax_2x.x), __habs(block_amax_2x.y)));

    const float S_dec_b_f32 =
        fminf(block_max * global_encode_scale_multiplier, detail::TypeExtrema<float>::max);
    const nvfp4_scale_t S_dec_b_fp8 = static_cast<nvfp4_scale_t>(S_dec_b_f32);
    output_scales[row_idx * scale_stride + sf_idx] = S_dec_b_fp8;

    constexpr float float_max = detail::TypeExtrema<float>::max;
    const float block_scale_inverse =
        fminf(1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);
    const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

    uint8_t *out_ptr = output_data + actual_row * (num_cols / 2) + col_start / 2;
    if constexpr (std::is_same_v<IType, __nv_bfloat16>) {
      auto *out_fp4_8x = reinterpret_cast<uint32_t *>(out_ptr);
      for (int j = 0; j < PERTOKEN_SF_VEC_SIZE / 2; j += 4) {
        const uint64_t elts03 = *reinterpret_cast<const uint64_t *>(&vals[j]);
        const uint64_t elts47 = *reinterpret_cast<const uint64_t *>(&vals[j + 2]);
        out_fp4_8x[j / 4] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(
            elts03, elts47, block_scale_inverse);
      }
    } else {
      auto *out_fp4 = reinterpret_cast<fp4e2m1x4 *>(out_ptr);
      for (int j = 0; j < PERTOKEN_SF_VEC_SIZE / 2; j += 2) {
        const float2 in01 =
            make_float2(static_cast<float>(vals[j].x), static_cast<float>(vals[j].y));
        const float2 in23 =
            make_float2(static_cast<float>(vals[j + 1].x), static_cast<float>(vals[j + 1].y));
        out_fp4[j / 2] = ptx::mul_cvt_fp32_to_fp4_4x</*USE_STOCHASTIC_ROUNDING=*/false>(
            in01, in23, block_scale_inverse_2x, /*rbits=*/0u);
      }
    }
  }
#endif
}

template <typename IType>
void launch_quantize_pertoken_nvfp4(const int num_rows, const int num_cols, const IType *input,
                                    const int *row_offsets, uint8_t *output_data,
                                    fp8e4m3 *output_scales, float *output_per_token_amax,
                                    const int scale_stride, cudaStream_t stream,
                                    const float *noop = nullptr) {
#if FP4_TYPE_SUPPORTED
  if (num_rows == 0 || num_cols == 0) return;

  NVTE_CHECK(num_cols % PERTOKEN_SF_VEC_SIZE == 0, "num_cols must be a multiple of ",
             PERTOKEN_SF_VEC_SIZE, " for per-token NVFP4 quantization, got ", num_cols);
  dim3 grid(num_rows);
  dim3 block(PERTOKEN_BLOCK_SIZE);

  quantize_pertoken_nvfp4_kernel<IType, PERTOKEN_BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(num_rows, num_cols, input, row_offsets, output_data,
                                   output_scales, output_per_token_amax, scale_stride, noop);
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif
}

}  // namespace quantize_pertoken_kernel

inline void quantize_pertoken(const Tensor &input, const Tensor *noop, Tensor *output,
                              cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->has_data(), "NVFP4 output tensor must be allocated.");
  NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  NVTE_CHECK(output->amax.dptr != nullptr, "Per-token amax tensor must be allocated.");
  NVTE_CHECK(!output->has_columnwise_data(),
             "Per-token NVFP4 quantization supports rowwise-only output.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales, "Output must have scales in compact format.");

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  NVTE_CHECK(cols % quantize_pertoken_kernel::PERTOKEN_SF_VEC_SIZE == 0,
             "Per-token NVFP4 quantization requires last dim divisible by ",
             quantize_pertoken_kernel::PERTOKEN_SF_VEC_SIZE, ".");

  const auto *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  auto *data_ptr = reinterpret_cast<uint8_t *>(output->data.dptr);
  auto *scale_ptr = reinterpret_cast<fp8e4m3 *>(output->scale_inv.dptr);
  auto *amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  const int *row_offsets = nullptr;
  const int scale_stride = static_cast<int>(output->scale_inv.shape.back());

  if (input.dtype() == DType::kBFloat16) {
    quantize_pertoken_kernel::launch_quantize_pertoken_nvfp4<__nv_bfloat16>(
        static_cast<int>(rows), static_cast<int>(cols),
        reinterpret_cast<const __nv_bfloat16 *>(input.data.dptr), row_offsets, data_ptr, scale_ptr,
        amax_ptr, scale_stride, stream, noop_ptr);
  } else if (input.dtype() == DType::kFloat16) {
    quantize_pertoken_kernel::launch_quantize_pertoken_nvfp4<half>(
        static_cast<int>(rows), static_cast<int>(cols),
        reinterpret_cast<const half *>(input.data.dptr), row_offsets, data_ptr, scale_ptr, amax_ptr,
        scale_stride, stream, noop_ptr);
  } else {
    NVTE_ERROR(
        "Unsupported input dtype for per-token NVFP4 quantization. "
        "Expected BFloat16 or Float16.");
  }
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_
