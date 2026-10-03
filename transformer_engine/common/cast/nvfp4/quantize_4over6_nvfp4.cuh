/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_4over6_nvfp4.cuh
 *  \brief Host dispatch for NVFP4 4over6 quantization.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>

#include "../../common.h"
#include "../../util/rtc.h"
#include "quantize_4over6_kernel.cuh"
#include "rtc_dispatch.h"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

#if FP4_TYPE_SUPPORTED

#define TRANSFORMER_ENGINE_NVFP4_4OVER6_MODE_SWITCH(MODE, MODE_CONST, ...) \
  switch (MODE) {                                                          \
    case kNVTENVFP44Over6MinMAE: {                                         \
      constexpr NVTENVFP44Over6Mode MODE_CONST = kNVTENVFP44Over6MinMAE;   \
      { __VA_ARGS__ }                                                      \
    } break;                                                               \
    case kNVTENVFP44Over6MinMSE: {                                         \
      constexpr NVTENVFP44Over6Mode MODE_CONST = kNVTENVFP44Over6MinMSE;   \
      { __VA_ARGS__ }                                                      \
    } break;                                                               \
    default: {                                                             \
      NVTE_ERROR("Unsupported NVFP4 4over6 mode.");                        \
    }                                                                      \
  }

#define TRANSFORMER_ENGINE_NVFP4_4OVER6_E4M3_MAX_SWITCH(E4M3_MAX_VALUE, E4M3_MAX_CONST, ...) \
  if ((E4M3_MAX_VALUE) == 256) {                                                             \
    constexpr int E4M3_MAX_CONST = 256;                                                      \
    { __VA_ARGS__ }                                                                          \
  } else {                                                                                   \
    NVTE_CHECK((E4M3_MAX_VALUE) == 448, "Unsupported NVFP4 E4M3 max.");                      \
    constexpr int E4M3_MAX_CONST = 448;                                                      \
    { __VA_ARGS__ }                                                                          \
  }

namespace quantize_4over6_kernel {

template <bool USE_2D_QUANTIZATION, typename Cfg, int E4M3_MAX, typename IType>
void launch_quantize_4over6(const Tensor &input, const Tensor *noop, Tensor *output,
                            cudaStream_t stream) {
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const bool row_scaled_nvfp4 = output->row_scaled_nvfp4;
  const bool return_identity = output->has_data();
  const bool return_transpose = output->has_columnwise_data();

  const auto *input_ptr = reinterpret_cast<const IType *>(input.data.dptr);
  auto *output_ptr = reinterpret_cast<fp4e2m1x2 *>(output->data.dptr);
  auto *output_t_ptr = reinterpret_cast<fp4e2m1x2 *>(output->columnwise_data.dptr);
  auto *scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  auto *scales_t_ptr = reinterpret_cast<nvfp4_scale_t *>(output->columnwise_scale_inv.dptr);
  const auto *amax_rowwise_ptr = reinterpret_cast<const float *>(output->amax.dptr);
  const auto *amax_colwise_ptr = reinterpret_cast<const float *>(output->columnwise_amax.dptr);
  const auto *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  const dim3 grid(DIVUP(cols, static_cast<size_t>(kTileCols)),
                  DIVUP(rows, static_cast<size_t>(kTileRows)));
  const dim3 block(kThreads);
  const size_t shmem = kPipelineStages * kStageRows * kTileCols * sizeof(IType);
  const size_t scale_stride = return_identity ? output->scale_inv.shape[1] : 0;
  const size_t scale_stride_t = return_transpose ? output->columnwise_scale_inv.shape[1] : 0;

#if NVTE_BUILD_LEGACY_STATIC_NVFP4
  const bool use_rtc = transformer_engine::rtc::is_enabled();
#else
  constexpr bool use_rtc = true;
#endif
  if (use_rtc) {
    rtc_nvfp4::launch_quantize_4over6_rtc<USE_2D_QUANTIZATION, Cfg, E4M3_MAX, IType>(
        input_ptr, output_ptr, output_t_ptr, scales_ptr, scales_t_ptr, amax_rowwise_ptr,
        amax_colwise_ptr, rows, cols, scale_stride, scale_stride_t, noop_ptr, return_identity,
        return_transpose, row_scaled_nvfp4, grid, block, shmem, stream);
  } else {
#if NVTE_BUILD_LEGACY_STATIC_NVFP4
    TRANSFORMER_ENGINE_SWITCH_CONDITION(return_identity, RETURN_IDENTITY, {
      TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE, {
        TRANSFORMER_ENGINE_SWITCH_CONDITION(row_scaled_nvfp4, ROW_SCALED_NVFP4, {
          auto kernel =
              quantize_4over6_kernel<USE_2D_QUANTIZATION, RETURN_IDENTITY, RETURN_TRANSPOSE,
                                     ROW_SCALED_NVFP4, Cfg, E4M3_MAX, IType>;
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
          kernel<<<grid, block, shmem, stream>>>(
              input_ptr, output_ptr, output_t_ptr, scales_ptr, scales_t_ptr, amax_rowwise_ptr,
              amax_colwise_ptr, rows, cols, scale_stride, scale_stride_t, noop_ptr);
        });
      });
    });
#else
    NVTE_ERROR(
        "NVFP4 4over6 quantize kernel requires NVRTC. Unset NVTE_DISABLE_NVRTC, or rebuild with "
        "NVTE_BUILD_LEGACY_STATIC_NVFP4=ON for the static fallback.");
#endif
  }
}

}  // namespace quantize_4over6_kernel

#endif  // FP4_TYPE_SUPPORTED

template <bool use_2d_quantization>
void quantize_4over6(const Tensor &input, const Tensor *noop, Tensor *output,
                     const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_4over6_kernel;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(quant_config != nullptr && quant_config->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled,
             "NVFP4 4over6 quantization requires a non-disabled 4over6 mode.");
  NVTE_CHECK(!quant_config->stochastic_rounding,
             "NVFP4 4over6 quantization does not support stochastic rounding.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "NVFP4 4over6 output tensor must have rowwise or columnwise data.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales, "Output must have scales in compact format.");
  NVTE_CHECK(input.flat_last_dim() % kGroupSize == 0,
             "NVFP4 4over6 quantization requires columns divisible by ", kGroupSize, ".");
  NVTE_CHECK(!(output->has_columnwise_data() || use_2d_quantization) ||
                 input.flat_first_dim() % kGroupSize == 0,
             "NVFP4 4over6 columnwise or 2D quantization requires rows divisible by ", kGroupSize,
             ".");
  NVTE_CHECK(!output->row_scaled_nvfp4 || !use_2d_quantization,
             "Row-scaled NVFP4 quantization does not support 2D quantization.");
  NVTE_CHECK(!output->row_scaled_nvfp4 || !output->has_columnwise_data(),
             "Row-scaled NVFP4 quantization does not produce columnwise output.");
  NVTE_CHECK(!use_2d_quantization || output->has_data(),
             "NVFP4 4over6 2D quantization requires rowwise output.");

  if (output->has_data()) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
    NVTE_CHECK(output->amax.dptr != nullptr, "Rowwise amax tensor must be allocated.");
    NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  }
  if (output->has_columnwise_data()) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Transposed scaling tensor must be allocated.");
    NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
               "Transposed output must have FP4 type.");
    NVTE_CHECK(output->columnwise_amax.dptr != nullptr || output->amax.dptr != nullptr,
               "NVFP4 4over6 columnwise quantization requires columnwise amax or rowwise amax.");
  }

  TRANSFORMER_ENGINE_NVFP4_4OVER6_E4M3_MAX_SWITCH(
      output->nvfp4_e4m3_max, E4M3_MAX,
      TRANSFORMER_ENGINE_NVFP4_4OVER6_MODE_SWITCH(
          quant_config->nvfp4_4over6_mode, MODE,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              quant_config->nvfp4_4over6_err_use_fast_math, ERR_USE_FAST_MATH, {
                using Cfg = quantize_4over6_kernel::Config<MODE, ERR_USE_FAST_MATH>;
                TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
                    input.dtype(), IType,
                    quantize_4over6_kernel::launch_quantize_4over6<use_2d_quantization, Cfg,
                                                                   E4M3_MAX, IType>(
                        input, noop, output, stream););
              });););

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
