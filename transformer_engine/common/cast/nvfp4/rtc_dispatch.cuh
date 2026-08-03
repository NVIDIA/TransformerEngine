/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file rtc_dispatch.cuh
 *  \brief Host-side NVRTC dispatch for the NVFP4 4over6 quantize kernel.
 */

#ifndef TRANSFORMER_ENGINE_CAST_NVFP4_RTC_DISPATCH_CUH_
#define TRANSFORMER_ENGINE_CAST_NVFP4_RTC_DISPATCH_CUH_

#if !defined(__CUDACC_RTC__)

#include <cuda.h>

#include <string>

#include "../../util/rtc.h"
// do not include util/string.h here — it pulls in <regex>, which is heavy.
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace rtc_nvfp4 {

void compile_quantize_4over6_rtc(const std::string &kernel_label, const std::string &itype_name,
                                 bool use_2d, bool return_identity, bool return_transpose,
                                 bool row_scaled, const std::string &mode_name, bool err_fast_math,
                                 int e4m3_max);

template <typename T>
inline const char *rtc_type_name() {
  return detail::type_name<T>();
}
template <>
inline const char *rtc_type_name<fp16>() {
  return "fp16";
}
template <>
inline const char *rtc_type_name<bf16>() {
  return "bf16";
}

// Enum literal spelling for the __MODE__ substitution / cache key.
inline const char *rtc_mode_name(NVTENVFP44Over6Mode mode) {
  switch (mode) {
    case kNVTENVFP44Over6MinMSE:
      return "kNVTENVFP44Over6MinMSE";
    case kNVTENVFP44Over6MinMAE:
      return "kNVTENVFP44Over6MinMAE";
    default:
      NVTE_ERROR("Unsupported NVFP4 4over6 mode.");
  }
}

template <bool USE_2D_QUANTIZATION, typename Cfg, int E4M3_MAX, typename IType>
inline void launch_quantize_4over6_rtc(const IType *input, fp4e2m1x2 *output, fp4e2m1x2 *output_t,
                                       nvfp4_scale_t *scales, nvfp4_scale_t *scales_t,
                                       const float *amax_rowwise, const float *amax_colwise,
                                       size_t rows, size_t cols, size_t scale_stride,
                                       size_t scale_stride_t, const float *noop,
                                       bool return_identity, bool return_transpose,
                                       bool row_scaled_nvfp4, dim3 grid, dim3 block, size_t shmem,
                                       cudaStream_t stream) {
  const std::string itype_name = rtc_type_name<IType>();
  const std::string mode_name = rtc_mode_name(Cfg::mode);

  // Cache key encodes everything that varies the compiled kernel.
  const std::string kernel_label =
      std::string("quantize_4over6,itype=") + itype_name +
      ",use2d=" + (USE_2D_QUANTIZATION ? "1" : "0") + ",id=" + (return_identity ? "1" : "0") +
      ",t=" + (return_transpose ? "1" : "0") + ",rowscaled=" + (row_scaled_nvfp4 ? "1" : "0") +
      ",mode=" + mode_name + ",fastmath=" + (Cfg::err_use_fast_math ? "1" : "0") +
      ",e4m3max=" + std::to_string(E4M3_MAX);

  auto &mgr = rtc::KernelManager::instance();
  if (!mgr.is_compiled(kernel_label)) {
    compile_quantize_4over6_rtc(kernel_label, itype_name, USE_2D_QUANTIZATION, return_identity,
                                return_transpose, row_scaled_nvfp4, mode_name,
                                Cfg::err_use_fast_math, E4M3_MAX);
  }

  if (shmem > 0) {
    mgr.set_function_attribute(kernel_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(shmem));
  }

  mgr.launch(kernel_label, grid, block, static_cast<unsigned int>(shmem), stream, input, output,
             output_t, scales, scales_t, amax_rowwise, amax_colwise, rows, cols, scale_stride,
             scale_stride_t, noop);
}

}  // namespace rtc_nvfp4
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // !__CUDACC_RTC__

#endif  // TRANSFORMER_ENGINE_CAST_NVFP4_RTC_DISPATCH_CUH_
