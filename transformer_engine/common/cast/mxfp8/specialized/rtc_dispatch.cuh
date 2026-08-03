/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file rtc_dispatch.cuh
 *  \brief Host-side NVRTC dispatch for the specialized MXFP8 cast-only kernels.
 */

#ifndef TRANSFORMER_ENGINE_SPECIALIZED_MXFP8_RTC_DISPATCH_CUH_
#define TRANSFORMER_ENGINE_SPECIALIZED_MXFP8_RTC_DISPATCH_CUH_

#if !defined(__CUDACC_RTC__)

#include <cuda.h>

#include <string>

#include "../../../util/rtc.h"
// NB: do not include util/string.h here — it pulls in <regex>, which is heavy
#include "quantize_mxfp8.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

// Compile (if not already cached) the rowwise cast-only RTC kernel for the given
// element-type spellings.
void compile_rowwise_cast_only_rtc(const std::string &kernel_label, const std::string &itype_name,
                                   const std::string &otype_name);
void compile_bidimensional_cast_only_rtc(const std::string &kernel_label,
                                         const std::string &itype_name,
                                         const std::string &otype_name, int num_stages, int iter_n,
                                         bool use_cvt_4x);

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
template <>
inline const char *rtc_type_name<fp8e4m3>() {
  return "fp8e4m3";
}
template <>
inline const char *rtc_type_name<fp8e5m2>() {
  return "fp8e5m2";
}

template <typename IType, typename OType>
inline void launch_rowwise_cast_only_rtc(IType *input, OType *output, e8m0_t *scales_rowwise,
                                         const float *noop, int32_t rows, int32_t cols,
                                         int32_t scale_stride_rowwise, int32_t scale_stride_colwise,
                                         cudaStream_t stream) {
  using traits = CastTraits<IType, OType, /*rowwise=*/true, /*colwise=*/false>;

  const std::string itype_name = rtc_type_name<IType>();
  const std::string otype_name = rtc_type_name<OType>();

  const std::string kernel_label =
      std::string("quantize_mxfp8_rowwise_cast_only,itype=") + itype_name + ",otype=" + otype_name;

  auto &mgr = rtc::KernelManager::instance();
  if (!mgr.is_compiled(kernel_label)) {
    compile_rowwise_cast_only_rtc(kernel_label, itype_name, otype_name);
  }

  if (traits::smem > 0) {
    mgr.set_function_attribute(kernel_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(traits::smem));
  }

  dim3 block(traits::threadLayout::num, traits::warpLayout::N, traits::warpLayout::M);
  dim3 grid((cols + traits::blockDimN - 1) / traits::blockDimN,
            (rows + traits::blockDimM - 1) / traits::blockDimM);
  mgr.launch(kernel_label, grid, block, static_cast<unsigned int>(traits::smem), stream, input,
             output, scales_rowwise, noop, rows, cols, scale_stride_rowwise, scale_stride_colwise);
}

// Compile-time config for the bidimensional kernel.
struct BidimConfig {
  int32_t num_stages;
  int32_t iter_n;
  bool use_cvt_4x;
};

// Derive the default from the static kernel traits so the static and RTC paths
// cannot drift when the shipped bidimensional configuration changes.
template <typename IType, typename OType>
constexpr BidimConfig static_bidim_config() {
  using traits = CastTraits<IType, OType, /*rowwise=*/true, /*colwise=*/true>;
  using iter_layout = typename traits::iterLayout;
  return {traits::numStages, iter_layout::N, traits::_use_cvt_4x};
}

template <typename IType, typename OType>
inline BidimConfig select_bidim_config(int32_t rows, int32_t cols) {
  // No tuned bidimensional overrides are shipped for now. In the future,
  // this selector can be expanded with measured shape- or dtype-specific configurations while
  // leaving unlisted problems on the static kernel's configuration.
  (void)rows;
  (void)cols;
  return static_bidim_config<IType, OType>();
}

// Compile+launch one concrete bidimensional config. Geometry/smem come straight
// from BidimTunableTraits, so every config is exactly what the JIT'd kernel uses.
template <typename IType, typename OType, int32_t NS, int32_t ITN, bool CVT>
inline void launch_bidim_impl(const CUtensorMap &tensor_map_input,
                              const CUtensorMap &tensor_map_rowwise_output,
                              const CUtensorMap &tensor_map_colwise_output, e8m0_t *scales_rowwise,
                              e8m0_t *scales_colwise, const float *noop, int32_t rows, int32_t cols,
                              int32_t scale_stride_rowwise, int32_t scale_stride_colwise,
                              cudaStream_t stream, const std::string &itype_name,
                              const std::string &otype_name) {
  using traits = BidimTunableTraits<IType, OType, NS, ITN, CVT>;
  const std::string kernel_label = std::string("quantize_mxfp8_bidimensional_cast_only,itype=") +
                                   itype_name + ",otype=" + otype_name +
                                   ",ns=" + std::to_string(NS) + ",itn=" + std::to_string(ITN) +
                                   ",cvt=" + (CVT ? "4x" : "2x");
  auto &mgr = rtc::KernelManager::instance();
  if (!mgr.is_compiled(kernel_label)) {
    compile_bidimensional_cast_only_rtc(kernel_label, itype_name, otype_name, NS, ITN, CVT);
  }
  if (traits::smem > 0) {
    mgr.set_function_attribute(kernel_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(traits::smem));
  }
  dim3 block(traits::rowThreadLayout::num, traits::numWarps);
  dim3 grid((cols + traits::blockDIM::N - 1) / traits::blockDIM::N,
            (rows + traits::blockDIM::M - 1) / traits::blockDIM::M);
  mgr.launch(kernel_label, grid, block, static_cast<unsigned int>(traits::smem), stream,
             tensor_map_input, tensor_map_rowwise_output, tensor_map_colwise_output, scales_rowwise,
             scales_colwise, noop, rows, cols, scale_stride_rowwise, scale_stride_colwise);
}

// Compile (on first use) and launch the 32x32 bidimensional cast-only kernel via
// NVRTC, selecting the (numStages, iterN, cvt) config for this shape. TMA
// descriptors are built host-side by the caller (config-independent).
template <typename IType, typename OType>
inline void launch_bidimensional_cast_only_rtc(const CUtensorMap &tensor_map_input,
                                               const CUtensorMap &tensor_map_rowwise_output,
                                               const CUtensorMap &tensor_map_colwise_output,
                                               e8m0_t *scales_rowwise, e8m0_t *scales_colwise,
                                               const float *noop, int32_t rows, int32_t cols,
                                               int32_t scale_stride_rowwise,
                                               int32_t scale_stride_colwise, cudaStream_t stream) {
  const std::string itype_name = rtc_type_name<IType>();
  const std::string otype_name = rtc_type_name<OType>();
  const BidimConfig config = select_bidim_config<IType, OType>(rows, cols);
  constexpr BidimConfig default_config = static_bidim_config<IType, OType>();

  // The static configuration is always supported and is the only configuration
  // selected today.
  if (config.num_stages == default_config.num_stages && config.iter_n == default_config.iter_n &&
      config.use_cvt_4x == default_config.use_cvt_4x) {
    launch_bidim_impl<IType, OType, default_config.num_stages, default_config.iter_n,
                      default_config.use_cvt_4x>(
        tensor_map_input, tensor_map_rowwise_output, tensor_map_colwise_output, scales_rowwise,
        scales_colwise, noop, rows, cols, scale_stride_rowwise, scale_stride_colwise, stream,
        itype_name, otype_name);
    return;
  }

#define NVTE_MXFP8_BIDIM_CASE(NS, ITN, CVT)                                                     \
  if (config.num_stages == (NS) && config.iter_n == (ITN) && config.use_cvt_4x == (CVT)) {      \
    launch_bidim_impl<IType, OType, NS, ITN, CVT>(                                              \
        tensor_map_input, tensor_map_rowwise_output, tensor_map_colwise_output, scales_rowwise, \
        scales_colwise, noop, rows, cols, scale_stride_rowwise, scale_stride_colwise, stream,   \
        itype_name, otype_name);                                                                \
    return;                                                                                     \
  }
  // Keep candidate configurations available for a future tuning PR. The
  // selector above does not currently choose any of them. cvt is fixed at 4x.
  NVTE_MXFP8_BIDIM_CASE(2, 1, true)
  NVTE_MXFP8_BIDIM_CASE(2, 2, true)
  NVTE_MXFP8_BIDIM_CASE(2, 4, true)
  NVTE_MXFP8_BIDIM_CASE(2, 8, true)
  NVTE_MXFP8_BIDIM_CASE(2, 16, true)
  NVTE_MXFP8_BIDIM_CASE(3, 1, true)
  NVTE_MXFP8_BIDIM_CASE(3, 2, true)
  NVTE_MXFP8_BIDIM_CASE(3, 4, true)
  NVTE_MXFP8_BIDIM_CASE(3, 8, true)
  NVTE_MXFP8_BIDIM_CASE(3, 16, true)
  NVTE_MXFP8_BIDIM_CASE(4, 1, true)
  NVTE_MXFP8_BIDIM_CASE(4, 2, true)
  NVTE_MXFP8_BIDIM_CASE(4, 4, true)
  NVTE_MXFP8_BIDIM_CASE(4, 8, true)
  NVTE_MXFP8_BIDIM_CASE(4, 16, true)
#undef NVTE_MXFP8_BIDIM_CASE

  NVTE_ERROR("Unsupported MXFP8 bidimensional RTC config: num_stages=", config.num_stages,
             ", iter_n=", config.iter_n, ", use_cvt_4x=", config.use_cvt_4x);
}

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // !__CUDACC_RTC__

#endif  // TRANSFORMER_ENGINE_SPECIALIZED_MXFP8_RTC_DISPATCH_CUH_
