/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <string>
#include <vector>

#include "../../../util/rtc.h"
#include "../../../util/string.h"

// Generated string headers: raw source of the RTC kernel and the device headers
// it needs as in-memory includes. These are large; keeping this in a host-only
// .cpp (never compiled by cicc) avoids blowing up the device compiler in every
// TU that merely launches the kernel.
#include "string_code_cast_mxfp8_specialized_quantize_mxfp8_cuh.h"
#include "string_code_cast_mxfp8_specialized_rtc_quantize_mxfp8_bidimensional_cu.h"
#include "string_code_cast_mxfp8_specialized_rtc_quantize_mxfp8_rowwise_cu.h"
#include "string_code_cast_mxfp8_specialized_state_counter_cuh.h"
#include "string_code_cast_mxfp8_specialized_swizzle_cuh.h"
#include "string_code_util_ptx_cuh.h"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

void compile_rowwise_cast_only_rtc(const std::string &kernel_label, const std::string &itype_name,
                                   const std::string &otype_name) {
  auto &mgr = rtc::KernelManager::instance();
  if (mgr.is_compiled(kernel_label)) {
    return;
  }

  std::string code = string_code_cast_mxfp8_specialized_rtc_quantize_mxfp8_rowwise_cu;
  code = regex_replace(code, "__ITYPE__", itype_name);
  code = regex_replace(code, "__OTYPE__", otype_name);

  const std::vector<rtc::Header> headers = {
      {string_code_cast_mxfp8_specialized_quantize_mxfp8_cuh, "specialized_quantize_mxfp8.cuh"},
      {string_code_util_ptx_cuh, "ptx.cuh"},
      {string_code_cast_mxfp8_specialized_state_counter_cuh, "state_counter.cuh"},
      {string_code_cast_mxfp8_specialized_swizzle_cuh, "swizzle.cuh"},
  };

  // --device-int128: ptx.cuh uses __uint128_t; -default-device: treat the
  // unannotated constexpr/inline helpers in ptx.cuh as __device__ under JIT.
  const std::vector<std::string> options = {"--device-int128", "-default-device"};
  constexpr rtc::ArchRequirement arch_requirement{100, rtc::ArchSpecificity::BlackwellSpecific};

  mgr.compile(kernel_label, "quantize_mxfp8_rowwise_rtc_kernel", code,
              "transformer_engine/common/cast/mxfp8/specialized/rtc/quantize_mxfp8_rowwise.cu",
              options, headers, arch_requirement);
}

void compile_bidimensional_cast_only_rtc(const std::string &kernel_label,
                                         const std::string &itype_name,
                                         const std::string &otype_name, int num_stages, int iter_n,
                                         bool use_cvt_4x) {
  auto &mgr = rtc::KernelManager::instance();
  if (mgr.is_compiled(kernel_label)) {
    return;
  }

  std::string code = string_code_cast_mxfp8_specialized_rtc_quantize_mxfp8_bidimensional_cu;
  code = regex_replace(code, "__ITYPE__", itype_name);
  code = regex_replace(code, "__OTYPE__", otype_name);
  code = regex_replace(code, "__NUM_STAGES__", std::to_string(num_stages));
  code = regex_replace(code, "__ITER_N__", std::to_string(iter_n));
  code = regex_replace(code, "__USE_CVT_4X__", use_cvt_4x ? "true" : "false");

  const std::vector<rtc::Header> headers = {
      {string_code_cast_mxfp8_specialized_quantize_mxfp8_cuh, "specialized_quantize_mxfp8.cuh"},
      {string_code_util_ptx_cuh, "ptx.cuh"},
      {string_code_cast_mxfp8_specialized_state_counter_cuh, "state_counter.cuh"},
      {string_code_cast_mxfp8_specialized_swizzle_cuh, "swizzle.cuh"},
  };
  const std::vector<std::string> options = {"--device-int128", "-default-device"};
  constexpr rtc::ArchRequirement arch_requirement{100, rtc::ArchSpecificity::BlackwellSpecific};

  mgr.compile(
      kernel_label, "quantize_mxfp8_bidimensional_rtc_kernel", code,
      "transformer_engine/common/cast/mxfp8/specialized/rtc/quantize_mxfp8_bidimensional.cu",
      options, headers, arch_requirement);
}

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
