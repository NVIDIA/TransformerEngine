/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <string>
#include <vector>

#include "../../util/rtc.h"
#include "../../util/string.h"

// Generated string headers: raw source of the RTC kernel and the device headers
// it needs as in-memory includes
#include "string_code_cast_nvfp4_core_nvfp4_cuh.h"
#include "string_code_cast_nvfp4_quantize_4over6_kernel_cuh.h"
#include "string_code_cast_nvfp4_rtc_quantize_4over6_cu.h"
#include "string_code_transformer_engine_nvfp4_4over6_h.h"
#include "string_code_util_ptx_cuh.h"
#include "string_code_util_type_extrema_h.h"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace rtc_nvfp4 {

void compile_quantize_4over6_rtc(const std::string &kernel_label, const std::string &itype_name,
                                 bool use_2d, bool return_identity, bool return_transpose,
                                 bool row_scaled, const std::string &mode_name, bool err_fast_math,
                                 int e4m3_max) {
  auto &mgr = rtc::KernelManager::instance();
  if (mgr.is_compiled(kernel_label)) {
    return;
  }

  auto bool_str = [](bool value) -> std::string { return value ? "true" : "false"; };

  std::string code = string_code_cast_nvfp4_rtc_quantize_4over6_cu;
  code = regex_replace(code, "__ITYPE__", itype_name);
  code = regex_replace(code, "__USE_2D__", bool_str(use_2d));
  code = regex_replace(code, "__RETURN_IDENTITY__", bool_str(return_identity));
  code = regex_replace(code, "__RETURN_TRANSPOSE__", bool_str(return_transpose));
  code = regex_replace(code, "__ROW_SCALED__", bool_str(row_scaled));
  code = regex_replace(code, "__MODE__", mode_name);
  code = regex_replace(code, "__ERR_FAST_MATH__", bool_str(err_fast_math));
  code = regex_replace(code, "__E4M3_MAX__", std::to_string(e4m3_max));

  const std::vector<rtc::Header> headers = {
      {string_code_cast_nvfp4_quantize_4over6_kernel_cuh, "quantize_4over6_kernel.cuh"},
      {string_code_cast_nvfp4_core_nvfp4_cuh, "core_nvfp4.cuh"},
      {string_code_util_ptx_cuh, "ptx.cuh"},
      {string_code_util_type_extrema_h, "util/type_extrema.h"},
      {string_code_transformer_engine_nvfp4_4over6_h, "transformer_engine/nvfp4/4over6.h"},
  };

  // --device-int128: ptx.cuh uses __uint128_t; -default-device: treat the
  // unannotated constexpr/inline helpers in ptx.cuh as __device__ under JIT.
  // -DCUDA_VERSION: NVRTC does not predefine CUDA_VERSION, but the 4over6 kernel
  // needs the FP4 types gated behind FP4_TYPE_SUPPORTED (== CUDA_VERSION >= 12080)
  // and <cuda_fp4.h>; forward the build's CUDA version so those are enabled.
  const std::vector<std::string> options = {"--device-int128", "-default-device",
                                            "-DCUDA_VERSION=" + std::to_string(CUDA_VERSION)};
  constexpr rtc::ArchRequirement arch_requirement{100, rtc::ArchSpecificity::ArchitectureSpecific};

  mgr.compile(kernel_label, "quantize_4over6_rtc_kernel", code,
              "transformer_engine/common/cast/nvfp4/rtc/quantize_4over6.cu", options, headers,
              arch_requirement);
}

}  // namespace rtc_nvfp4
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine
