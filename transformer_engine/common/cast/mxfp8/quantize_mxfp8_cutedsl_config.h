/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_QUANTIZE_MXFP8_CUTEDSL_CONFIG_H_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_QUANTIZE_MXFP8_CUTEDSL_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

#include "../../tvm_ffi_bridge.h"

namespace transformer_engine {
namespace cutedsl_backend {

using namespace tvm_ffi_bridge;

struct MXFP8QuantConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_quantization_function";

  DType dtype;                       // The input format
  DType fp8_dtype;                   // The fp8 output format
  bool rowwise;                      // If quantize rowwisely
  bool colwise;                      // If quantize columnwisely
  bool swizzled;                     // If the scale output is used for cudnn's swizzled layout
  bool with_amax;                    // If the kernel should return the amax
  bool with_dbias = false;           // If the dbias is computated (via the workspace tensor)
  bool with_dact = false;            // If an activation derivative operation is fused
  bool with_act = false;             // If an activation operation is fused
  bool use_2d_quantization = false;  // If use 2D quantization
  Activation activation = Activation::kNone;
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  uint32_t to_id() const {
    static_assert(static_cast<uint32_t>(DType::kNumTypes) <= 16,
                  "DType no longer fits in the 4 bits to_id() gives it.");
    static_assert(static_cast<uint32_t>(Activation::kNumTypes) <= 64,
                  "Activation no longer fits in the 6 bits to_id() gives it.");
    NVTE_CHECK(sm_arch < 512, "SM architecture no longer fits in the 9 bits to_id() gives it.");
    return static_cast<uint32_t>(dtype) | (static_cast<uint32_t>(fp8_dtype) << 4) |
           (static_cast<uint32_t>(rowwise) << 8) | (static_cast<uint32_t>(colwise) << 9) |
           (static_cast<uint32_t>(swizzled) << 10) | (static_cast<uint32_t>(with_amax) << 11) |
           (static_cast<uint32_t>(with_dbias) << 12) | (static_cast<uint32_t>(with_dact) << 13) |
           (static_cast<uint32_t>(with_act) << 14) |
           (static_cast<uint32_t>(use_2d_quantization) << 15) |
           (static_cast<uint32_t>(activation) << 16) | (sm_arch << 22);
  }

  std::optional<tvm::ffi::Function> get_kernel() const {
    static TVMFFIConfigCache &cache = TVMFFIConfigCache::create();
    return cache.get_or_load(*this);
  }

  // Globally unique TVM-FFI registry key used when the CuTeDSL function is
  // compiled and registered on a cache miss.
  std::string to_key() const {
    std::string key;
    key.reserve(72);  // longest: cutedsl_mxfp8_smXXX_bf16_fp8_e4m3fn_..._dqgelu
    key.append("cutedsl_mxfp8_sm")
        .append(std::to_string(sm_arch))
        .append("_")
        .append(te_dtype_to_str(dtype))
        .append("_")
        .append(te_dtype_to_str(fp8_dtype))
        .append("_")
        .append(rowwise ? "1" : "0")
        .append("_")
        .append(colwise ? "1" : "0")
        .append("_")
        .append(swizzled ? "1" : "0")
        .append("_")
        .append(with_amax ? "1" : "0")
        .append("_")
        .append(with_dbias ? "1" : "0")
        .append("_")
        .append(with_dact ? "1" : "0")
        .append("_")
        .append(with_act ? "1" : "0")
        .append("_")
        .append(use_2d_quantization ? "1" : "0")
        .append("_")
        .append(activation_to_str(activation));
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result =
        (*entrypoint)(tvm::ffi::String(fn_name), tvm::ffi::String(te_dtype_to_str(dtype)),
                      tvm::ffi::String(te_dtype_to_str(fp8_dtype)), rowwise, colwise, swizzled,
                      with_amax, with_dbias, with_dact, with_act, use_2d_quantization,
                      tvm::ffi::String(activation_to_str(activation)));
    return result.try_cast<bool>().value_or(false);
  }
};

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_QUANTIZE_MXFP8_CUTEDSL_CONFIG_H_
