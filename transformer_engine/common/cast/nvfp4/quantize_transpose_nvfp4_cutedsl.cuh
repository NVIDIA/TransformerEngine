/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_QUANTIZE_TRANSPOSE_NVFP4_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_QUANTIZE_TRANSPOSE_NVFP4_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <optional>
#include <string>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"

namespace transformer_engine {
namespace cutedsl_backend {

using namespace tvm_ffi_bridge;

// Instantiation parameters of the CuTE DSL kernel
struct NVFP4QuantConfig {
  static constexpr const char *kEntrypointName = "get_nvfp4_quantization_function";

  // todo

  std::string to_key() const {
    std::string key;

    // todo

    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result = (*entrypoint)(tvm::ffi::String(fn_name), /* todo */);
    return result.try_cast<bool>().value_or(false);
  }
};

// Dispatch to Python side via TVM FFI
inline bool nvfp4_quantize_transpose_cutedsl(const NVFP4QuantConfig &config, const Tensor &input,
                                             const Tensor *noop, Tensor *output,
                                             const QuantizationConfig *quant_config,
                                             cudaStream_t stream) {
  // todo

  std::optional<tvm::ffi::Function> nvfp4_quant_func_opt =
      tvm_ffi_bridge::TVMFFICentral::getInstance().lazyload_function(config);
  if (!nvfp4_quant_func_opt.has_value()) {
    return false;
  }

  // todo

  (*nvfp4_quant_func_opt)(/* todo */);
  return true;
}

// CuTE DSL counterpart of nvfp4::quantize_transpose
template <bool use_2d_quantization>
bool nvfp4_quantize_transpose_cutedsl(const Tensor &input, const Tensor *noop, Tensor *output,
                                      const QuantizationConfig *quant_config, cudaStream_t stream) {
  // Currently, only the CuTE DSL counterpart of quantize_transpose_tuned_1D is supported
  if constexpr (use_2d_quantization) {
    return false;
  } else {
    if (input.dtype() != DType::kBFloat16) {
      return false;
    }
    const bool with_noop = noop != nullptr && noop->data.dptr != nullptr;
    const NVFP4QuantConfig config{
        // todo
    };
    return nvfp4_quantize_transpose_cutedsl(config, input, noop, output, quant_config, stream);
  }
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_QUANTIZE_TRANSPOSE_NVFP4_CUTEDSL_CUH_
