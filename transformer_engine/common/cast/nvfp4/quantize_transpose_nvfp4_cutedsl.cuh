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
#if FP4_TYPE_SUPPORTED

using namespace tvm_ffi_bridge;

// Instantiation parameters of the CuTE DSL kernel
struct NVFP4QuantConfig {
  static constexpr const char *kEntrypointName = "get_nvfp4_quantization_function";

  bool use_stochastic_rounding;  // If stochastic rounding is applied during quantization
  bool use_fast_math;            // If fast math approximations are used during quantization
  bool row_scaled_nvfp4;         // If scales are computed per row (row-scaled NVFP4)
  bool return_transpose;         // If columnwise (transposed) output is produced

  std::string to_key() const {
    std::string key;
    key.reserve(32);
    key.append("cutedsl_nvfp4_")
        .append(use_stochastic_rounding ? "1" : "0")
        .append("_")
        .append(use_fast_math ? "1" : "0")
        .append("_")
        .append(row_scaled_nvfp4 ? "1" : "0")
        .append("_")
        .append(return_transpose ? "1" : "0");
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result = (*entrypoint)(tvm::ffi::String(fn_name), use_stochastic_rounding,
                                         use_fast_math, row_scaled_nvfp4, return_transpose);
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
#endif  // FP4_TYPE_SUPPORTED

// CuTE DSL counterpart of nvfp4::quantize_transpose
template <bool use_2d_quantization>
bool nvfp4_quantize_transpose_cutedsl(const Tensor &input, const Tensor *noop, Tensor *output,
                                      const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  // Currently, only the CuTE DSL counterpart of quantize_transpose_tuned_1D is supported
  if constexpr (use_2d_quantization) {
    return false;
  } else {
    if (input.dtype() != DType::kBFloat16) {
      return false;
    }
    const bool use_stochastic_rounding =
        quant_config != nullptr ? quant_config->stochastic_rounding : false;
    const bool use_fast_math = quant_config != nullptr ? quant_config->use_fast_math : false;
    const bool row_scaled_nvfp4 = output->row_scaled_nvfp4;
    const bool return_transpose = output->has_columnwise_data();
    const NVFP4QuantConfig config{
        use_stochastic_rounding,
        use_fast_math,
        row_scaled_nvfp4,
        return_transpose,
    };

    // Checks mirroring the CUDA version
    checkCuDriverContext(stream);
    CheckNoopTensor(*noop, "cast_noop");
    CheckInputTensor(input, "input");
    CheckOutputTensor(*output, "output", false);

    NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
    NVTE_CHECK(output->has_data(), "NVFP4 output tensor must be allocated.");
    NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
    NVTE_CHECK(!row_scaled_nvfp4 || output->amax.dptr != nullptr,
               "Row-scaled NVFP4 quantization requires rowwise amax.");
    NVTE_CHECK(!row_scaled_nvfp4 || !output->has_columnwise_data(),
               "Row-scaled NVFP4 quantization does not produce columnwise output.");

    if (return_transpose) {
      NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
                 "Transposed output must have FP4 type.");
      NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
                 "Transposed scaling tensor must be allocated");
    }

    const auto [rows, cols] = input.flat_2d_dims();

    NVTE_CHECK(rows % 32 == 0,
               "Number of tensor rows must be a multiple of 32");  // 16B alignment for TMA
    NVTE_CHECK(cols % 32 == 0,
               "Number of tensor cols must be a multiple of 32");  // 16B alignment for TMA

    const NVTETensor rng_state_tensor =
        (quant_config != nullptr) ? quant_config->rng_state : nullptr;
    const size_t *rng_state = nullptr;
    if (rng_state_tensor != nullptr) {
      Tensor &rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
      NVTE_CHECK(rng_state_te_tensor.dtype() == DType::kInt64,
                 "RNG state should contain 2 64-bit values.");
      NVTE_CHECK(rng_state_te_tensor.data.shape == Shape{2},
                 "Shape of the RNG state should be [2], but got ", rng_state_te_tensor.data.shape);
    }

    return nvfp4_quantize_transpose_cutedsl(config, input, noop, output, quant_config, stream);
  }
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_QUANTIZE_TRANSPOSE_NVFP4_CUTEDSL_CUH_
