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
#include <cstdint>
#include <optional>
#include <string>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"

namespace transformer_engine {
namespace cutedsl_backend {
#if FP4_TYPE_SUPPORTED

// DLTensorWrapper and TVMFFICentral live in transformer_engine::tvm_ffi_bridge.
using namespace tvm_ffi_bridge;

// Instantiation parameters of the CuTeDSL kernel
struct NVFP4QuantConfig {
  static constexpr const char *kEntrypointName = "get_nvfp4_quantization_function";

  bool use_stochastic_rounding;  // If stochastic rounding is applied during quantization
  bool use_fast_math;            // If fast math approximations are used during quantization
  bool row_scaled_nvfp4;         // If scales are computed per row (row-scaled NVFP4)
  bool return_transpose;         // If columnwise (transposed) output is produced
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  // Bit layout: flags [3:0], SM architecture [12:4] (9 used/reserved).
  uint32_t to_id() const {
    NVTE_CHECK(sm_arch < 512, "SM architecture no longer fits in the 9 bits to_id() gives it.");
    return static_cast<uint32_t>(use_stochastic_rounding) |
           (static_cast<uint32_t>(use_fast_math) << 1) |
           (static_cast<uint32_t>(row_scaled_nvfp4) << 2) |
           (static_cast<uint32_t>(return_transpose) << 3) | (sm_arch << 4);
  }

  std::optional<tvm::ffi::Function> get_kernel() const {
    static TVMFFIConfigCache &cache = TVMFFIConfigCache::create();
    return cache.get_or_load(*this);
  }

  // Globally unique TVM-FFI registry key used when the CuTeDSL function is
  // compiled and registered on a cache miss.
  std::string to_key() const {
    std::string key;
    key.reserve(32);
    key.append("cutedsl_nvfp4_sm")
        .append(std::to_string(sm_arch))
        .append("_")
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
  std::optional<tvm::ffi::Function> nvfp4_quant_func_opt = config.get_kernel();
  if (!nvfp4_quant_func_opt.has_value()) {
    return false;
  }

  // Data tensors auto-flatten to 2D (DLTensorWrapper's default); the amaxes and the RNG state are
  // rank-1 in the compiled ABI.
  const int32_t device_index = transformer_engine::cuda::current_device();
  DLTensorWrapper mX(input.data, true, device_index);
  DLTensorWrapper mO_row(output->data, true, device_index);
  DLTensorWrapper mS_row(output->scale_inv, true, device_index);
  DLTensorWrapper mAmaxRow(output->amax, false, device_index);

  DLTensorWrapper mO_col, mS_col, mAmaxCol;
  if (config.return_transpose) {
    // The transposed output of a rank > 2 input is shaped {cols, d0, d1, ...}, which the default
    // flattening would collapse to (cols * d0, ...) rather than (cols, rows).
    const auto [rows, cols] = input.flat_2d_dims();
    const SimpleTensor col_data(output->columnwise_data.dptr, {cols, rows},
                                output->columnwise_data.dtype);
    mO_col = DLTensorWrapper(col_data, false, device_index);
    mS_col = DLTensorWrapper(output->columnwise_scale_inv, true, device_index);
    mAmaxCol = DLTensorWrapper(output->columnwise_amax, false, device_index);
  }

  // Optional Philox RNG state ({seed, offset}); required (and only used) for SR.
  DLTensorWrapper mRngState;
  if (config.use_stochastic_rounding) {
    const NVTETensor rng_state_tensor =
        (quant_config != nullptr) ? quant_config->rng_state : nullptr;
    NVTE_CHECK(rng_state_tensor != nullptr,
               "Stochastic rounding requires an RNG state tensor to be provided.");
    Tensor &rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
    mRngState = DLTensorWrapper(rng_state_te_tensor.data, false, device_index);
  }

  // noop passed as ptr because the null-check must be done at runtime on device
  void *noop_ptr = (noop != nullptr) ? noop->data.dptr : nullptr;
  // noop and stream are tvm-ffi opaque "handles"; pass them as void*.
  (*nvfp4_quant_func_opt)(&mX, &mO_row, &mS_row, &mO_col, &mS_col, &mAmaxRow, &mAmaxCol, noop_ptr,
                          &mRngState, static_cast<void *>(stream));
  return true;
}
#endif  // FP4_TYPE_SUPPORTED

// CuTeDSL counterpart of nvfp4::quantize_transpose. Returns false to fall back to the CUDA kernel.
template <bool use_2d_quantization>
bool nvfp4_quantize_transpose_cutedsl(const Tensor &input, const Tensor *noop, Tensor *output,
                                      const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  if (!TVMFFICentral::getInstance().get_cutedsl_backend_enabled()) {
    maybe_warn_cutedsl_not_chosen("the CuTeDSL backend is disabled.");
    return false;
  }
  // Currently, only the CuTeDSL counterpart of quantize_transpose_tuned_1D is supported
  if constexpr (use_2d_quantization) {
    maybe_warn_cutedsl_not_chosen("2D NVFP4 quantization is not supported.");
    return false;
  } else {
    if (input.dtype() != DType::kBFloat16) {
      maybe_warn_cutedsl_not_chosen("NVFP4 quantization supports only BF16 input.");
      return false;
    }
    if (output->with_gemm_swizzled_scales) {
      maybe_warn_cutedsl_not_chosen("NVFP4 quantization does not produce GEMM-swizzled scales.");
      return false;
    }
    if (output->has_data() && output->scale_inv.dtype != DType::kFloat8E4M3) {
      maybe_warn_cutedsl_not_chosen("NVFP4 quantization supports only E4M3 scales.");
      return false;
    }
    const bool return_transpose = output->has_columnwise_data();
    // The compiled ABI takes the amaxes as tensors, not optionals.
    if (output->amax.dptr == nullptr ||
        (return_transpose && output->columnwise_amax.dptr == nullptr)) {
      maybe_warn_cutedsl_not_chosen("NVFP4 quantization requires the amax tensors.");
      return false;
    }
    const auto [rows, cols] = input.flat_2d_dims();
    if (rows % 32 != 0 || cols % 32 != 0) {
      maybe_warn_cutedsl_not_chosen("NVFP4 quantization needs rows and columns divisible by 32.");
      return false;
    }

    const bool use_stochastic_rounding =
        quant_config != nullptr ? quant_config->stochastic_rounding : false;
    const bool use_fast_math = quant_config != nullptr ? quant_config->use_fast_math : false;
    const bool row_scaled_nvfp4 = output->row_scaled_nvfp4;
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
    NVTE_CHECK(!row_scaled_nvfp4 || !output->has_columnwise_data(),
               "Row-scaled NVFP4 quantization does not produce columnwise output.");

    if (return_transpose) {
      NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
                 "Transposed output must have FP4 type.");
      NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
                 "Transposed scaling tensor must be allocated");
    }

    const NVTETensor rng_state_tensor =
        (quant_config != nullptr) ? quant_config->rng_state : nullptr;
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
