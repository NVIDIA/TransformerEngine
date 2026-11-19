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
#include "cutedsl_gemm_swizzled_scales.h"

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
  bool swizzled;                 // If the scales are written in the GEMM-swizzled layout
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  // Bit layout: flags [4:0], SM architecture [13:5] (9 used/reserved).
  uint32_t to_id() const {
    NVTE_CHECK(sm_arch < 512, "SM architecture no longer fits in the 9 bits to_id() gives it.");
    return static_cast<uint32_t>(use_stochastic_rounding) |
           (static_cast<uint32_t>(use_fast_math) << 1) |
           (static_cast<uint32_t>(row_scaled_nvfp4) << 2) |
           (static_cast<uint32_t>(return_transpose) << 3) | (static_cast<uint32_t>(swizzled) << 4) |
           (sm_arch << 5);
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
        .append(return_transpose ? "1" : "0")
        .append("_")
        .append(swizzled ? "1" : "0");
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result =
        (*entrypoint)(tvm::ffi::String(fn_name), use_stochastic_rounding, use_fast_math,
                      row_scaled_nvfp4, return_transpose, swizzled);
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

  // Data tensors auto-flatten to 2D (DLTensorWrapper's default); the amaxes are rank-1 in the
  // compiled ABI.
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

  // Philox {seed, offset}, used only for SR. The kernel takes the seed as
  // seed_ptr ? *seed_ptr : seed_val and the offset as (offset_ptr ? *offset_ptr : 0) + offset_val,
  // so a captured CUDA graph reads them when it is replayed.
  void *rng_seed_ptr = nullptr;
  void *rng_offset_ptr = nullptr;
  int64_t rng_seed_val = 0;
  int64_t rng_offset_val = 0;
  if (config.use_stochastic_rounding) {
    if (quant_config->philox_state.has_value()) {
      const NVTEPhiloxState &philox_state = *quant_config->philox_state;
      if (philox_state.captured) {
        rng_seed_ptr = philox_state.seed_ptr;
        rng_offset_ptr = philox_state.offset_ptr;
        rng_offset_val = static_cast<int64_t>(philox_state.offset_intragraph);
      } else {
        rng_seed_val = static_cast<int64_t>(philox_state.seed_val);
        rng_offset_val = static_cast<int64_t>(philox_state.offset_val);
      }
    } else {
      NVTE_CHECK(quant_config->rng_state != nullptr,
                 "Stochastic rounding requires an RNG state tensor to be provided.");
      auto *rng_state =
          static_cast<int64_t *>(convertNVTETensor(quant_config->rng_state)->data.dptr);
      rng_seed_ptr = rng_state;
      rng_offset_ptr = rng_state + 1;
    }
  }

  // noop passed as ptr because the null-check must be done at runtime on device
  void *noop_ptr = (noop != nullptr) ? noop->data.dptr : nullptr;
  // noop, the RNG pointers and stream are tvm-ffi opaque "handles"; pass them as void*.
  (*nvfp4_quant_func_opt)(&mX, &mO_row, &mS_row, &mO_col, &mS_col, &mAmaxRow, &mAmaxCol, noop_ptr,
                          rng_seed_ptr, rng_offset_ptr, rng_seed_val, rng_offset_val,
                          static_cast<void *>(stream));
  return true;
}
#endif  // FP4_TYPE_SUPPORTED

// CuTeDSL counterpart of nvfp4::quantize_transpose. Returns false to fall back to the CUDA kernel.
template <bool use_2d_quantization>
bool nvfp4_quantize_transpose_cutedsl(const Tensor &input, const Tensor *noop, Tensor *output,
                                      const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  if (!TVMFFICentral::getInstance().get_cutedsl_backend_enabled()) {
    return false;
  }
  // Currently, only the CuTeDSL counterpart of quantize_transpose_tuned_1D is supported
  if constexpr (use_2d_quantization) {
    return false;
  } else {
    // Like 2D, configs the CuTeDSL kernels do not implement fall back without a report.
    if (input.dtype() != DType::kBFloat16) {
      return false;
    }
    if (output->has_data() && output->scale_inv.dtype != DType::kFloat8E4M3) {
      return false;
    }
    const bool return_transpose = output->has_columnwise_data();
    // The compiled ABI takes the amaxes as tensors, not optionals.
    if (output->amax.dptr == nullptr ||
        (return_transpose && output->columnwise_amax.dptr == nullptr)) {
      return false;
    }
    if (output->row_scaled_nvfp4 && return_transpose) {
      return false;
    }
    // The stochastic rounding kernels use cvt.rs, which only SM 10.x has.
    if (quant_config != nullptr && quant_config->stochastic_rounding &&
        cuda::sm_arch() / 10 != 10) {
      return false;
    }
    const auto [rows, cols] = input.flat_2d_dims();
    // GEMM-swizzled scales only where the rule the framework uses to hand them to the kernel
    // accepts them. A config the rule declines falls back without a report, like those above.
    if (output->with_gemm_swizzled_scales) {
      const bool nvfp4_use_4over6 =
          quant_config != nullptr && quant_config->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled;
      const auto alignment = nvfp4_gemm_swizzled_scales_alignment(
          /*with_rht=*/false, use_2d_quantization, output->row_scaled_nvfp4, nvfp4_use_4over6,
          cuda::sm_arch(), output->scale_inv.dtype, output->amax.dptr != nullptr,
          output->has_data(), quant_config != nullptr && quant_config->stochastic_rounding);
      if (!alignment) {
        return false;
      }
      if (rows % alignment->first != 0 || cols % alignment->second != 0) {
        report_cutedsl_not_chosen(
            "NVFP4 quantization with GEMM-swizzled scales needs rows divisible by ",
            alignment->first, " and columns divisible by ", alignment->second, ".");
        return false;
      }
    }
    if (rows % 32 != 0 || cols % 32 != 0) {
      report_cutedsl_not_chosen("NVFP4 quantization needs rows and columns divisible by 32.");
      return false;
    }

    const bool use_stochastic_rounding =
        quant_config != nullptr ? quant_config->stochastic_rounding : false;
    const bool use_fast_math = quant_config != nullptr ? quant_config->use_fast_math : false;
    const bool row_scaled_nvfp4 = output->row_scaled_nvfp4;
    const bool swizzled = output->with_gemm_swizzled_scales;
    const NVFP4QuantConfig config{
        use_stochastic_rounding, use_fast_math, row_scaled_nvfp4, return_transpose, swizzled,
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
