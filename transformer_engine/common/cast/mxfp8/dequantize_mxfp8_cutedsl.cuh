/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_DEQUANTIZE_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_DEQUANTIZE_MXFP8_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <optional>
#include <string>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"

namespace transformer_engine {
namespace cutedsl_backend {

using namespace tvm_ffi_bridge;

struct MXFP8DequantConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_dequantization_function";

  DType fp8_dtype;  // The fp8 input format
  DType out_dtype;  // The high-precision output format
  bool rowwise;     // If rowwise data and scales are passed
  bool colwise;     // If columnwise data and scales are passed
  bool swizzled;    // If the scales are in the GEMM-swizzled layout
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  // Bit layout: fp8_dtype [3:0], out_dtype [7:4], flags [10:8], SM architecture [19:11].
  uint32_t to_id() const {
    static_assert(static_cast<uint32_t>(DType::kNumTypes) <= 16,
                  "DType no longer fits in the 4 bits to_id() gives it.");
    NVTE_CHECK(sm_arch < 512, "SM architecture no longer fits in the 9 bits to_id() gives it.");
    return static_cast<uint32_t>(fp8_dtype) | (static_cast<uint32_t>(out_dtype) << 4) |
           (static_cast<uint32_t>(rowwise) << 8) | (static_cast<uint32_t>(colwise) << 9) |
           (static_cast<uint32_t>(swizzled) << 10) | (sm_arch << 11);
  }

  std::optional<tvm::ffi::Function> get_kernel() const {
    static TVMFFIConfigCache &cache = TVMFFIConfigCache::create();
    return cache.get_or_load(*this);
  }

  std::string to_key() const {
    std::string key;
    key.reserve(64);  // longest: cutedsl_mxfp8_dequant_smXXX_Float8E4M3_BFloat16_1_1_1
    key.append("cutedsl_mxfp8_dequant_sm")
        .append(std::to_string(sm_arch))
        .append("_")
        .append(to_string(fp8_dtype))
        .append("_")
        .append(to_string(out_dtype))
        .append("_")
        .append(rowwise ? "1" : "0")
        .append("_")
        .append(colwise ? "1" : "0")
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
        (*entrypoint)(tvm::ffi::String(fn_name), tvm::ffi::String(to_string(fp8_dtype)),
                      tvm::ffi::String(to_string(out_dtype)), rowwise, colwise, swizzled);
    return result.try_cast<bool>().value_or(false);
  }
};

// Signature mirrors mxfp8::dequantize. Returns false to fall back to the CUDA kernel.
inline bool mxfp8_dequantize_cutedsl(const Tensor &input, Tensor *output, cudaStream_t stream) {
  if (!tvm_ffi_bridge::TVMFFICentral::getInstance().get_cutedsl_backend_enabled()) {
    return false;
  }
  // Pass every data copy present; the Python side chooses which direction to dequantize.
  const bool rowwise = input.has_data();
  const bool colwise = input.has_columnwise_data();

  // Sanity checks, mirroring mxfp8::dequantize in dequantize_mxfp8.cuh
  NVTE_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions.");
  NVTE_CHECK(rowwise || colwise, "Cannot dequantize tensor without data.");
  if (rowwise) {
    NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  }
  if (colwise) {
    NVTE_CHECK(is_fp8_dtype(input.columnwise_data.dtype), "Input must have FP8 type.");
  }
  NVTE_CHECK(!(rowwise && colwise) || input.data.dtype == input.columnwise_data.dtype,
             "Rowwise and columnwise data must have the same FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->shape() == input.shape(), "Input and output shapes need to match.");

  if (input.flat_last_dim() % 16 != 0) {
    report_cutedsl_not_chosen("the last dimension of input is not a multiple of 16.");
    return false;
  }
  const bool misaligned = !is_aligned_ptr(output->data.dptr, 16) ||
                          (rowwise && (!is_aligned_ptr(input.data.dptr, 16) ||
                                       !is_aligned_ptr(input.scale_inv.dptr, 4))) ||
                          (colwise && (!is_aligned_ptr(input.columnwise_data.dptr, 16) ||
                                       !is_aligned_ptr(input.columnwise_scale_inv.dptr, 4)));
  if (misaligned) {
    report_cutedsl_not_chosen("the input, output or scale pointer is misaligned.");
    return false;
  }

  checkCuDriverContext(stream);
  const MXFP8DequantConfig config{/*fp8_dtype=*/input.dtype(),
                                  /*out_dtype=*/output->data.dtype,
                                  /*rowwise=*/rowwise,
                                  /*colwise=*/colwise,
                                  /*swizzled=*/input.with_gemm_swizzled_scales};
  std::optional<tvm::ffi::Function> func = config.get_kernel();
  if (!func.has_value()) {
    return false;
  }

  const int32_t device_index = transformer_engine::cuda::current_device();
  tvm_ffi_bridge::DLTensorWrapper mX_row, mS_row, mX_col, mS_col;
  if (rowwise) {
    mX_row = tvm_ffi_bridge::DLTensorWrapper(input.data, true, device_index);
    mS_row = tvm_ffi_bridge::DLTensorWrapper(input.scale_inv, true, device_index);
  }
  if (colwise) {
    mX_col = tvm_ffi_bridge::DLTensorWrapper(input.columnwise_data, true, device_index);
    mS_col = tvm_ffi_bridge::DLTensorWrapper(input.columnwise_scale_inv, true, device_index);
  }
  tvm_ffi_bridge::DLTensorWrapper mO(output->data, true, device_index);
  (*func)(&mX_row, &mS_row, &mX_col, &mS_col, &mO, static_cast<void *>(stream));
  return true;
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_DEQUANTIZE_MXFP8_CUTEDSL_CUH_
