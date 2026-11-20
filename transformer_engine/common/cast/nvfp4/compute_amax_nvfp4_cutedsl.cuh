/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_COMPUTE_AMAX_NVFP4_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_COMPUTE_AMAX_NVFP4_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <optional>
#include <string>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"

namespace transformer_engine {
namespace cutedsl_backend {

// DLTensorWrapper and TVMFFICentral live in transformer_engine::tvm_ffi_bridge.
using namespace tvm_ffi_bridge;

// Instantiation parameters of the CuTeDSL kernels that compute the global amax of an input
struct NVFP4GlobalAmaxConfig {
  static constexpr const char *kEntrypointName = "get_nvfp4_global_amax_function";

  bool columnwise_amax;  // If the columnwise amax is written too
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  uint32_t to_id() const { return static_cast<uint32_t>(columnwise_amax) | (sm_arch << 1); }

  std::optional<tvm::ffi::Function> get_kernel() const {
    static TVMFFIConfigCache &cache = TVMFFIConfigCache::create();
    return cache.get_or_load(*this);
  }

  std::string to_key() const {
    return "cutedsl_nvfp4_amax_sm" + std::to_string(sm_arch) + (columnwise_amax ? "_1" : "_0");
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result = (*entrypoint)(tvm::ffi::String(fn_name), columnwise_amax);
    return result.try_cast<bool>().value_or(false);
  }
};

// CuTeDSL counterpart of the amax kernels of nvte_compute_amax for an NVFP4 output. Zeroes `amax`
// and, if not null, `columnwise_amax`, then takes the amax of the input into both, unless the
// noop flag is set. Returns false to fall back to the CUDA kernels.
inline bool nvfp4_compute_amax_cutedsl(const Tensor &input, float *amax, float *columnwise_amax,
                                       float *noop, cudaStream_t stream) {
  if (!TVMFFICentral::getInstance().get_cutedsl_backend_enabled()) {
    return false;
  }
  // Like the quantize kernels, the amax kernels take a nonempty BF16 input whose flattened dims
  // are multiples of 32, 16-byte aligned. Other inputs fall back without a report.
  const auto [rows, cols] = input.flat_2d_dims();
  if (input.dtype() != DType::kBFloat16 || input.numel() == 0 || rows % 32 != 0 || cols % 32 != 0 ||
      !is_aligned_ptr(input.data.dptr, 16)) {
    return false;
  }
  std::optional<tvm::ffi::Function> amax_func_opt =
      NVFP4GlobalAmaxConfig{columnwise_amax != nullptr}.get_kernel();
  if (!amax_func_opt.has_value()) {
    return false;
  }

  // The input auto-flattens to 2D (DLTensorWrapper's default); the amaxes are rank-1.
  const int32_t device_index = transformer_engine::cuda::current_device();
  DLTensorWrapper mX(input.data, true, device_index);
  DLTensorWrapper mAmaxRow(SimpleTensor(amax, {1}, DType::kFloat32), false, device_index);
  DLTensorWrapper mAmaxCol;
  if (columnwise_amax != nullptr) {
    mAmaxCol =
        DLTensorWrapper(SimpleTensor(columnwise_amax, {1}, DType::kFloat32), false, device_index);
  }
  // noop and stream are tvm-ffi opaque "handles"; pass them as void*.
  (*amax_func_opt)(&mX, &mAmaxRow, &mAmaxCol, static_cast<void *>(noop),
                   static_cast<void *>(stream));
  return true;
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_COMPUTE_AMAX_NVFP4_CUTEDSL_CUH_
