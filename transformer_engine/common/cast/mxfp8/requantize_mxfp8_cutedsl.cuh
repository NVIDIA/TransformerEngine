/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_REQUANTIZE_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_REQUANTIZE_MXFP8_CUTEDSL_CUH_

#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"

namespace transformer_engine {
namespace cutedsl_backend {

struct MXFP8RequantConfig {
  DType dtype;
  size_t rows, hidden, groups, uniform_rows;
  bool swizzled, rowwise_output;
  int sm_arch = cuda::sm_arch();
  int sm_count = cuda::sm_count();

  std::string to_key() const {
    return "cutedsl_mxfp8_requant_" + std::string(to_string(dtype)) + "_" + std::to_string(rows) +
           "_" + std::to_string(hidden) + "_" + std::to_string(groups) + "_" +
           std::to_string(uniform_rows) + "_" + std::to_string(swizzled) + "_" +
           std::to_string(rowwise_output) + "_" + std::to_string(sm_arch) + "_" +
           std::to_string(sm_count);
  }

  bool retrieve_func_from_python(const std::string &name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal("get_mxfp8_requantization_function");
    if (!entrypoint) return false;
    return (*entrypoint)(tvm::ffi::String(name), tvm::ffi::String(to_string(dtype)),
                         static_cast<int64_t>(rows), static_cast<int64_t>(hidden),
                         static_cast<int64_t>(groups), static_cast<int64_t>(uniform_rows), swizzled,
                         rowwise_output, sm_count)
        .try_cast<bool>()
        .value_or(false);
  }

  std::optional<tvm_ffi_bridge::TVMFFIKernel> get_kernel() const {
    using namespace tvm_ffi_bridge;
    // Shape-specialized keys do not fit the quantization cache's 32-bit ID.
    // Preserve cached TVM handles until process exit, as TVMFFIConfigCache does.
    struct Cache {
      std::shared_mutex mutex;
      std::unordered_map<std::string, std::optional<TVMFFIKernel>> kernels;
    };
    static auto *cache = new Cache();
    auto &central = TVMFFICentral::getInstance();
    if (!central.get_cutedsl_backend_enabled()) return std::nullopt;
    const auto key = to_key();
    {
      std::shared_lock<std::shared_mutex> lock(cache->mutex);
      auto it = cache->kernels.find(key);
      if (it != cache->kernels.end()) return it->second;
    }
    std::unique_lock<std::shared_mutex> lock(cache->mutex);
    auto it = cache->kernels.find(key);
    if (it != cache->kernels.end()) return it->second;
    std::optional<TVMFFIKernel> kernel;
    if (auto fn = central.load_tvm_ffi_function(*this)) {
      kernel.emplace(make_tvm_ffi_kernel(std::move(*fn)));
    }
    cache->kernels.emplace(key, kernel);
    return kernel;
  }
};

inline bool mxfp8_requantize_cutedsl(const GroupedTensor &input, GroupedTensor *output,
                                     bool use_fast_math, cudaStream_t stream) {
  using namespace tvm_ffi_bridge;
  if (!TVMFFICentral::getInstance().get_cutedsl_backend_enabled() || !use_fast_math ||
      !input.all_same_last_dim())
    return false;
  const size_t rows = input.logical_shape.data[0];
  const size_t hidden = input.logical_shape.data[1];
  const size_t groups = input.num_tensors;
  if (rows == 0 || hidden == 0 || rows % 128 || hidden % 128 || (hidden >= 512 && hidden % 512))
    return false;
  // TMA's compact scale matrix requires a 16-byte row stride for wide tensors.
  // The narrow specializations copy full-width compact scales as 1D bulk data.
  if (input.all_same_shape() && (rows % groups || (rows / groups) % 128)) return false;
  if (!input.all_same_shape() &&
      (input.tensor_offsets.dptr == nullptr || input.tensor_offsets.numel() < groups + 1)) {
    return false;
  }
  const bool rowwise = output->with_gemm_swizzled_scales && output->scale_inv.dptr != nullptr;
  if (!is_aligned_ptr(input.scale_inv.dptr, 16) ||
      !is_aligned_ptr(output->columnwise_scale_inv.dptr, 16) ||
      (rowwise && !is_aligned_ptr(output->scale_inv.dptr, 16)))
    return false;
  const MXFP8RequantConfig config{input.data.dtype,
                                  rows,
                                  hidden,
                                  groups,
                                  input.all_same_shape() ? rows / groups : 0,
                                  output->with_gemm_swizzled_scales,
                                  rowwise};
  auto kernel = config.get_kernel();
  if (!kernel) return false;

  const int device = cuda::current_device();
  const SimpleTensor src(input.data.dptr, {rows, hidden}, input.data.dtype);
  const SimpleTensor sf(input.scale_inv.dptr, {rows, hidden / 32}, DType::kFloat8E8M0);
  const SimpleTensor dst(output->columnwise_data.dptr, {rows, hidden}, DType::kFloat8E4M3);
  const SimpleTensor col_sf(output->columnwise_scale_inv.dptr, {rows * hidden / 32},
                            DType::kFloat8E8M0);
  DLTensorWrapper mSrc(src, false, device), mSf(sf, false, device), mDst(dst, false, device),
      mColSf(col_sf, false, device), mOffsets, mRowSf;
  if (!input.all_same_shape()) {
    const SimpleTensor offsets(input.tensor_offsets.dptr, {groups + 1}, DType::kInt64);
    mOffsets = DLTensorWrapper(offsets, false, device);
  }
  if (rowwise) {
    const SimpleTensor row_sf(output->scale_inv.dptr, {rows * hidden / 32}, DType::kFloat8E8M0);
    mRowSf = DLTensorWrapper(row_sf, false, device);
  }
  (*kernel)(&mSrc, &mSf, &mOffsets, &mDst, &mRowSf, &mColSf, static_cast<void *>(stream));
  return true;
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_REQUANTIZE_MXFP8_CUTEDSL_CUH_
