/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_DEQUANTIZE_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_DEQUANTIZE_MXFP8_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"
#include "../../util/cuda_runtime.h"
#include "../../utils.cuh"                   // ShapeRepresentation
#include "group_dequantize_mxfp8.cuh"        // g_tensor_maps_input, g_tensor_maps_output
#include "group_quantize_mxfp8_cutedsl.cuh"  // shape_rep_to_str

namespace transformer_engine {
namespace cutedsl_backend {

using namespace tvm_ffi_bridge;

struct MXFP8GroupDequantConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_group_dequantization_function";

  DType fp8_dtype;                // The fp8 input format
  DType out_dtype;                // The high-precision output format
  bool rowwise;                   // If dequantized from the rowwise (else columnwise) data
  ShapeRepresentation shape_rep;  // How the member shapes vary across the group
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  // Bit layout: fp8_dtype [3:0], out_dtype [7:4], rowwise [8], shape_rep [10:9],
  // SM architecture [19:11].
  uint32_t to_id() const {
    static_assert(static_cast<uint32_t>(DType::kNumTypes) <= 16,
                  "DType no longer fits in the 4 bits to_id() gives it.");
    static_assert(ShapeRepresentation::VARYING_BOTH_DIMS < 4,
                  "ShapeRepresentation no longer fits in the 2 bits to_id() gives it.");
    NVTE_CHECK(sm_arch < 512, "SM architecture no longer fits in the 9 bits to_id() gives it.");
    return static_cast<uint32_t>(fp8_dtype) | (static_cast<uint32_t>(out_dtype) << 4) |
           (static_cast<uint32_t>(rowwise) << 8) | (static_cast<uint32_t>(shape_rep) << 9) |
           (sm_arch << 11);
  }

  std::optional<tvm::ffi::Function> get_kernel() const {
    static TVMFFIConfigCache &cache = TVMFFIConfigCache::create();
    return cache.get_or_load(*this);
  }

  std::string to_key() const {
    std::string key;
    key.reserve(64);  // longest: cutedsl_group_mxfp8_dequant_smXXX_Float8E4M3_BFloat16_1_vfd
    key.append("cutedsl_group_mxfp8_dequant_sm")
        .append(std::to_string(sm_arch))
        .append("_")
        .append(to_string(fp8_dtype))
        .append("_")
        .append(to_string(out_dtype))
        .append("_")
        .append(rowwise ? "1" : "0")
        .append("_")
        .append(shape_rep_to_str(shape_rep));
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result =
        (*entrypoint)(tvm::ffi::String(fn_name), tvm::ffi::String(to_string(fp8_dtype)),
                      tvm::ffi::String(to_string(out_dtype)), rowwise,
                      tvm::ffi::String(shape_rep_to_str(shape_rep)));
    return result.try_cast<bool>().value_or(false);
  }
};

// Device addresses of the CUDA kernel's per-member input and output descriptor arrays, cached
// once per device. Internal linkage keeps this cache paired with the translation unit's symbols.
static std::pair<void *, void *> group_dequantize_tensor_map_ptrs() {
  using dispatch::mxfp8::group_dequantize_kernel::g_tensor_maps_input;
  using dispatch::mxfp8::group_dequantize_kernel::g_tensor_maps_output;
  static std::vector<std::pair<void *, void *>> cache(cuda::num_devices(), {nullptr, nullptr});
  static std::vector<std::once_flag> flags(cuda::num_devices());
  const int device_id = cuda::current_device();
  NVTE_CHECK(0 <= device_id && device_id < cuda::num_devices(), "invalid CUDA device ID");
  std::call_once(flags[device_id], [&]() {
    NVTE_CHECK_CUDA(cudaGetSymbolAddress(&cache[device_id].first, g_tensor_maps_input));
    NVTE_CHECK_CUDA(cudaGetSymbolAddress(&cache[device_id].second, g_tensor_maps_output));
  });
  return cache[device_id];
}

// Signature mirrors mxfp8::group_dequantize. Returns false to fall back to the CUDA kernel,
// which also raises the errors for invalid inputs.
inline bool mxfp8_group_dequantize_cutedsl(const GroupedTensor &input, GroupedTensor *output,
                                           cudaStream_t stream) {
  if (!TVMFFICentral::getInstance().get_cutedsl_backend_enabled()) {
    return false;
  }
  const bool rowwise = input.has_data();
  const bool colwise = input.has_columnwise_data();
  const size_t num_tensors = input.num_tensors;
  const size_t first_logical_dim = input.logical_shape.data[0];
  const size_t last_logical_dim = input.logical_shape.data[1];
  if (rowwise == colwise || input.with_gemm_swizzled_scales || !is_fp8_dtype(input.dtype()) ||
      is_fp8_dtype(output->dtype()) || is_fp4_dtype(output->dtype()) ||
      num_tensors != output->num_tensors || first_logical_dim * last_logical_dim == 0) {
    return false;
  }

  // Mirrors the shape-representation selection in mxfp8::group_dequantize.
  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (input.all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (input.all_same_first_dim()) {
    shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
  } else if (input.all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else if (input.varying_both_dims()) {
    shape_rep = ShapeRepresentation::VARYING_BOTH_DIMS;
  }
  const bool is_single_tensor = shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                                shape_rep == ShapeRepresentation::VARYING_FIRST_DIM;
  if (!is_single_tensor &&
      (num_tensors == 0 ||
       num_tensors > dispatch::mxfp8::group_dequantize_kernel::MAX_SUPPORTED_TENSOR_DESCRIPTORS ||
       last_logical_dim % 128 != 0 || !input.tensor_offsets.has_data())) {
    return false;
  }

  const SimpleTensor &data = rowwise ? input.data : input.columnwise_data;
  const SimpleTensor &scales = rowwise ? input.scale_inv : input.columnwise_scale_inv;
  if (last_logical_dim % 16 != 0) {
    report_cutedsl_not_chosen("the last logical dimension is not divisible by 16.");
    return false;
  }
  if (!is_aligned_ptr(data.dptr, 16) || !is_aligned_ptr(output->data.dptr, 16) ||
      !is_aligned_ptr(scales.dptr, 4)) {
    report_cutedsl_not_chosen("the input, output or scale pointer is misaligned.");
    return false;
  }

  checkCuDriverContext(stream);
  const MXFP8GroupDequantConfig config{/*fp8_dtype=*/input.dtype(),
                                       /*out_dtype=*/output->dtype(),
                                       /*rowwise=*/rowwise,
                                       /*shape_rep=*/shape_rep};
  std::optional<tvm::ffi::Function> func = config.get_kernel();
  if (!func.has_value()) {
    return false;
  }

  const int32_t device_index = cuda::current_device();
  // The group's payload is stored flat; the kernel wants it as the logical 2D view.
  const std::vector<size_t> logical_shape{first_logical_dim, last_logical_dim};
  DLTensorWrapper mX(make_basic_tensor(data.dptr, data.dtype, logical_shape), true, device_index);
  DLTensorWrapper mO(make_basic_tensor(output->data.dptr, output->data.dtype, logical_shape), true,
                     device_index);
  // Per-member scale bases and strides are derived from the member shapes.
  DLTensorWrapper mS(make_basic_tensor(scales.dptr, scales.dtype, {scales.numel()}), false,
                     device_index);

  // Optional inputs: a wrapper over a null buffer packs as TVM-FFI None.
  DLTensorWrapper mOffsets, mFirstDims, mLastDims, mInputTensormaps, mOutputTensormaps;
  if (!is_single_tensor) {
    mOffsets = DLTensorWrapper(
        make_basic_tensor(input.tensor_offsets.dptr, DType::kInt64, {num_tensors + 1}), false,
        device_index);
    if (shape_rep == ShapeRepresentation::VARYING_BOTH_DIMS) {
      mFirstDims =
          DLTensorWrapper(make_basic_tensor(input.first_dims.dptr, DType::kInt64, {num_tensors}),
                          false, device_index);
    }
    mLastDims = DLTensorWrapper(
        make_basic_tensor(input.last_dims.dptr, DType::kInt64, {num_tensors}), false, device_index);

    // Reuse the CUDA kernel's descriptor arrays, viewed as flat int64.
    constexpr size_t kTensormapsInt64 =
        dispatch::mxfp8::group_dequantize_kernel::MAX_SUPPORTED_TENSOR_DESCRIPTORS *
        sizeof(CUtensorMap) / sizeof(int64_t);
    const auto [input_maps, output_maps] = group_dequantize_tensor_map_ptrs();
    mInputTensormaps = DLTensorWrapper(
        make_basic_tensor(input_maps, DType::kInt64, {kTensormapsInt64}), false, device_index);
    mOutputTensormaps = DLTensorWrapper(
        make_basic_tensor(output_maps, DType::kInt64, {kTensormapsInt64}), false, device_index);
  }

  (*func)(&mX, &mS, &mO, &mOffsets, &mFirstDims, &mLastDims, &mInputTensormaps, &mOutputTensormaps,
          static_cast<int32_t>(num_tensors), static_cast<void *>(stream));
  return true;
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_DEQUANTIZE_MXFP8_CUTEDSL_CUH_
