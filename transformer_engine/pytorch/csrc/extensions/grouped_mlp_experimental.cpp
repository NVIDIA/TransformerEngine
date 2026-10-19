/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Experimental helpers for the fused grouped MLP.

#include <ATen/cuda/CUDAContext.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/common.h"
#include "extensions.h"

namespace transformer_engine {
namespace pytorch {
namespace grouped_mlp_experimental {

std::tuple<at::Tensor, at::Tensor, at::Tensor> swizzle_scales_and_pack_ptrs_for_discrete_weights(
    const std::vector<at::Tensor> &data_tensors, const std::vector<at::Tensor> &scale_tensors,
    const std::string &swizzle_type_str, const c10::Device &device) {
  const size_t num_tensors = data_tensors.size();
  NVTE_CHECK(scale_tensors.size() == num_tensors,
             "Expected data_tensors and scale_tensors to have matching sizes, but got ",
             num_tensors, " and ", scale_tensors.size(), ".");

  // Parse swizzle type
  enum class SwizzleType { Invalid, MXFP8Rowwise, MXFP8Columnwise, NVFP4 };
  SwizzleType swizzle_type = SwizzleType::Invalid;
  if (swizzle_type_str == "mxfp8_rowwise") {
    swizzle_type = SwizzleType::MXFP8Rowwise;
  } else if (swizzle_type_str == "mxfp8_columnwise") {
    swizzle_type = SwizzleType::MXFP8Columnwise;
  } else if (swizzle_type_str == "nvfp4") {
    swizzle_type = SwizzleType::NVFP4;
  } else {
    NVTE_ERROR("Unsupported swizzle type (", swizzle_type_str,
               "). Expected one of: mxfp8_rowwise, mxfp8_columnwise, nvfp4.");
  }

  // Trivial case: no tensors. Return empty tensors.
  if (num_tensors == 0) {
    auto empty_ptrs = at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
    auto empty_scales = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    return {empty_ptrs, empty_ptrs.clone(), std::move(empty_scales)};
  }

  // CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // Tensor properties
  NVTEScalingMode scaling_mode;
  transformer_engine::DType data_dtype, scale_dtype;
  NVTETensorParam data_param_name, scale_param_name;
  switch (swizzle_type) {
    case SwizzleType::MXFP8Rowwise:
    case SwizzleType::MXFP8Columnwise:
      scaling_mode = NVTE_MXFP8_1D_SCALING;
      data_dtype = transformer_engine::DType::kFloat8E4M3;
      scale_dtype = transformer_engine::DType::kFloat8E8M0;
      if (swizzle_type == SwizzleType::MXFP8Rowwise) {
        data_param_name = kNVTERowwiseData;
        scale_param_name = kNVTERowwiseScaleInv;
      } else {
        data_param_name = kNVTEColumnwiseData;
        scale_param_name = kNVTEColumnwiseScaleInv;
      }
      break;
    case SwizzleType::NVFP4:
      scaling_mode = NVTE_NVFP4_1D_SCALING;
      data_dtype = transformer_engine::DType::kFloat4E2M1;
      scale_dtype = transformer_engine::DType::kFloat8E4M3;
      data_param_name = kNVTERowwiseData;
      scale_param_name = kNVTERowwiseScaleInv;
      break;
    default:
      NVTE_ERROR("Unsupported swizzle type (", static_cast<int>(swizzle_type), ").");
  }

  // Data shape
  NVTEShape data_shape = convertTorchShape(data_tensors[0].sizes());
  if (swizzle_type == SwizzleType::NVFP4) {
    // NVFP4 packs two 4-bit values per byte
    NVTE_CHECK(data_shape.ndim > 0, "Invalid shape for NVFP4 data tensor (",
               getTensorShape(data_tensors[0]), ").");
    data_shape.data[data_shape.ndim - 1] *= 2;
  }

  // Scale shape
  const NVTEShape scale_shape = convertTorchShape(scale_tensors[0].sizes());
  NVTE_CHECK(scale_shape.ndim == 2,
             "Expected 2D scale tensor, but got shape=", getTensorShape(scale_tensors[0]), ".");
  const size_t scale_numel = scale_shape.data[0] * scale_shape.data[1];
  const size_t scale_dtype_bits = transformer_engine::pytorch::typeToNumBits(scale_dtype);
  const size_t scale_bytes = ceildiv(scale_numel * scale_dtype_bits, 8);

  // Allocate single buffer for swizzled scales. Uses a uniform stride since
  // all tensors share the same scale shape.
  const size_t swizzled_scales_stride = roundup(scale_bytes, 16);  // Align to 16 bytes
  auto swizzled_scales = at::empty({static_cast<int64_t>(swizzled_scales_stride * num_tensors)},
                                   at::TensorOptions().dtype(at::kByte).device(device));
  uint8_t *swizzled_scales_dptr = reinterpret_cast<uint8_t *>(swizzled_scales.data_ptr());

  // Allocate input/output NVTETensors as a single batch. The first
  // num_tensors entries are inputs; the next num_tensors are outputs.
  MultiTensorWrapper nvte_tensors(2 * num_tensors, scaling_mode);
  NVTETensor *inputs_nvte = nvte_tensors.data();
  NVTETensor *outputs_nvte = nvte_tensors.data() + num_tensors;

  auto set_param = [](NVTETensor t, NVTETensorParam param, void *dptr,
                      transformer_engine::DType dtype, const NVTEShape &shape) {
    NVTEBasicTensor data{dptr, static_cast<NVTEDType>(dtype), shape};
    nvte_set_tensor_param_v2(t, param, &data, sizeof(data));
  };

  // Configure NVTETensors
  for (size_t i = 0; i < num_tensors; ++i) {
    const uint8_t swizzled_flag = 1;
    nvte_set_tensor_param_v2(outputs_nvte[i], kNVTEWithGEMMSwizzledScales, &swizzled_flag,
                             sizeof(swizzled_flag));
    void *in_scale_ptr = scale_tensors[i].data_ptr();
    void *out_scale_ptr = swizzled_scales_dptr + i * swizzled_scales_stride;
    set_param(inputs_nvte[i], data_param_name, nullptr, data_dtype, data_shape);
    set_param(inputs_nvte[i], scale_param_name, in_scale_ptr, scale_dtype, scale_shape);
    set_param(outputs_nvte[i], data_param_name, nullptr, data_dtype, data_shape);
    set_param(outputs_nvte[i], scale_param_name, out_scale_ptr, scale_dtype, scale_shape);
  }

  // Launch swizzle kernel
  nvte_multi_tensor_swizzle_scaling_factors(inputs_nvte, outputs_nvte, num_tensors, stream);

  // Pack data pointers (first half) and swizzled scale pointers (second half)
  // into a single host buffer and copy to device with one kernel launch.
  std::vector<uint64_t> packed_ptrs_host(2 * num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    packed_ptrs_host[i] = reinterpret_cast<uintptr_t>(data_tensors[i].data_ptr());
    packed_ptrs_host[num_tensors + i] =
        reinterpret_cast<uintptr_t>(swizzled_scales_dptr + i * swizzled_scales_stride);
  }
  auto packed_ptrs_device = at::empty({static_cast<int64_t>(2 * num_tensors)},
                                      at::TensorOptions().dtype(at::kLong).device(device));
  nvte_copy_host_to_device_via_kernel(packed_ptrs_host.data(), packed_ptrs_device.data_ptr(),
                                      2 * num_tensors * sizeof(uint64_t), stream);

  // Return the two pointer arrays as views into the packed device buffer.
  auto data_ptrs = packed_ptrs_device.narrow(0, 0, static_cast<int64_t>(num_tensors));
  auto scale_ptrs = packed_ptrs_device.narrow(0, static_cast<int64_t>(num_tensors),
                                              static_cast<int64_t>(num_tensors));
  return {std::move(data_ptrs), std::move(scale_ptrs), std::move(swizzled_scales)};
}

}  // namespace grouped_mlp_experimental
}  // namespace pytorch
}  // namespace transformer_engine
