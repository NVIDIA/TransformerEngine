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

namespace transformer_engine::pytorch {

std::tuple<at::Tensor, at::Tensor, at::Tensor> swizzle_scales_and_pack_ptrs_for_discrete_weights(
    const std::vector<at::Tensor> &data_tensors,
    const std::vector<at::Tensor> &scale_tensors, const std::string &format,
    const c10::Device &device) {
  const size_t num_tensors = data_tensors.size();
  NVTE_CHECK(scale_tensors.size() == num_tensors,
             "Expected data_tensors and scale_tensors to have matching sizes, but got ",
             num_tensors, " and ", scale_tensors.size(), ".");

  // Decode format
  const bool is_mxfp8_rowwise = format == "mxfp8_rowwise";
  const bool is_mxfp8_columnwise = format == "mxfp8_columnwise";
  const bool is_nvfp4 = format == "nvfp4";
  NVTE_CHECK(is_mxfp8_rowwise || is_mxfp8_columnwise || is_nvfp4, "Unsupported format (", format,
             "). Expected one of: mxfp8_rowwise, mxfp8_columnwise, nvfp4.");

  // Trivial case: no tensors. Return three empty tensors.
  if (num_tensors == 0) {
    auto empty_ptrs = at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
    auto empty_scales = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    return {empty_ptrs, empty_ptrs.clone(), std::move(empty_scales)};
  }

  // CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // Tensor format
  const NVTEScalingMode scaling_mode = is_nvfp4 ? NVTE_NVFP4_1D_SCALING : NVTE_MXFP8_1D_SCALING;
  const transformer_engine::DType data_dtype =
      is_nvfp4 ? transformer_engine::DType::kFloat4E2M1 : transformer_engine::DType::kFloat8E4M3;
  const transformer_engine::DType scale_dtype =
      is_nvfp4 ? transformer_engine::DType::kFloat8E4M3 : transformer_engine::DType::kFloat8E8M0;

  // Scale shape
  const NVTEShape scale_shape = convertTorchShape(scale_tensors[0].sizes());
  NVTE_CHECK(scale_shape.ndim == 2,
             "Expected 2D scale tensor, but got shape=", getTensorShape(scale_tensors[0]), ".");
  const size_t scale_numel = scale_shape.data[0] * scale_shape.data[1];
  const size_t scale_dtype_bits = transformer_engine::pytorch::typeToNumBits(scale_dtype);
  const size_t scale_bytes = ceildiv(scale_numel * scale_dtype_bits, 8);

  // Expected data shape.
  // Note: May not match actual data shape since the scales are padded.
  // This is fine since the swizzle kernel does not touch the data.
  NVTEShape data_shape;
  data_shape.ndim = 2;
  if (is_mxfp8_rowwise) {
    data_shape.data[0] = scale_shape.data[0];
    data_shape.data[1] = scale_shape.data[1] * 32;
  } else if (is_mxfp8_columnwise) {
    data_shape.data[0] = scale_shape.data[0] * 32;
    data_shape.data[1] = scale_shape.data[1];
  } else {  // nvfp4
    data_shape.data[0] = scale_shape.data[0];
    data_shape.data[1] = scale_shape.data[1] * 16;
  }

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

  // MXFP8 columnwise tags its data/scale params on the columnwise side; the
  // other two formats (mxfp8_rowwise, nvfp4) tag on the rowwise side.
  const NVTETensorParam data_param = is_mxfp8_columnwise ? kNVTEColumnwiseData : kNVTERowwiseData;
  const NVTETensorParam scale_param =
      is_mxfp8_columnwise ? kNVTEColumnwiseScaleInv : kNVTERowwiseScaleInv;

  for (size_t i = 0; i < num_tensors; ++i) {
    const uint8_t swizzled_flag = 1;
    nvte_set_tensor_param_v2(outputs_nvte[i], kNVTEWithGEMMSwizzledScales, &swizzled_flag,
                             sizeof(swizzled_flag));
    void *in_scale_ptr = scale_tensors[i].data_ptr();
    void *out_scale_ptr = swizzled_scales_dptr + i * swizzled_scales_stride;
    set_param(inputs_nvte[i], data_param, nullptr, data_dtype, data_shape);
    set_param(inputs_nvte[i], scale_param, in_scale_ptr, scale_dtype, scale_shape);
    set_param(outputs_nvte[i], data_param, nullptr, data_dtype, data_shape);
    set_param(outputs_nvte[i], scale_param, out_scale_ptr, scale_dtype, scale_shape);
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

}  // namespace transformer_engine::pytorch
