/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/common.h"
#include "extensions.h"

namespace transformer_engine::pytorch {

at::Tensor load_data_ptrs_on_device(const std::vector<at::Tensor> &tensors,
                                    const c10::Device &device) {
  // Collect data pointers
  std::vector<uint64_t> ptrs_host;
  ptrs_host.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    ptrs_host.push_back(reinterpret_cast<uintptr_t>(tensor.data_ptr()));
  }

  // Allocate device buffer
  auto ptrs_device = at::empty({static_cast<int64_t>(tensors.size())},
                               at::TensorOptions().dtype(at::kLong).device(device));

  // Load pointers on device
  nvte_load_value_on_device(ptrs_host.data(), ptrs_device.data_ptr(),
                            tensors.size() * sizeof(uint64_t), at::cuda::getCurrentCUDAStream());

  return ptrs_device;
}

std::tuple<at::Tensor, std::optional<at::Tensor>> transform_and_load_data_ptrs_on_device(
    const std::string &transform_type, const std::vector<at::Tensor> &tensors,
    const c10::Device &device) {
  const size_t num_tensors = tensors.size();

  // Trivial cases
  if (transform_type.empty()) {
    // No transform, just load pointers on device
    return {load_data_ptrs_on_device(tensors, device), std::nullopt};
  }
  if (num_tensors == 0) {
    // No input tensors, return tensor with no elements
    return {at::empty({int64_t{0}}, at::TensorOptions().dtype(at::kLong).device(device)),
            std::nullopt};
  }

  // CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // Swizzle scales for GEMM, with uniform tensor sizes
  const bool uniform_mxfp8_rowwise_swizzle = transform_type == "uniform_mxfp8_rowwise_swizzle";
  const bool uniform_mxfp8_colwise_swizzle = transform_type == "uniform_mxfp8_columnwise_swizzle";
  const bool uniform_nvfp4_swizzle = transform_type == "uniform_nvfp4_swizzle";
  if (uniform_mxfp8_rowwise_swizzle || uniform_mxfp8_colwise_swizzle || uniform_nvfp4_swizzle) {
    // Tensor format
    NVTEScalingMode scaling_mode = NVTE_INVALID_SCALING;
    if (uniform_mxfp8_rowwise_swizzle || uniform_mxfp8_colwise_swizzle) {
      scaling_mode = NVTE_MXFP8_1D_SCALING;
    } else if (uniform_nvfp4_swizzle) {
      scaling_mode = NVTE_NVFP4_1D_SCALING;
    }

    // Data types
    transformer_engine::DType data_dtype, scale_dtype;
    switch (scaling_mode) {
      case NVTE_MXFP8_1D_SCALING:
        data_dtype = transformer_engine::DType::kFloat8E4M3;
        scale_dtype = transformer_engine::DType::kFloat8E8M0;
        break;
      case NVTE_NVFP4_1D_SCALING:
        data_dtype = transformer_engine::DType::kFloat4E2M1;
        scale_dtype = transformer_engine::DType::kFloat8E4M3;
        break;
      default:
        NVTE_ERROR("Unsupported case.");
    }

    // Scale shape
    const NVTEShape scale_shape = convertTorchShape(tensors[0].sizes());
    NVTE_CHECK(scale_shape.ndim == 2,
               "Expected 2D scale tensor, but got shape=", getTensorShape(tensors[0]), ".");
    const size_t scale_numel = scale_shape.data[0] * scale_shape.data[1];
    const size_t scale_dtype_bits = transformer_engine::pytorch::typeToNumBits(scale_dtype);
    const size_t scale_bytes = ceildiv(scale_numel * scale_dtype_bits, 8);

    // Expected data shape
    // Note: May not match actual data shape since the scales are padded.
    // This is fine since we're not actually touching the data.
    NVTEShape data_shape;
    data_shape.ndim = 2;
    if (uniform_mxfp8_rowwise_swizzle) {
      data_shape.data[0] = scale_shape.data[0];
      data_shape.data[1] = scale_shape.data[1] * 32;
    } else if (uniform_mxfp8_colwise_swizzle) {
      data_shape.data[0] = scale_shape.data[0] * 32;
      data_shape.data[1] = scale_shape.data[1];
    } else if (uniform_nvfp4_swizzle) {
      data_shape.data[0] = scale_shape.data[0];
      data_shape.data[1] = scale_shape.data[1] * 16;
    } else {
      NVTE_ERROR("Unsupported case.");
    }

    // Allocate single buffer for swizzled scales.
    // Uses a uniform stride since all tensors share the same scale shape.
    const size_t swizzled_scales_stride = roundup(scale_bytes, 16);  // Align to 16 bytes
    auto swizzled_scales = at::empty({static_cast<int64_t>(swizzled_scales_stride * num_tensors)},
                                     at::TensorOptions().dtype(at::kByte).device(device));
    uint8_t *swizzled_scales_dptr = reinterpret_cast<uint8_t *>(swizzled_scales.data_ptr());

    // Allocate input/output NVTETensors as a single batch. The first
    // num_tensors entries are inputs; the next num_tensors are outputs.
    std::vector<NVTETensor> nvte_tensors(2 * num_tensors);
    nvte_create_tensors(scaling_mode, nvte_tensors.data(), nvte_tensors.size());
    struct DestroyGuard {
      NVTETensor *data;
      size_t n;
      ~DestroyGuard() { nvte_destroy_tensors(data, n); }
    } destroy_guard{nvte_tensors.data(), nvte_tensors.size()};
    NVTETensor *inputs_nvte = nvte_tensors.data();
    NVTETensor *outputs_nvte = nvte_tensors.data() + num_tensors;

    auto set_param = [](NVTETensor t, NVTETensorParam param, void *dptr,
                        transformer_engine::DType dtype, const NVTEShape &shape) {
      NVTEBasicTensor data{dptr, static_cast<NVTEDType>(dtype), shape};
      nvte_set_tensor_param_v2(t, param, &data, sizeof(data));
    };

    for (size_t i = 0; i < num_tensors; ++i) {
      const uint8_t swizzled_flag = 1;
      nvte_set_tensor_param_v2(outputs_nvte[i], kNVTEWithGEMMSwizzledScales, &swizzled_flag,
                               sizeof(swizzled_flag));
      void *in_scale_ptr = tensors[i].data_ptr();
      void *out_scale_ptr = swizzled_scales_dptr + i * swizzled_scales_stride;
      if (uniform_mxfp8_rowwise_swizzle || uniform_nvfp4_swizzle) {
        set_param(inputs_nvte[i], kNVTERowwiseData, nullptr, data_dtype, data_shape);
        set_param(inputs_nvte[i], kNVTERowwiseScaleInv, in_scale_ptr, scale_dtype, scale_shape);
        set_param(outputs_nvte[i], kNVTERowwiseData, nullptr, data_dtype, data_shape);
        set_param(outputs_nvte[i], kNVTERowwiseScaleInv, out_scale_ptr, scale_dtype, scale_shape);
      } else if (uniform_mxfp8_colwise_swizzle) {
        set_param(inputs_nvte[i], kNVTEColumnwiseData, nullptr, data_dtype, data_shape);
        set_param(inputs_nvte[i], kNVTEColumnwiseScaleInv, in_scale_ptr, scale_dtype, scale_shape);
        set_param(outputs_nvte[i], kNVTEColumnwiseData, nullptr, data_dtype, data_shape);
        set_param(outputs_nvte[i], kNVTEColumnwiseScaleInv, out_scale_ptr, scale_dtype, scale_shape);
      }
    }

    // Launch kernel
    nvte_multi_tensor_swizzle_scaling_factors(inputs_nvte, outputs_nvte, num_tensors, stream);

    // Collect data pointers
    std::vector<uint64_t> ptrs_host;
    ptrs_host.reserve(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      ptrs_host.push_back(
          reinterpret_cast<uintptr_t>(swizzled_scales_dptr + i * swizzled_scales_stride));
    }

    // Load pointers on device
    auto ptrs_device = at::empty({static_cast<int64_t>(num_tensors)},
                                 at::TensorOptions().dtype(at::kLong).device(device));
    nvte_load_value_on_device(ptrs_host.data(), ptrs_device.data_ptr(),
                              num_tensors * sizeof(uint64_t), stream);

    return {std::move(ptrs_device), std::move(swizzled_scales)};
  }

  // Unsupported transform
  NVTE_ERROR("Unsupported transform type (", transform_type, ")");
}

}  // namespace transformer_engine::pytorch
