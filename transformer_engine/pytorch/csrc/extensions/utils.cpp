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
  auto ptrs_device = at::empty({tensors.size()},
                               at::TensorOptions().dtype(at::kLong).device(device));

  // Load pointers on device
  nvte_store_value_on_device(ptrs_host.data(), ptrs_device.data_ptr(),
                             tensors.size() * sizeof(uint64_t),
                             at::cuda::getCurrentCUDAStream());

  return ptrs_device;
}

std::tuple<at::Tensor, std::optional<at::Tensor>> transform_and_load_data_ptrs_on_device(const std::string &transform_type,
                                                                                         const std::vector<at::Tensor> &tensors,
                                                                                         const c10::Device &device) {
  const size_t num_tensors = tensors.size();

  // Trivial cases
  if (transform_type.empty()) {
    // No transform, just load pointers on device
    return {load_data_ptrs_on_device(tensors, device), std::nullopt};
  }
  if (num_tensors == 0) {
    // No input tensors, return tensor with no elements
    return {
      at::empty(std::vector<int64_t>{0}, at::TensorOptions().dtype(at::kLong).device(device)),
      std::nullopt};
  }

  // CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // Swizzle scales for GEMM, with uniform tensor sizes
  const bool uniform_mxfp8_rowwise_swizzle = transform_type == "uniform_mxfp8_rowwise_swizzle";
  const bool uniform_mxfp8_colwise_swizzle = transform_type == "uniform_mxfp8_columnwise_swizzle";
  const bool uniform_nvfp4_swizzle = transform_type == "uniform_nvfp4_swizzle";
  if (uniform_mxfp8_rowwise_swizzle
      || uniform_mxfp8_colwise_swizzle
      || uniform_nvfp4_swizzle) {
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

    // Allocate single buffer for swizzled scales
    const size_t swizzled_scales_stride = roundup(scale_bytes, 16);  // Align to 16 bytes
    const int64_t swizzled_scales_bytes = swizzled_scales_stride * num_tensors;
    auto swizzled_scales = at::empty({swizzled_scales_bytes},
                                     at::TensorOptions().dtype(at::kByte).device(device));
    uint8_t *swizzled_scales_dptr = reinterpret_cast<uint8_t *>(swizzled_scales.data_ptr());

    // Build TensorWrapper input/output pairs with scales
    std::vector<transformer_engine::TensorWrapper> inputs_nvte, outputs_nvte;
    inputs_nvte.reserve(num_tensors);
    outputs_nvte.reserve(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      inputs_nvte.emplace_back(scaling_mode);
      outputs_nvte.emplace_back(scaling_mode);
      auto& input_nvte = inputs_nvte.back();
      auto& output_nvte = outputs_nvte.back();
      output_nvte.set_with_gemm_swizzled_scales(true);
      void *in_scale_ptr = tensors[i].data_ptr();
      void *out_scale_ptr = swizzled_scales_dptr + i * swizzled_scales_stride;
      if (uniform_mxfp8_rowwise_swizzle || uniform_nvfp4_swizzle) {
        input_nvte.set_rowwise_data(nullptr, data_dtype, data_shape);
        input_nvte.set_rowwise_scale_inv(in_scale_ptr, scale_dtype, scale_shape);
        output_nvte.set_rowwise_data(nullptr, data_dtype, data_shape);
        output_nvte.set_rowwise_scale_inv(out_scale_ptr, scale_dtype, scale_shape);
      } else if (uniform_mxfp8_colwise_swizzle) {
        input_nvte.set_columnwise_data(nullptr, data_dtype, data_shape);
        input_nvte.set_columnwise_scale_inv(in_scale_ptr, scale_dtype, scale_shape);
        output_nvte.set_columnwise_data(nullptr, data_dtype, data_shape);
        output_nvte.set_columnwise_scale_inv(out_scale_ptr, scale_dtype, scale_shape);
      }
    }

    // Pack raw NVTETensors into vectors
    std::vector<NVTETensor> inputs_nvte_raw, outputs_nvte_raw;
    inputs_nvte_raw.reserve(num_tensors);
    outputs_nvte_raw.reserve(num_tensors);
    for (auto& t : inputs_nvte) inputs_nvte_raw.push_back(t.data());
    for (auto& t : outputs_nvte) outputs_nvte_raw.push_back(t.data());

    // Launch kernel
    nvte_multi_tensor_swizzle_scaling_factors(inputs_nvte_raw.data(), outputs_nvte_raw.data(),
                                              inputs_nvte_raw.size(),
                                              stream);

    // Collect data pointers
    std::vector<uint64_t> ptrs_host;
    ptrs_host.reserve(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      ptrs_host.push_back(reinterpret_cast<uintptr_t>(swizzled_scales_dptr
                                                      + i * swizzled_scales_stride));
    }

    // Load pointers on device
    auto ptrs_device = at::empty(std::vector<int64_t>{num_tensors},
                                 at::TensorOptions().dtype(at::kLong).device(device));
    nvte_store_value_on_device(ptrs_host.data(), ptrs_device.data_ptr(),
                               num_tensors * sizeof(uint64_t),
                               stream);

    return {std::move(ptrs_device), std::move(swizzled_scales)};
  }

  // Unsupported transform
  NVTE_ERROR("Unsupported transform type (", transform_type, ")");
}

}  // namespace transformer_engine::pytorch
