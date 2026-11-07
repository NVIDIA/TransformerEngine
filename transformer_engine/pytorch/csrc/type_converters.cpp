/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <transformer_engine/transformer_engine.h>

#include <optional>
#include <vector>

#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {
namespace detail {

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, Quantizer *quantizer) {
  auto ret = TensorWrapper(quantizer->get_scaling_mode());

  const py::object py_data = tensor.attr("_data");
  const bool data_exists = !py_data.is_none();
  py::object py_transpose = py::none();
  if (!tensor.attr("_transpose_invalid").cast<bool>()) {
    py_transpose = tensor.attr("_transpose");
  }
  const bool transpose_exists = !py_transpose.is_none();

  NVTE_CHECK(data_exists || transpose_exists, "No data found for FP8 Tensor.");

  // FP8 data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (data_exists) {
    const auto &data = py_data.cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
  }

  // FP8 data transpose
  if (transpose_exists) {
    const auto &data_transpose = py_transpose.cast<at::Tensor>();
    ret.set_columnwise_data(data_transpose.data_ptr(), fp8_dtype, getTensorShape(data_transpose));
  }

  // Scale-inverse
  {
    const auto &scale_inv = tensor.attr("_scale_inv").cast<at::Tensor>();
    float *dptr = reinterpret_cast<float *>(scale_inv.data_ptr());
    const auto &dtype = GetTransformerEngineDType(scale_inv.scalar_type());
    const auto &shape = getTensorShape(scale_inv);
    ret.set_rowwise_scale_inv(dptr, dtype, shape);
    ret.set_columnwise_scale_inv(dptr, dtype, shape);
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromMXFP8Tensor(py::handle tensor, Quantizer *quantizer) {
  auto ret = TensorWrapper(NVTE_MXFP8_1D_SCALING);

  const py::object py_rowwise_data = tensor.attr("_rowwise_data");
  const bool rowwise_usage = !py_rowwise_data.is_none();
  const py::object py_columnwise_data = tensor.attr("_columnwise_data");
  const bool columnwise_usage = !py_columnwise_data.is_none();
  const bool with_gemm_swizzled_scales = tensor.attr("_with_gemm_swizzled_scales").cast<bool>();

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for MXFP8 Tensor.");

  // Row-scaled data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (rowwise_usage) {
    const auto &data = py_rowwise_data.cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = py_columnwise_data.cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    ret.set_columnwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0,
                                 getTensorShape(scale_inv));
  }

  // Scale layout
  ret.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromFloat8BlockwiseQTensor(py::handle tensor, Quantizer *quantizer) {
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();
  bool is_2D_scaled = tensor.attr("_is_2D_scaled").cast<bool>();

  const py::object py_rowwise_data = tensor.attr("_rowwise_data");
  bool rowwise_usage = !py_rowwise_data.is_none();
  const py::object py_columnwise_data = tensor.attr("_columnwise_data");
  bool columnwise_usage = !py_columnwise_data.is_none();

  auto ret = TensorWrapper(is_2D_scaled ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);

  // Row-wise data
  if (rowwise_usage) {
    const at::Tensor &data_rowwise = py_rowwise_data.cast<at::Tensor>();
    const at::Tensor &scale_inv_rowwise = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    void *scale_inv_rowwise_dptr = scale_inv_rowwise.data_ptr();
    const auto &rowwise_shape = getTensorShape(data_rowwise);
    ret.set_rowwise_data(data_rowwise.data_ptr(), dtype, rowwise_shape);
    const auto scale_inv_rowwise_shape = getTensorShape(scale_inv_rowwise);
    ret.set_rowwise_scale_inv(scale_inv_rowwise_dptr, DType::kFloat32, scale_inv_rowwise_shape);
  }

  // Column-wise data
  if (columnwise_usage) {
    const at::Tensor &data_colwise = py_columnwise_data.cast<at::Tensor>();
    const at::Tensor &scale_inv_colwise = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    void *scale_inv_colwise_dptr = scale_inv_colwise.data_ptr();
    const auto &shape = getTensorShape(data_colwise);
    ret.set_columnwise_data(data_colwise.data_ptr(), dtype, shape);

    const auto scale_inv_colwise_shape = getTensorShape(scale_inv_colwise);
    ret.set_columnwise_scale_inv(scale_inv_colwise_dptr, DType::kFloat32, scale_inv_colwise_shape);
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromNVFP4Tensor(py::handle tensor, Quantizer *quantizer) {
  const DType dtype = tensor.attr("_fp4_dtype").cast<DType>();

  auto ret = TensorWrapper(NVTE_NVFP4_1D_SCALING);

  const py::object py_rowwise_data = tensor.attr("_rowwise_data");
  const bool rowwise_usage = !py_rowwise_data.is_none();
  const py::object py_columnwise_data = tensor.attr("_columnwise_data");
  const bool columnwise_usage = !py_columnwise_data.is_none();
  const bool with_gemm_swizzled_scales = tensor.attr("_with_gemm_swizzled_scales").cast<bool>();
  const bool row_scaled_nvfp4 = tensor.attr("_row_scaled_nvfp4").cast<bool>();
  const int nvfp4_e4m3_max = tensor.attr("_nvfp4_e4m3_max").cast<int>();
  const DType scale_inv_dtype = tensor.attr("_scale_dtype").cast<DType>();

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for NVFP4 Tensor.");

  // Row-scaled data
  if (rowwise_usage) {
    const auto &data = py_rowwise_data.cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), dtype,
                         convert_shape_back_from_fp4(getTensorShape(data), false));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), scale_inv_dtype, getTensorShape(scale_inv));
    const auto amax_rowwise = tensor.attr("_amax_rowwise");
    if (!amax_rowwise.is_none()) {
      const auto &amax = amax_rowwise.cast<at::Tensor>();
      ret.set_amax(amax.data_ptr(), DType::kFloat32, getTensorShape(amax));
    }
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = py_columnwise_data.cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    ret.set_columnwise_data(data.data_ptr(), DType::kFloat4E2M1,
                            convert_shape_back_from_fp4(getTensorShape(data), false));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), scale_inv_dtype, getTensorShape(scale_inv));
    const auto amax_columnwise = tensor.attr("_amax_columnwise");
    if (!amax_columnwise.is_none()) {
      const auto &amax = amax_columnwise.cast<at::Tensor>();
      ret.set_columnwise_amax(amax.data_ptr(), DType::kFloat32, getTensorShape(amax));
    }
  }

  // Scale layout
  ret.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);
  ret.set_row_scaled_nvfp4(row_scaled_nvfp4);
  if (nvfp4_e4m3_max != 0) {
    ret.set_nvfp4_e4m3_max(nvfp4_e4m3_max);
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

NVTEScalingMode ScalingModeFromQuantizer(py::handle quantizer) {
  auto *quantizer_ptr = quantizer.ptr();
  if (IsMXFP8Quantizers(quantizer_ptr)) {
    return NVTE_MXFP8_1D_SCALING;
  }
  if (IsNVFP4Quantizers(quantizer_ptr)) {
    return NVTE_NVFP4_1D_SCALING;
  }
  if (IsFloat8BlockwiseQuantizers(quantizer_ptr)) {
    const int block_scaling_dim = quantizer.attr("block_scaling_dim").cast<int>();
    return (block_scaling_dim == 2) ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D;
  }
  return NVTE_DELAYED_TENSOR_SCALING;
}

DType GetTransformerEngineDTypeForScaleInv(py::handle quantizer, at::Tensor scale_inv) {
  auto *quantizer_ptr = quantizer.ptr();
  if (IsMXFP8Quantizers(quantizer_ptr)) {
    return DType::kFloat8E8M0;
  }
  if (IsFloat8BlockwiseQuantizers(quantizer_ptr)) {
    return DType::kFloat32;
  }
  if (IsNVFP4Quantizers(quantizer_ptr)) {
    return quantizer.attr("scale_dtype").cast<DType>();
  }
  return GetTransformerEngineDType(scale_inv.scalar_type());
}

GroupedTensorWrapper GroupedTensorFromPyTorchGroupedTensor(py::handle tensor) {
  // Returns a GroupedTensorWrapper from a PyTorch GroupedTensor.
  const auto num_tensors = tensor.attr("num_tensors").cast<size_t>();
  const auto logical_shape = tensor.attr("logical_shape").cast<std::vector<size_t>>();
  const py::object quantizer = tensor.attr("quantizer");
  DType quantizer_dtype = DType::kNumTypes;
  NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
  if (!quantizer.is_none()) {
    scaling_mode = ScalingModeFromQuantizer(quantizer);
    quantizer_dtype = quantizer.attr("dtype").cast<DType>();
  }
  auto ret = GroupedTensorWrapper(num_tensors, logical_shape, scaling_mode);

  auto get_initialized_storage_shape = [](const at::Tensor &data) -> std::vector<size_t> {
    const auto numel = data.numel();
    if (numel == 0) {
      // PyTorch may use a null pointer for a valid zero-sized allocation. TE Common reserves
      // {nullptr, {0}} for uninitialized storage, so use an orientation-neutral 2D empty shape
      // to distinguish explicitly provided rowwise or columnwise storage from missing storage.
      return {0, 0};
    }
    // Grouped data is contiguous storage; its Python view shape is not the grouped logical shape.
    return {static_cast<size_t>(numel)};
  };

  // Rowwise data
  const py::object py_rowwise_data = tensor.attr("rowwise_data");
  if (!py_rowwise_data.is_none()) {
    const auto &data = py_rowwise_data.cast<at::Tensor>();
    DType data_dtype =
        quantizer.is_none() ? GetTransformerEngineDType(data.scalar_type()) : quantizer_dtype;
    ret.set_rowwise_data(data.data_ptr(), data_dtype, get_initialized_storage_shape(data));
  } else if (quantizer_dtype != DType::kNumTypes) {
    ret.set_rowwise_data(nullptr, quantizer_dtype, std::vector<size_t>{0});
  }

  // Columnwise data
  const py::object py_columnwise_data = tensor.attr("columnwise_data");
  if (!py_columnwise_data.is_none()) {
    const auto &data = py_columnwise_data.cast<at::Tensor>();
    DType data_dtype =
        quantizer.is_none() ? GetTransformerEngineDType(data.scalar_type()) : quantizer_dtype;
    ret.set_columnwise_data(data.data_ptr(), data_dtype, get_initialized_storage_shape(data));
  } else if (quantizer_dtype != DType::kNumTypes) {
    ret.set_columnwise_data(nullptr, quantizer_dtype, std::vector<size_t>{0});
  }

  // Scale
  const py::object py_scale = tensor.attr("scale");
  if (!py_scale.is_none()) {
    const auto &scale = py_scale.cast<at::Tensor>();
    ret.set_scale(scale.data_ptr(), GetTransformerEngineDType(scale.scalar_type()),
                  getTensorShape(scale));
  }

  // Amax
  const py::object py_amax = tensor.attr("amax");
  if (!py_amax.is_none()) {
    const auto &amax = py_amax.cast<at::Tensor>();
    ret.set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                 getTensorShape(amax));
  }
  const py::object py_columnwise_amax = tensor.attr("columnwise_amax");
  if (!py_columnwise_amax.is_none()) {
    const auto &amax = py_columnwise_amax.cast<at::Tensor>();
    ret.set_columnwise_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                            getTensorShape(amax));
  }

  // Scale inverse dtype
  py::object py_scale_inv_dtype = tensor.attr("scale_inv_dtype");
  const std::optional<DType> scale_inv_dtype = py_scale_inv_dtype.cast<std::optional<DType>>();

  // Scale inverse
  const py::object py_scale_inv = tensor.attr("scale_inv");
  if (!py_scale_inv.is_none()) {
    const auto &scale_inv = py_scale_inv.cast<at::Tensor>();
    NVTE_CHECK(scale_inv_dtype, "Could not determine dtype of scale_inv buffer.");
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), *scale_inv_dtype, getTensorShape(scale_inv));
  }
  const py::object py_columnwise_scale_inv = tensor.attr("columnwise_scale_inv");
  if (!py_columnwise_scale_inv.is_none()) {
    const auto &scale_inv = py_columnwise_scale_inv.cast<at::Tensor>();
    NVTE_CHECK(scale_inv_dtype, "Could not determine dtype of scale_inv buffer.");
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), *scale_inv_dtype, getTensorShape(scale_inv));
  }

  // Shape metadata
  const py::object py_first_dims = tensor.attr("first_dims");
  if (!py_first_dims.is_none()) {
    const auto &first_dims = py_first_dims.cast<at::Tensor>();
    ret.set_first_dims(first_dims.data_ptr(), GetTransformerEngineDType(first_dims.scalar_type()),
                       getTensorShape(first_dims));
  }
  const py::object py_last_dims = tensor.attr("last_dims");
  if (!py_last_dims.is_none()) {
    const auto &last_dims = py_last_dims.cast<at::Tensor>();
    ret.set_last_dims(last_dims.data_ptr(), GetTransformerEngineDType(last_dims.scalar_type()),
                      getTensorShape(last_dims));
  }
  const py::object py_tensor_offsets = tensor.attr("tensor_offsets");
  if (!py_tensor_offsets.is_none()) {
    const auto &tensor_offsets = py_tensor_offsets.cast<at::Tensor>();
    ret.set_tensor_offsets(tensor_offsets.data_ptr(),
                           GetTransformerEngineDType(tensor_offsets.scalar_type()),
                           getTensorShape(tensor_offsets));
  }

  const bool with_gemm_swizzled = tensor.attr("_with_gemm_swizzled_scales").cast<bool>();
  ret.set_with_gemm_swizzled_scales(with_gemm_swizzled);

  if (py::hasattr(tensor, "_row_scaled_nvfp4")) {
    ret.set_row_scaled_nvfp4(tensor.attr("_row_scaled_nvfp4").cast<bool>());
  }

  return ret;
}

}  // namespace detail

}  // namespace transformer_engine::pytorch
