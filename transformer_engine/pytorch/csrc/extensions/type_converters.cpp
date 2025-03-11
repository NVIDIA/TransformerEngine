/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <transformer_engine/transformer_engine.h>

#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {
namespace detail {

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, Quantizer *quantizer) {
  auto ret = TensorWrapper(quantizer->get_scaling_mode());

  bool data_exists = !tensor.attr("_data").is_none();
  bool transpose_exists =
      !tensor.attr("_transpose_invalid").cast<bool>() && !tensor.attr("_transpose").is_none();

  NVTE_CHECK(data_exists || transpose_exists, "No data found for FP8 Tensor.");

  // FP8 data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (data_exists) {
    const auto &data = tensor.attr("_data").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
  }

  // FP8 data transpose
  if (transpose_exists) {
    const auto &data_transpose = tensor.attr("_transpose").cast<at::Tensor>();
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

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for MXFP8 Tensor.");

  // Row-scaled data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (rowwise_usage) {
    const auto &data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    ret.set_columnwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0,
                                 getTensorShape(scale_inv));
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromFloat8BlockwiseQTensor(py::handle tensor, Quantizer *quantizer) {
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();
  bool is_2D_scaled = tensor.attr("_is_2D_scaled").cast<bool>();

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  auto ret = TensorWrapper(is_2D_scaled ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);

  if (rowwise_usage) {
    const at::Tensor &data_rowwise = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const at::Tensor &scale_inv_rowwise = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    void *scale_inv_rowwise_dptr = scale_inv_rowwise.data_ptr();
    const auto &rowwise_shape = getTensorShape(data_rowwise);
    ret.set_rowwise_data(data_rowwise.data_ptr(), dtype, rowwise_shape);
    const auto scale_inv_rowwise_shape = getTensorShape(scale_inv_rowwise);
    ret.set_rowwise_scale_inv(scale_inv_rowwise_dptr, DType::kFloat32, scale_inv_rowwise_shape);
  }
  if (columnwise_usage) {
    const at::Tensor &data_colwise = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const at::Tensor &scale_inv_colwise = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    void *scale_inv_colwise_dptr = scale_inv_colwise.data_ptr();
    const auto &shape = getTensorShape(data_colwise);
    ret.set_columnwise_data(data_colwise.data_ptr(), dtype, shape);

    const auto scale_inv_colwise_shape = getTensorShape(scale_inv_colwise);
    ret.set_columnwise_scale_inv(scale_inv_colwise_dptr, DType::kFloat32, scale_inv_colwise_shape);
  }
  quantizer->set_quantization_params(&ret);
  return ret;
}

}  // namespace detail

}  // namespace transformer_engine::pytorch
