/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {
namespace detail {

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, Quantizer *quantizer) {
  const at::Tensor &data = tensor.attr("_data").cast<at::Tensor>();
  const at::Tensor &scale_inv = tensor.attr("_scale_inv").cast<at::Tensor>();
  float *scale_inv_dptr = reinterpret_cast<float *>(scale_inv.data_ptr());
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();

  const auto &shape = getTensorShape(data);

  bool transpose_valid = !tensor.attr("_transpose_invalid").cast<bool>();
  std::optional<at::Tensor> transpose = std::nullopt;
  if (transpose_valid) {
    transpose = tensor.attr("_transpose").cast<std::optional<at::Tensor>>();
  }

  auto ret = TensorWrapper(quantizer->get_scaling_mode());

  ret.set_rowwise_data(data.data_ptr(), dtype, shape);
  if (transpose_valid && transpose != std::nullopt) {
    const auto &transpose_shape = getTensorShape(*transpose);
    ret.set_columnwise_data(transpose->data_ptr(), dtype, transpose_shape);
  }

  const auto scale_inv_dtype = GetTransformerEngineDType(scale_inv.scalar_type());
  const auto scale_inv_shape = getTensorShape(scale_inv);
  ret.set_rowwise_scale_inv(scale_inv_dptr, scale_inv_dtype, scale_inv_shape);
  ret.set_columnwise_scale_inv(scale_inv_dptr, scale_inv_dtype, scale_inv_shape);
  quantizer->set_quantization_params(&ret);
  return ret;
}

TensorWrapper NVTETensorFromMXFP8Tensor(py::handle tensor, Quantizer *quantizer) {
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();
  auto ret = TensorWrapper(NVTE_MXFP8_1D_SCALING);

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  if (rowwise_usage) {
    const at::Tensor &data_rowwise = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const at::Tensor &scale_inv_rowwise = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    void *scale_inv_rowwise_dptr = scale_inv_rowwise.data_ptr();
    const auto &shape = getTensorShape(data_rowwise);
    ret.set_rowwise_data(data_rowwise.data_ptr(), dtype, shape);

    const auto scale_inv_rowwise_shape = getTensorShape(scale_inv_rowwise);
    ret.set_rowwise_scale_inv(scale_inv_rowwise_dptr, DType::kFloat8E8M0, scale_inv_rowwise_shape);
  }

  if (columnwise_usage) {
    const at::Tensor &data_colwise = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const at::Tensor &scale_inv_colwise = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    void *scale_inv_colwise_dptr = scale_inv_colwise.data_ptr();
    const auto &shape = getTensorShape(data_colwise);
    ret.set_columnwise_data(data_colwise.data_ptr(), dtype, shape);

    const auto scale_inv_colwise_shape = getTensorShape(scale_inv_colwise);
    ret.set_columnwise_scale_inv(scale_inv_colwise_dptr, DType::kFloat8E8M0,
                                 scale_inv_colwise_shape);
  }

  quantizer->set_quantization_params(&ret);
  return ret;
}

}  // namespace detail

}  // namespace transformer_engine::pytorch
