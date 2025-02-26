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

  // FP8 data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (!tensor.attr("_data").is_none()) {
    const auto &data = tensor.attr("_data").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
  }

  // FP8 data transpose
  if (!tensor.attr("_transpose_invalid").cast<bool>() && !tensor.attr("_transpose").is_none()) {
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

  // Row-scaled data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (!tensor.attr("_rowwise_data").is_none()) {
    const auto &data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
  }

  // Column-scaled data
  if (!tensor.attr("_columnwise_data").is_none()) {
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

}  // namespace detail

}  // namespace transformer_engine::pytorch
