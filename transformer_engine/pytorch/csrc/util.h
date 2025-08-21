/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_

#include <torch/extension.h>

#include <optional>

#include "transformer_engine/transformer_engine.h"

namespace transformer_engine::pytorch {

/*! \brief Swizzle the scaling factor of the input tensor.
 *
 * The returned swizzled scaling factor tensor should be kept alive during the GEMM.
 */
std::optional<at::Tensor> swizzle_scaling_factors(transformer_engine::TensorWrapper &input,
                                                  bool rowwise);

/*! \brief Swizzle the scaling factor of the input tensors.
 *
 * The returned swizzled scaling factor tensors should be kept alive during the GEMMs.
 */
std::optional<at::Tensor> multi_tensor_swizzle_scaling_factors(
    std::vector<transformer_engine::TensorWrapper> &inputs, bool rowwise);

/*! \brief Split a quantized tensor into multiple quantized tensors.
 *
 * Only MXFP8 quantized tensor is supported. Because when we ensure that `m_splits` is padded to
 * 32, quantizing the whole tensor will produce exactly the same data as splitting and then
 * quantizing, only the scaling factors are different due to padding. While for other recipes,
 * this doesn't hold.
 */
std::vector<py::object> split_quantized_tensor(py::handle tensor, std::vector<size_t> &m_splits);
}  // namespace transformer_engine::pytorch
#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_
