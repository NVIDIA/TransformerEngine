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

/* Swizzle the scaling factor of the input tensor.
 *
 * The returned swizzled scaling factor tensor should be kept alive during the GEMM.
 */
std::optional<at::Tensor> swizzle_scaling_factors(transformer_engine::TensorWrapper &input,
                                                  bool trans);

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_
