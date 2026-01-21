/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_TRANSPOSE_TRANSPOSE_H_
#define TRANSFORMER_ENGINE_COMMON_TRANSPOSE_TRANSPOSE_H_

#include "../common.h"

namespace transformer_engine {
namespace detail {

void transpose(const Tensor &input, const Tensor &noop, Tensor *output_, cudaStream_t stream);

}  // namespace detail
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_TRANSPOSE_TRANSPOSE_H_
