/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_CAST_MXFP8_REQUANTIZE_MXFP8_H_
#define TRANSFORMER_ENGINE_CAST_MXFP8_REQUANTIZE_MXFP8_H_

#include "../../common.h"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {

void group_requantize(const GroupedTensor &input, GroupedTensor *output, Tensor *dequantized,
                      cudaStream_t stream);

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CAST_MXFP8_REQUANTIZE_MXFP8_H_
