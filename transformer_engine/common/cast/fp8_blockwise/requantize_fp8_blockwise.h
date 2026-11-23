/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#ifndef TRANSFORMER_ENGINE_REQUANTIZE_FP8_BLOCKWISE_H_
#define TRANSFORMER_ENGINE_REQUANTIZE_FP8_BLOCKWISE_H_
#include "../../common.h"
namespace transformer_engine::requantize {
void group_requantize_block_scaling(const GroupedTensor& input, GroupedTensor* output,
                                    Tensor* dequantized, const QuantizationConfig* config,
                                    cudaStream_t stream);
}
#endif
