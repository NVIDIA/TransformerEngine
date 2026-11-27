/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_2D_NVFP4_H_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_2D_NVFP4_H_

#include "../../common.h"

namespace transformer_engine {

// Internal amax implementation, shared with the existing grouped RHT entry.
void group_hadamard_transform_amax_graph_safe(const GroupedTensor* input, GroupedTensor* output,
                                              uint16_t random_sign_mask,
                                              uint16_t random_sign_mask_t,
                                              bool broadcast_pre_rht_amax, cudaStream_t stream,
                                              const float* noop = nullptr);

namespace dispatch::nvfp4 {
void group_quantize_2d(const GroupedTensor* input, const Tensor* noop, GroupedTensor* output,
                       const QuantizationConfig* config, cudaStream_t stream);
}
}  // namespace transformer_engine
#endif
