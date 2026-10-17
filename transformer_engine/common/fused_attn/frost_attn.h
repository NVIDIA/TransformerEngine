/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FROST_ATTN_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FROST_ATTN_H_

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>

#include "config_and_params.h"

namespace transformer_engine {
namespace fused_attn {

// q,k,v,o,do,stats,dq,dk,dv,cu_q,cu_kv followed by the nine physical origins.
using FrostPointers = std::array<void *, 20>;

// A null workspace prepares the plan and returns the framework-owned byte count.
// False selects the existing backend. Once execution starts errors propagate.
bool frost_attention_backward(const FusedAttnConfig &cfg, const FrostPointers &pointers,
                              void *workspace, size_t *workspace_size, cudaStream_t stream);

}  // namespace fused_attn
}  // namespace transformer_engine
#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FROST_ATTN_H_
