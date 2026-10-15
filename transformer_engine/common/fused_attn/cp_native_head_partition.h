/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE for license information.
 ************************************************************************/
#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CP_NATIVE_HEAD_PARTITION_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CP_NATIVE_HEAD_PARTITION_H_

#include <cstddef>

namespace transformer_engine {
namespace fused_attn {

struct CPNativeHeadGeometry {
  size_t batch, query_heads, kv_heads, query_dim, value_dim;
  size_t query_length, kv_length, query_pitch, kv_pitch;
};

inline bool cp_native_head_geometry_supported(const CPNativeHeadGeometry &g, size_t chunk) {
  const bool small = g.batch == 2 && g.query_length == 64;
  const bool full = g.batch == 64 && g.query_length == 1024;
  return (small || full) && (chunk == 4 || chunk == 8 || chunk == 16) && g.query_heads == 32 &&
         g.kv_heads == 8 && g.query_dim == 256 && g.value_dim == 256 &&
         g.query_pitch == 2 * g.query_length && g.kv_pitch == 8 * g.query_length &&
         g.kv_length >= g.query_length && g.kv_length <= g.kv_pitch &&
         g.kv_length % g.query_length == 0;
}

struct CPNativeHeadOffsets {
  size_t query, kv, stats;
};

inline CPNativeHeadOffsets cp_native_head_offsets(const CPNativeHeadGeometry &g, size_t head) {
  return {head * g.query_dim, head / (g.query_heads / g.kv_heads) * g.query_dim,
          head * g.query_length};
}

}  // namespace fused_attn
}  // namespace transformer_engine
#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CP_NATIVE_HEAD_PARTITION_H_
