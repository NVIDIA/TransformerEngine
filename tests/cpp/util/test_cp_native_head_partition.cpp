/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE for license information.
 ************************************************************************/
// Verify partition coverage and physical-address equivalence in the actual C++ helper.
#include <set>

#include "fused_attn/cp_native_head_partition.h"
#include "gtest/gtest.h"

using transformer_engine::fused_attn::cp_native_head_geometry_supported;
using transformer_engine::fused_attn::cp_native_head_offsets;
using transformer_engine::fused_attn::CPNativeHeadGeometry;

TEST(CPNativeHeadPartition, PreservesPhysicalStorageAndWholeGqaGroups) {
  size_t cases = 0, addresses = 0;
  for (size_t sq : {size_t(64), size_t(1024)}) {
    const size_t batch = sq == 64 ? 2 : 64;
    for (size_t sk_multiple = 1; sk_multiple <= 8; ++sk_multiple) {
      CPNativeHeadGeometry g{batch, 32, 8, 256, 256, sq, sk_multiple * sq, 2 * sq, 8 * sq};
      const size_t origin = sk_multiple > 4 ? sq : 0;
      for (size_t chunk : {size_t(4), size_t(8), size_t(16)}) {
        ASSERT_TRUE(cp_native_head_geometry_supported(g, chunk));
        std::set<size_t> query_heads, kv_heads;
        for (size_t head = 0; head < 32; head += chunk) {
          auto offsets = cp_native_head_offsets(g, head);
          for (size_t local = 0; local < chunk; ++local) {
            const size_t h = head + local;
            ASSERT_TRUE(query_heads.insert(h).second);
            for (size_t b : {size_t(0), batch - 1}) {
              for (size_t token : {size_t(0), sq / 2, sq - 1}) {
                for (size_t d : {size_t(0), size_t(17), size_t(255)}) {
                  const size_t full = ((b * g.query_pitch + origin + token) * 32 + h) * 256 + d;
                  const size_t partition = origin * 32 * 256 + offsets.query +
                                           b * g.query_pitch * 32 * 256 + token * 32 * 256 +
                                           local * 256 + d;
                  ASSERT_TRUE(full == partition && full < batch * g.query_pitch * 32 * 256);
                  const size_t full_stats = (b * 32 + h) * sq + token;
                  const size_t part_stats = offsets.stats + b * 32 * sq + local * sq + token;
                  ASSERT_TRUE(full_stats == part_stats && part_stats < batch * 32 * sq);
                  ++addresses;
                }
              }
            }
          }
          for (size_t local = 0; local < chunk / 4; ++local) {
            const size_t h = head / 4 + local;
            ASSERT_TRUE(kv_heads.insert(h).second);
            for (size_t b : {size_t(0), batch - 1}) {
              for (size_t token : {size_t(0), g.kv_length / 2, g.kv_length - 1}) {
                for (size_t d : {size_t(0), size_t(17), size_t(255)}) {
                  const size_t full = ((b * g.kv_pitch + token) * 8 + h) * 256 + d;
                  const size_t partition =
                      offsets.kv + b * g.kv_pitch * 8 * 256 + token * 8 * 256 + local * 256 + d;
                  ASSERT_TRUE(full == partition && full < batch * g.kv_pitch * 8 * 256);
                  ++addresses;
                }
              }
            }
          }
        }
        ASSERT_TRUE(query_heads.size() == 32 && kv_heads.size() == 8);
        // Concrete negative controls: compacting strides to the chunk head extent
        // or applying the query origin twice must change a valid physical address.
        ASSERT_TRUE(g.query_pitch * chunk * 256 != g.query_pitch * 32 * 256);
        ASSERT_TRUE(chunk * sq != 32 * sq);
        if (origin) {
          ASSERT_TRUE(origin * 32 * 256 != 2 * origin * 32 * 256);
        }
        ++cases;
      }
      for (size_t requested : {size_t(0), size_t(1), size_t(2), size_t(32), size_t(64)})
        ASSERT_TRUE(!cp_native_head_geometry_supported(g, requested));
      auto bad = g;
      ++bad.batch;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
      bad = g;
      bad.query_heads = 16;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
      bad = g;
      bad.kv_heads = 4;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
      bad = g;
      bad.value_dim = 128;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
      bad = g;
      ++bad.query_pitch;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
      bad = g;
      ++bad.kv_pitch;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
      bad = g;
      --bad.kv_length;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
      bad = g;
      bad.query_length = 0;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
      bad = g;
      bad.kv_length = 9 * sq;
      ASSERT_TRUE(!cp_native_head_geometry_supported(bad, 8));
    }
  }
  ASSERT_TRUE(cases == 48 && addresses == 34560);
}
