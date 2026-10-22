/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FROST_OUTPUT_POLICY_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FROST_OUTPUT_POLICY_H_

#include <cstddef>

#include "../util/system.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine::fused_attn {

// Default-off output initialization policy. The framework may omit output initialization
// only under this same predicate used by common dispatch. Strict mode prevents
// native fallback; common dispatch validates the device lengths/origins on every
// execution before FROST writes all rows. Pitched and unequal layouts keep zeros.
inline bool frost_d128_compact_outputs(NVTE_QKV_Layout qkv, NVTE_QKV_Layout dqkv, NVTEDType dtype,
                                       size_t dq, size_t dv, size_t batch, size_t tq, size_t tk,
                                       size_t sq, size_t sk, size_t hq, size_t hkv) {
  return getenv<int>("NVTE_FUSED_ATTN_FROST") == 2 &&
         getenv<bool>("NVTE_FUSED_ATTN_FROST_D128_NOZERO") && qkv == NVTE_THD_THD_THD &&
         dqkv == qkv && dtype == kNVTEBFloat16 && dq == 128 && dv == 128 && batch > 0 && sq > 0 &&
         sq == sk && sq % 128 == 0 && tq % batch == 0 && tk % batch == 0 && tq / batch == sq &&
         tk / batch == sk && hkv > 0 && hq > hkv && hq % hkv == 0;
}

}  // namespace transformer_engine::fused_attn
#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FROST_OUTPUT_POLICY_H_
