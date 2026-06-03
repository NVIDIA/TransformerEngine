/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NORM_RTC_DISPATCH_H_
#define TRANSFORMER_ENGINE_COMMON_NORM_RTC_DISPATCH_H_

#include <transformer_engine/transformer_engine.h>

#include "common.h"

namespace transformer_engine {
namespace normalization {
namespace rtc_norm {

// Register an RTC-backed launcher for a single LayerNorm Forward "tuned"
// (multi-CTA-capable) config. Compiles and launches via NVRTC on first use.
void register_ln_fwd_tuned(DType wtype, DType itype, DType otype, DType ctype, int hidden_size,
                           int ctas_per_row, int warps_m, int warps_n, int bytes_per_ldg);

// Register an RTC-backed launcher for a single LayerNorm Forward "general"
// (no multi-CTA) config.
void register_ln_fwd_general(DType wtype, DType itype, DType otype, DType ctype, int hidden_size,
                             int warps_m, int warps_n, int bytes_per_ldg);

// Register an RTC-backed launcher for a single LayerNorm Backward "tuned" config.
void register_ln_bwd_tuned(DType wtype, DType itype, DType otype, DType ctype, int hidden_size,
                           int ctas_per_row, int warps_m, int warps_n, int bytes_per_ldg_main,
                           int bytes_per_ldg_final);

// Register an RTC-backed launcher for a single LayerNorm Backward "general" config.
void register_ln_bwd_general(DType wtype, DType itype, DType otype, DType ctype, int hidden_size,
                             int warps_m, int warps_n, int bytes_per_ldg_main,
                             int bytes_per_ldg_final);

// Same set for RMSNorm.
void register_rmsnorm_fwd_tuned(DType wtype, DType itype, DType otype, DType ctype, int hidden_size,
                                int ctas_per_row, int warps_m, int warps_n, int bytes_per_ldg);
void register_rmsnorm_fwd_general(DType wtype, DType itype, DType otype, DType ctype,
                                  int hidden_size, int warps_m, int warps_n, int bytes_per_ldg);
void register_rmsnorm_bwd_tuned(DType wtype, DType itype, DType otype, DType ctype, int hidden_size,
                                int ctas_per_row, int warps_m, int warps_n, int bytes_per_ldg_main,
                                int bytes_per_ldg_final, bool with_add);
void register_rmsnorm_bwd_general(DType wtype, DType itype, DType otype, DType ctype,
                                  int hidden_size, int warps_m, int warps_n, int bytes_per_ldg_main,
                                  int bytes_per_ldg_final, bool with_add);

}  // namespace rtc_norm
}  // namespace normalization
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_NORM_RTC_DISPATCH_H_
