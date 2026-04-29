/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nvfp4_1x64.h
 *  \brief Small helpers for NVFP4 hierarchical 1x64 cast (env + config +
 *         preconditions), shared between NVFP4Quantizer and split_quantize to
 *         avoid duplicating policy. The "rowwise" in the env-var name is
 *         historical -- the kernel now produces rowwise and/or columnwise
 *         (transposed) output in a single fused pass.
 */
#ifndef TRANSFORMER_ENGINE_PYTORCH_NVFP4_1X64_H_
#define TRANSFORMER_ENGINE_PYTORCH_NVFP4_1X64_H_

#include <transformer_engine/transformer_engine.h>

#include "common/util/logging.h"
#include "common/util/system.h"

namespace transformer_engine::pytorch::nvfp4_1x64 {

/// Whether the hierarchical 1x64 cast is requested. The env-var name retains
/// its original ROWWISE_ prefix for backward compatibility with users of the
/// rowwise-only kernel that shipped first; both directions are now supported.
[[nodiscard]] inline bool local_encode_from_env() {
  return transformer_engine::getenv<bool>("NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE", false);
}

/// Apply 2D mode, SR, and optional 1x64 flag to a quantization config.
inline void config_apply(QuantizationConfigWrapper& cfg, bool nvfp4_2d, bool stochastic_rounding,
                         bool use_1x64) {
  cfg.set_nvfp4_2d_quantization(nvfp4_2d);
  cfg.set_stochastic_rounding(stochastic_rounding);
  cfg.set_nvfp4_rowwise_1x64_local_encode(use_1x64);
}

/// Preconditions for \p NVFP4Quantizer::quantize_impl (non-split).
inline void require_ok_for_non_split(bool with_rht, bool /* columnwise */, bool sr) {
  if (!local_encode_from_env()) {
    return;
  }
  NVTE_CHECK(
      !with_rht,
      "NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE=1 requires non-RHT (e.g. NVTE_NVFP4_DISABLE_RHT=1).");
  NVTE_CHECK(!sr,
             "NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE=1 is incompatible with stochastic rounding.");
}

/// Preconditions for \p split_quantize (non-RHT path).
inline void require_ok_for_split(bool /* want_rowwise */, bool /* have_columnwise */, bool sr) {
  if (!local_encode_from_env()) {
    return;
  }
  NVTE_CHECK(!sr,
             "NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE in split_quantize is incompatible with SR.");
}

}  // namespace transformer_engine::pytorch::nvfp4_1x64

#endif  // TRANSFORMER_ENGINE_PYTORCH_NVFP4_1X64_H_
