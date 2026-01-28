/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_GEMM_CONFIG_H_
#define TRANSFORMER_ENGINE_GEMM_CONFIG_H_

#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <optional>

namespace transformer_engine {

struct MatmulConfig {
  NVTETensor bias_tensor = nullptr;
  NVTETensor dbias_tensor = nullptr;
  bool with_gelu_epilogue = false;
  bool with_dgelu_epilogue = false;
  NVTETensor epilogue_aux_tensor = nullptr;
  bool use_split_accumulator = false;
  int sm_count = 0;

  static constexpr size_t attr_sizes[] = {
      sizeof(NVTETensor),  // bias_tensor
      sizeof(NVTETensor),  // dbias_tensor
      sizeof(uint8_t),     // with_gelu_epilogue
      sizeof(uint8_t),     // with_dgelu_epilogue
      sizeof(NVTETensor),  // epilogue_aux_tensor
      sizeof(uint8_t),     // use_split_accumulator
      sizeof(int32_t)      // sm_count
  };
};

struct GroupedMatmulConfig {
  // Average dimension hints for cuBLASLt algorithm selection heuristics.
  // nullopt means "not set" - compute automatically from tensor shapes.
  std::optional<int64_t> avg_m;
  std::optional<int64_t> avg_n;
  std::optional<int64_t> avg_k;

  // Number of streaming multiprocessors to use in GEMM kernel
  int sm_count = 0;

  // Note: API transfers the value type, not std::optional
  static constexpr size_t attr_sizes[] = {sizeof(decltype(avg_m)::value_type),
                                          sizeof(decltype(avg_n)::value_type),
                                          sizeof(decltype(avg_k)::value_type), sizeof(sm_count)};
};

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GEMM_CONFIG_H_
