/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_GEMM_CONFIG_H_
#define TRANSFORMER_ENGINE_GEMM_CONFIG_H_

#include <transformer_engine/transformer_engine.h>

#include <cstdint>

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
      sizeof(bool),        // with_gelu_epilogue
      sizeof(bool),        // with_dgelu_epilogue
      sizeof(NVTETensor),  // epilogue_aux_tensor
      sizeof(bool),        // use_split_accumulator
      sizeof(int)          // sm_count
  };
};

struct GroupedMatmulConfig {
  // Average dimension hints for cuBLASLt algorithm selection heuristics.
  // Value of 0 means "not set" - compute automatically from tensor shapes.
  int64_t avg_m = 0;
  int64_t avg_n = 0;
  int64_t avg_k = 0;

  // Number of streaming multiprocessors to use in GEMM kernel
  int sm_count = 0;

  // Track which attributes have been explicitly set
  bool avg_m_set = false;
  bool avg_n_set = false;
  bool avg_k_set = false;

  static constexpr size_t attr_sizes[] = {
      sizeof(avg_m), 
      sizeof(avg_n),
      sizeof(avg_k),
      sizeof(sm_count)
  };
};

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GEMM_CONFIG_H_
