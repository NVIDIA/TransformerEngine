/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PERMUTATION_H_
#define TRANSFORMER_ENGINE_PERMUTATION_H_

#include "transformer_engine.h"

void nvte_permute(const NVTETensor input, NVTETensor output, const NVTETensor sorted_row_id,
                  NVTETensor row_id_map, const NVTETensor prob, NVTETensor prob_grad,
                  const NVTETensor input_fwd, const int num_rows, const int topK,
                  const int num_cols, const int num_out_tokens, cudaStream_t stream = nullptr);

void nvte_unpermute(const NVTETensor input, NVTETensor output, NVTETensor row_id_map,
                    const NVTETensor prob, const int num_rows, const int topK, const int num_cols,
                    cudaStream_t stream = nullptr);

#endif  // TRANSFORMER_ENGINE_PERMUTATION_H_
