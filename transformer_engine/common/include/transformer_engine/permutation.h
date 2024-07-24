/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PERMUTATION_H_
#define TRANSFORMER_ENGINE_PERMUTATION_H_

#include "transformer_engine.h"

void nvte_permute(const void *input, void *output, const transformer_engine::DType dtype,
                  const int *sorted_row_id, int *row_id_map, const float *prob, const int num_rows,
                  const int num_topK, const int num_cols, const int num_out_tokens,
                  float *prob_grad = nullptr, const void *input_fwd = nullptr,
                  cudaStream_t stream = nullptr);

void nvte_unpermute(const void *input, void *output, const transformer_engine::DType dtype,
                    int *row_id_map, const float *prob, const int num_rows, const int num_topK,
                    const int num_cols, cudaStream_t stream = nullptr);

#endif  // TRANSFORMER_ENGINE_PERMUTATION_H_
