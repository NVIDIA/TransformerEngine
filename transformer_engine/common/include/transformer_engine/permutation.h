/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PERMUTATION_H_
#define TRANSFORMER_ENGINE_PERMUTATION_H_

#include "transformer_engine.h"

template <typename T, bool FWD>
void nvte_permutation(const void *input, void *output, const int *sorted_row_id, int *row_id_map,
                      const float *prob, const int num_rows, const int num_topK, const int num_cols,
                      const int num_out_tokens, float *prob_grad = nullptr,
                      const void *input_fwd = nullptr, cudaStream_t stream = nullptr);

#endif  // TRANSFORMER_ENGINE_PERMUTATION_H_
