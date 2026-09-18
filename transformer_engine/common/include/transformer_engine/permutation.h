/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void nvte_permute_mask(const NVTETensor input, NVTETensor output, NVTETensor row_id_map,
                       const NVTETensor probs, NVTETensor permuted_probs, const int num_tokens,
                       const int num_experts, const int num_out_tokens, const int hidden_size,
                       musaStream_t stream = nullptr);

void nvte_unpermute_mask(const NVTETensor input, NVTETensor output, NVTETensor row_id_map,
                         const NVTETensor merging_probs, const NVTETensor permuted_probs,
                         NVTETensor unpermuted_probs, const int num_tokens, const int num_experts,
                         const int hidden_size, musaStream_t stream = nullptr);

void nvte_permute_mask_high_precision_probs(const NVTETensor input, NVTETensor output,
                                            NVTETensor row_id_map, const NVTETensor probs,
                                            NVTETensor permuted_probs, const int num_tokens,
                                            const int num_experts, const int num_out_tokens,
                                            const int hidden_size, musaStream_t stream = nullptr);
void nvte_unpermute_mask_high_precision_probs(const NVTETensor input, NVTETensor output,
                                              NVTETensor row_id_map, const NVTETensor merging_probs,
                                              const NVTETensor permuted_probs,
                                              NVTETensor unpermuted_probs, const int num_tokens,
                                              const int num_experts, const int hidden_size,
                                              musaStream_t stream = nullptr);

void nvte_unpermute_mask_bwd_with_merging_probs(
    const NVTETensor fwd_output_grad, NVTETensor fwd_input_grad, const NVTETensor fwd_input,
    const NVTETensor merging_probs, NVTETensor merging_probs_grad, NVTETensor row_id_map,
    const int num_tokens, const int num_experts, const int hidden_size,
    musaStream_t stream = nullptr);

void nvte_device_radix_sort_pairs(void *temp_storage, size_t *temp_storage_bytes, int *keys_in,
                                  int *keys_out, int *values_in, int *values_out, size_t num_items);

#endif  // TRANSFORMER_ENGINE_PERMUTATION_H_
