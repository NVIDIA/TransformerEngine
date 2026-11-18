/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file linear_attention.h
 *  \brief Functions for linear-attention state composition.
 */
#ifndef TRANSFORMER_ENGINE_LINEAR_ATTENTION_H_
#define TRANSFORMER_ENGINE_LINEAR_ATTENTION_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compose recurrent states from ordered affine summaries.
 *
 * Each output starts from a selected initial memory and repeatedly evaluates
 * ``state = state @ transitions[summary_index] + additive[summary_index]``. This operation
 * serves both forward prefixes and backward suffixes; backward callers pass
 * the backward-oriented transitions and a negative record stride.
 *
 * All tensors must be contiguous device tensors. No memory is allocated.
 *
 * \param[in]  initial_states FP32 initial memories,
 *                            [initial_memories, heads, value_dim, key_dim].
 * \param[in]  additive       FP32 additive summaries,
 *                            [summaries, heads, value_dim, key_dim].
 * \param[in]  transitions    FP32 transition matrices,
 *                            [summaries, heads, key_dim, key_dim].
 * \param[in]  descriptors    INT32 [outputs, 4], with rows
 *                            (initial_memory_index, first_summary, count, stride).
 *                            An initial_memory_index of -1 selects an implicit zero state.
 *                            Other referenced memory and summary indices must be valid.
 *                            Count must be nonnegative; zero copies the initial memory
 *                            (or writes zeros).
 * \param[out] output         Caller-allocated FP32 composed memories,
 *                            [outputs, heads, value_dim, key_dim].
 * \param[in]  stream         CUDA stream on which to execute the operation.
 */
void nvte_linear_attention_affine_prefix(const NVTETensor initial_states, const NVTETensor additive,
                                         const NVTETensor transitions, const NVTETensor descriptors,
                                         NVTETensor output, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_LINEAR_ATTENTION_H_
