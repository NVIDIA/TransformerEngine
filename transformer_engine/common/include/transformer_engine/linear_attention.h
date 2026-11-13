/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file linear_attention.h
 *  \brief Framework-independent linear-attention state composition.
 */
#ifndef TRANSFORMER_ENGINE_LINEAR_ATTENTION_H_
#define TRANSFORMER_ENGINE_LINEAR_ATTENTION_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Apply ordered affine summaries to selected recurrent states.
 *
 * Each output starts from a selected seed and repeatedly evaluates
 * ``state = state @ transition[record] + additive[record]``. This operation
 * serves both forward prefixes and backward suffixes; backward callers pass
 * the backward-oriented transitions and a negative record stride.
 *
 * All tensors are contiguous device tensors. Seeds and additives are FP32
 * [seeds, heads, value_dim, key_dim] and [records, heads, value_dim, key_dim].
 * Transitions are FP32 [records, heads, key_dim, key_dim]. Descriptors are
 * INT32 [outputs, 4], with rows (seed_index, first_record, count, stride).
 * Outputs are FP32 [outputs, heads, value_dim, key_dim]. Referenced seed and
 * record indices must be valid; count must be nonnegative.
 * A seed_index of -1 means an implicit zero state. Zero-count descriptors
 * copy their seed (or write zeros). No memory is allocated.
 */
void nvte_linear_attention_affine_prefix(const NVTETensor seeds, const NVTETensor additive,
                                         const NVTETensor transitions, const NVTETensor descriptors,
                                         NVTETensor output, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_LINEAR_ATTENTION_H_
