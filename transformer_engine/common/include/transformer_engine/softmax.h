/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_SOFTMAX_H_
#define TRANSFORMER_ENGINE_SOFTMAX_H_

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif


inline int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}


void nvte_scaled_softmax_forward_cuda(
    const NVTETensor input,
    NVTETensor softmax_results,
    float scale_factor,
    cudaStream_t stream
);


void nvte_scaled_softmax_backward_cuda(
    const NVTETensor output_grads,
    const NVTETensor softmax_results,
    float scale_factor,
    cudaStream_t stream
);


int get_batch_per_block_cuda(int query_seq_len, int key_seq_len, int batches, int attn_heads);


void nvte_scaled_masked_softmax_forward_cuda(
    const NVTETensor input,
    const NVTETensor mask,
    NVTETensor softmax_results,
    float scale_factor,
    cudaStream_t stream
);


void nvte_scaled_masked_softmax_backward_cuda(
    const NVTETensor input,
    NVTETensor softmax_results,
    float scale_factor,
    cudaStream_t stream
);


void nvte_scaled_upper_triang_masked_softmax_forward_cuda(
    const NVTETensor input,
    NVTETensor softmax_results,
    float scale_factor,
    cudaStream_t stream
);


void nvte_scaled_upper_triang_masked_softmax_backward_cuda(
    const NVTETensor output_grads,
    const NVTETensor softmax_results,
    float scale_factor,
    cudaStream_t stream
);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_SOFTMAX_H_
