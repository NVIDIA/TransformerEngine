/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_CUB_H_
#define TRANSFORMER_ENGINE_CUB_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute the top-K largest (key, value) pairs using CUB.
 *
 *  \param[in]     stream           CUDA stream used for the operation.
 *  \param[in]     keys_in          Input 1D keys tensor, shape (num_items,)
 *  \param[in]     values_in        Input 1D values tensor, shape (num_items,)
 *  \param[in,out] keys_out         Output 1D keys tensor, shape (k,)
 *  \param[in,out] values_out       Output 1D values tensor, shape (k,)
 *  \param[in,out] workspace        Workspace tensor, shape (workspace_bytes,)
 *  \param[in]     num_items        Number of items in the input tensor
 *  \param[in]     k                Number of top-K largest values to return
 *  \param[in]     workspace_bytes  Workspace size in bytes
 *
 *  Requirements:
 *  - Only supports float32, float16, bfloat16 keys and int32 values.
 */
void nvte_cub_topk(cudaStream_t stream, const NVTETensor keys_in, const NVTETensor values_in,
                   NVTETensor keys_out, NVTETensor values_out, NVTETensor workspace,
                   const int num_items, const int k, const size_t workspace_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
