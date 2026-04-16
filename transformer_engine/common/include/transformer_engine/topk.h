/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_TOPK_H_
#define TRANSFORMER_ENGINE_TOPK_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute the top-K (key, index) pairs using the AIR radix algorithm.
 *
 *  Operates on a batch of rows: each row of length \p seq_len is processed
 *  independently and the \p k largest entries are selected.
 *
 *  \param[in]     stream           CUDA stream used for the operation.
 *  \param[in]     keys_in          Input keys tensor, flat storage for
 *                                  batch_size rows of seq_len elements.
 *  \param[in]     lengths_in       Per-row lengths, shape (batch_size,); int32.
 *                                  Fill with seq_len for uniform-length batches.
 *  \param[in,out] keys_out         Output top-k keys, flat storage for
 *                                  batch_size rows of k elements.
 *  \param[in,out] indices_out      Output top-k indices (within each row),
 *                                  flat storage for batch_size rows of k int32 elements.
 *  \param[in,out] workspace        Workspace tensor, shape (workspace_bytes,).
 *  \param[in]     batch_size       Number of rows.
 *  \param[in]     seq_len          Number of elements per row.
 *  \param[in]     k                Number of top-K entries to select per row.
 *  \param[in]     workspace_bytes  Workspace size in bytes; must be >=
 *                                  nvte_get_topk_workspace_bytes(batch_size, seq_len, k).
 *
 *  Supported key dtypes: float32, bfloat16.
 *  Index dtype: int32.
 */
void nvte_topk(cudaStream_t stream, const NVTETensor keys_in, const NVTETensor lengths_in,
               NVTETensor keys_out, NVTETensor indices_out, NVTETensor workspace,
               int batch_size, int seq_len, int k, size_t workspace_bytes);

/*! \brief Query the workspace size required by nvte_topk.
 *
 *  \param[in]  batch_size  Number of rows.
 *  \param[in]  seq_len     Number of elements per row.
 *  \param[in]  k           Top-K count.
 *  \return     Required workspace size in bytes.
 */
size_t nvte_get_topk_workspace_bytes(int batch_size, int seq_len, int k);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_TOPK_H_
