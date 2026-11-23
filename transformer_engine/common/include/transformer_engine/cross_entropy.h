/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cross_entropy.h
 *  \brief Functions for the fused cross entropy loss.
 */

#ifndef TRANSFORMER_ENGINE_CROSS_ENTROPY_H_
#define TRANSFORMER_ENGINE_CROSS_ENTROPY_H_

#include <stdint.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Single-rank cross entropy backward that writes the gradient of the logits as MXFP8.
 *
 *  The gradient of row r is (softmax(logits[r]) - onehot(target[r])) * grad_output[r] (divided
 *  by n_non_ignore with reduce_loss, zero if target[r] == ignore_idx), computed in FP32 from the
 *  softmax statistics of the forward and rounded to BF16, with the same FP32 operations as the
 *  Triton backward of transformer_engine.pytorch.cross_entropy. It is then quantized as the
 *  MXFP8 cast quantizes a BF16 tensor (1x32 row-wise and 32x1 column-wise blocks), so the
 *  output has the bytes of quantizing that BF16 gradient. No label smoothing.
 *
 *  Requires SM 10.0+, a number of rows and a vocabulary size that are multiples of 128.
 *
 *  \param[in]     logits         BF16 logits, [rows, vocab].
 *  \param[in]     target         int64 target of each row, [rows].
 *  \param[in]     stats          FP32 (row maximum, sum of exp(logit - maximum)) of each row,
 *                                [rows, 2].
 *  \param[in]     n_non_ignore   int64 number of rows not ignored, [1]. Only read with
 *                                reduce_loss; may be NULL otherwise.
 *  \param[in]     grad_output    FP32 gradient of the loss: a scalar with reduce_loss, else
 *                                one value per row.
 *  \param[in,out] grad           MXFP8 gradient of the logits, [rows, vocab], E4M3 or E5M2,
 *                                with row-wise and/or column-wise data. Its scaling factors are
 *                                written compact or GEMM-swizzled, as the tensor specifies.
 *  \param[in]     ignore_idx     Target value of ignored rows.
 *  \param[in]     reduce_loss    Whether the loss was averaged over the rows not ignored.
 *  \param[in]     stream         CUDA stream used for the operation.
 */
void nvte_cross_entropy_backward_mxfp8(const NVTETensor logits, const NVTETensor target,
                                       const NVTETensor stats, const NVTETensor n_non_ignore,
                                       const NVTETensor grad_output, NVTETensor grad,
                                       int64_t ignore_idx, bool reduce_loss, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CROSS_ENTROPY_H_
