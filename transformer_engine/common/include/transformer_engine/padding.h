/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file padding.h
 *  \brief Functions handling padding.
 */

#ifndef TRANSFORMER_ENGINE_PADDING_H_
#define TRANSFORMER_ENGINE_PADDING_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Padding multiple tensors.
 *
 *  NOTE: Padding mode only support bottom.
 *
 *  For example, 3x3 matrix pad to 4x3 matrix.
 *
 *  source
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *
 *  destination
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *  | 0 | 0 | 0 |
 *
 *  \param[in]     num_tensors              Number of tensors.
 *  \param[in]     input_list               List of 2D input tensors.
 *  \param[in,out] output_list              List of padded tensors. Dimensions
 *                                          match tensors in input_list.
 *  \param[in]     padded_num_rows_list     List of padded num rows corresponding to input tensors.
 *  \param[in]     stream                   CUDA stream used for the operation.
 */
void nvte_multi_padding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                        const int* padded_num_rows_list, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_PADDING_H_
