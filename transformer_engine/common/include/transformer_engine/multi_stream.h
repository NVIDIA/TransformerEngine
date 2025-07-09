/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file multi_stream.h
 *  \brief Functions for multi streams executions.
 */

#ifndef TRANSFORMER_ENGINE_MULTI_STREAM_H
#define TRANSFORMER_ENGINE_MULTI_STREAM_H

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Number of CUDA streams to use in multi-stream operations */
int nvte_get_num_compute_streams();

/*! \brief Get a CUDA stream for compute operations.
 *
 *  \param[in] idx Index of the stream to retrieve.Add commentMore actions
 *  \return A cudaStream_t.
 *
 *  This function returns a CUDA stream that can be used for compute operations.
 *  The index should be in the range [0, nvte_get_num_compute_streams() - 1].
 */
cudaStream_t nvte_get_compute_stream(const int idx);

/*! \brief Get a CUDA event for compute operations.
 *
 *  \param[in] idx Index of the event to retrieve.
 *  \return A cudaEvent_t.
 *
 *  This function returns a CUDA event that can be used to synchronize compute operations.
 *  The index should be in the range [0, nvte_get_num_compute_streams() - 1].
 */
cudaEvent_t nvte_get_compute_stream_event(const int idx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_MULTI_STREAM_H
