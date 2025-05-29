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

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Number of CUDA streams to use in multi-stream operations */
int nvte_get_num_compute_streams();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_MULTI_STREAM_H
