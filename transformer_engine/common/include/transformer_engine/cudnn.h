/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cudnn.h
 *  \brief Helper for cuDNN initialization
 */

#ifndef TRANSFORMER_ENGINE_CUDNN_H_
#define TRANSFORMER_ENGINE_CUDNN_H_

#include "transformer_engine.h"

/*! \namespace transformer_engine
 */
namespace transformer_engine {

/*! \brief TE/JAX cudaGraph requires the cuDNN initialization to happen outside of the capturing
 * region. This function is a helper to call cudnnCreate() which allocate memory for the handle.
 * The function will be called in the initialize() phase of the related XLA custom calls.
 */

void nvte_cudnn_handle_init();

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CUDNN_H_
