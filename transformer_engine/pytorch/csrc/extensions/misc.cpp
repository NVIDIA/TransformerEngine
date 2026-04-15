/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine::pytorch {

// size_t get_mublas_version() { return cublasLtGetVersion(); }

// size_t get_mudnn_version() { return cudnnGetVersion(); }

size_t get_mublas_version() { return MUBLAS_VERSION_MAJOR * 10000ul + MUBLAS_VERSION_MINOR * 100ul + MUBLAS_VERSION_PATCH; }

size_t get_mudnn_version() { return ::musa::dnn::GetVersion(); }


}  // namespace transformer_engine::pytorch
