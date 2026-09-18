/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <mublasLt.h>
#include <mudnncxx/mudnn.h>

#include "../extensions.h"

namespace transformer_engine::pytorch {

// size_t get_mublas_version() { return cublasLtGetVersion(); }

// size_t get_mudnn_version() { return cudnnGetVersion(); }

size_t get_mublas_version() { return mublasLtGetVersion(); }

size_t get_mudnn_version() { return ::musa::dnn::GetVersion(); }

}  // namespace transformer_engine::pytorch
