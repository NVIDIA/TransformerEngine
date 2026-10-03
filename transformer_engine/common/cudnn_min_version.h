/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_CUDNN_MIN_VERSION_H_
#define TRANSFORMER_ENGINE_CUDNN_MIN_VERSION_H_

#include <cudnn.h>

#include <cstddef>

// Split out from cudnn_utils.h so the framework extensions can assert the build-time cuDNN
// version without pulling in the cuDNN frontend headers, which they do not otherwise need.

namespace transformer_engine {

// Minimum cuDNN version supported by Transformer Engine, encoded as cudnnGetVersion() reports it.
// Keep in sync with MIN_CUDNN_VERSION in transformer_engine/common/__init__.py.
constexpr size_t kMinCudnnVersion = 91200;

static_assert(static_cast<size_t>(CUDNN_VERSION) >= kMinCudnnVersion,
              "Transformer Engine must be built against cuDNN 9.12.0 or later headers.");

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CUDNN_MIN_VERSION_H_
