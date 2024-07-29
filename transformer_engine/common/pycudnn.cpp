/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

namespace cudnn_frontend {

// This is needed to define the symbol `cudnn_dlhandle`
// When using the flag NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
// to enable dynamic loading.
void *cudnn_dlhandle = nullptr;

}  // namespace cudnn_frontend
