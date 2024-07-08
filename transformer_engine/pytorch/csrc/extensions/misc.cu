/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }
