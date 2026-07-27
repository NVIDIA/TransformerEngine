/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVRTC source file for LayerNorm forward kernels. The host bundles this as a
// string header; specific template instantiations are requested via
// nvrtcAddNameExpression at runtime.

#include "kernel_traits.h"
#include "ln_fwd_kernels.cuh"
