/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file common.cuh
 *  \brief Common functions in quantize.
 *
 *  Umbrella header. The contents are split into:
 *   - grouped_layout.cuh: architecture-neutral work-decomposition helpers
 *     (offset/tensor-id lookup, job/block descriptors, dbias reductions).
 *   - grouped_tma.cuh: architecture-specific TMA descriptor management and
 *     bulk-copy staging (pulls in arch-specific PTX via ptx.cuh).
 *
 *  Sources that only need the arch-neutral helpers should include
 *  grouped_layout.cuh directly so they are not forced into arch-specific
 *  (smXXXa/smXXXf) compilation.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_CORE_COMMON_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_CORE_COMMON_CUH_

#include "grouped_layout.cuh"
#include "grouped_tma.cuh"

#endif  // TRANSFORMER_ENGINE_QUANTIZE_CORE_COMMON_CUH_
