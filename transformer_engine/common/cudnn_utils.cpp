/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../fused_attn/utils.h"
#include "transformer_engine/cudnn.h"

namespace transformer_engine {

void nvte_cudnn_handle_init() {
  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
}

}  // namespace transformer_engine
