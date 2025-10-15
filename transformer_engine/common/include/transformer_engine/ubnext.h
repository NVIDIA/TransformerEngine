/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UBNEXT_H_
#define TRANSFORMER_ENGINE_UBNEXT_H_

#include "transformer_engine.h"

namespace transformer_engine {

#ifdef __cplusplus
extern "C" {
#endif

void allreduce_2shot_mc(int ranks, int myrank, void* uc0ptr, void* mc0ptr, void* mcptr_in,
                        void* mcptr_out, size_t bytes, void* residual_in, void* residual_out,
                        bool fuse_layernorm, void* gamma, float eps, const int hidden_size,
                        cudaStream_t stream);
void allreduce_2shot_mc_lamport(int ranks, int myrank, void* uc0ptr, void* mc0ptr, void* ucptr_out,
                                void* mcptr_in, void* mcptr_out, void* clear_ptr, size_t bytes,
                                bool poisoned, void* residual_in, void* residual_out,
                                bool fuse_layernorm, void* gamma, float eps, const int hidden_size,
                                cudaStream_t stream);
void allreduce_2shot_uc(int ranks, int myrank, void* uc0ptr, void* ucptr_in, void* ucptr_out,
                        size_t bytes, void* residual_in, void* residual_out, bool fuse_layernorm,
                        void* gamma, float eps, const int hidden_size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
}  // namespace transformer_engine

#endif
