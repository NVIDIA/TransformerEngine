/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_
#define TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_

namespace transformer_engine::detail {

int get_num_compute_streams();

cudaStream_t get_compute_stream(int idx);

cudaEvent_t get_compute_stream_event(int idx);

}  // namespace transformer_engine::detail

#endif  // TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_
