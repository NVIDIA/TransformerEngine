/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_SM_CARVEOUT_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_SM_CARVEOUT_H_

#include <cuda_runtime_api.h>

namespace transformer_engine {
namespace cuda {

/* \brief Fork a stream restricted to (sm_count() - sm_margin) SMs via a CUDA green context.
 *
 * Returns `stream` unchanged if sm_margin is zero or unsupported, so callers can use the
 * result unconditionally. Pair with sm_carveout_stream_end() (same `stream` and `sm_margin`)
 * before the launched work needs to be visible to `stream`'s other consumers.
 */
cudaStream_t sm_carveout_stream_begin(cudaStream_t stream, int sm_margin);

/* \brief Rejoin a stream from sm_carveout_stream_begin() into `stream`. */
void sm_carveout_stream_end(cudaStream_t stream, cudaStream_t carveout_stream, int sm_margin);

}  // namespace cuda
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_SM_CARVEOUT_H_
