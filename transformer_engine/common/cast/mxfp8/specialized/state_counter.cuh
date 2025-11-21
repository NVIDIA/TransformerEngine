/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file state_counter.cuh
 *  \brief CUDA kernels to count state.
 */

#ifndef TRANSFORMER_ENGINE_SPECIALIZED_STATE_COUNTER_CUH_
#define TRANSFORMER_ENGINE_SPECIALIZED_STATE_COUNTER_CUH_

#include <cstdint>

namespace transformer_engine {

template <int32_t numStages, bool Flip = false>
struct PipeState {
  int2 _storage;  // x: index, y: phase

  __device__ __forceinline__ PipeState() : _storage{0, 0} {
    if constexpr (Flip) {
      _storage.y ^= 1;
    }
  }

  __device__ __forceinline__ int32_t index() const { return _storage.x; }

  __device__ __forceinline__ int32_t phase() const { return _storage.y; }

  __device__ __forceinline__ void operator++(int32_t) {
    if constexpr (numStages > 0) {
      _storage.x++;
      if (_storage.x == numStages) {
        _storage.x = 0;
        _storage.y ^= 1;
      }
    }
  }
};

template <int32_t numStages>
struct PipeStateCounter {
  int32_t _counter;

  __device__ __forceinline__ PipeStateCounter() : _counter(0) {}

  __device__ __forceinline__ int32_t index() const { return _counter; }

  __device__ __forceinline__ void operator++(int32_t) {
    if constexpr (numStages > 0) {
      _counter++;
      _counter = _counter == numStages ? 0 : _counter;
    }
  }
};

}  // namespace transformer_engine

#endif  // #ifndef TRANSFORMER_ENGINE_SPECIALIZED_STATE_COUNTER_CUH_
