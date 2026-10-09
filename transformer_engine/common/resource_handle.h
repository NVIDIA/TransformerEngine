/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_RESOURCE_HANDLE_H_
#define TRANSFORMER_ENGINE_COMMON_RESOURCE_HANDLE_H_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <transformer_engine/resource_handle.h>

#include <cstddef>

namespace transformer_engine {

// Passed by value to the updater and consumer kernels. Null pointers select the
// original per-translation-unit device symbols for calls without a handle.
struct GroupedTmaStorageView {
  CUtensorMap *input = nullptr;
  CUtensorMap *act_input = nullptr;
  CUtensorMap *output_rowwise = nullptr;
  CUtensorMap *output_colwise = nullptr;
  size_t *rows = nullptr;
  size_t *cols = nullptr;
  size_t *offsets = nullptr;
};

GroupedTmaStorageView get_grouped_tma_storage(NVTEResourceHandle handle, size_t num_tensors,
                                               cudaStream_t stream);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_RESOURCE_HANDLE_H_
