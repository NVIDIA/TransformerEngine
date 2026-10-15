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

// Non-owning view passed by value to the updater and consumer kernels. Null pointers select the
// original per-translation-unit device symbols for calls without a handle.
struct GroupedTmaStorageView {
  GroupedTmaStorageView() = default;

  // The allocation holds four tensor-map regions followed by three metadata regions.
  GroupedTmaStorageView(char *base, size_t map_bytes, size_t metadata_bytes)
      : input(reinterpret_cast<CUtensorMap *>(base)),
        act_input(reinterpret_cast<CUtensorMap *>(base + map_bytes)),
        output_rowwise(reinterpret_cast<CUtensorMap *>(base + 2 * map_bytes)),
        output_colwise(reinterpret_cast<CUtensorMap *>(base + 3 * map_bytes)),
        rows(reinterpret_cast<size_t *>(base + 4 * map_bytes)),
        cols(reinterpret_cast<size_t *>(base + 4 * map_bytes + metadata_bytes)),
        offsets(reinterpret_cast<size_t *>(base + 4 * map_bytes + 2 * metadata_bytes)) {}

  CUtensorMap *input = nullptr;
  CUtensorMap *act_input = nullptr;
  CUtensorMap *output_rowwise = nullptr;
  CUtensorMap *output_colwise = nullptr;
  size_t *rows = nullptr;
  size_t *cols = nullptr;
  size_t *offsets = nullptr;
};

// Storage is private to the calling host thread and CUDA stream. Repeated
// lookups for that pair do not acquire a handle-wide lock.
GroupedTmaStorageView get_grouped_tma_storage(NVTEResourceHandle handle, size_t num_tensors,
                                               cudaStream_t stream);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_RESOURCE_HANDLE_H_
