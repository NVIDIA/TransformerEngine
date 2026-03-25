/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/device_tensor_from_host_pointers.h>

#include "../common.h"
#include "../util/logging.h"

namespace {

constexpr int64_t kMaxKernelAddresses = 64;

struct HostPointersArgs {
  int64_t ptrs[kMaxKernelAddresses];
};

__global__ void write_pointers_kernel(HostPointersArgs args, int64_t *out, int64_t count,
                                      int64_t offset) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[offset + idx] = args.ptrs[idx];
  }
}

}  // namespace

void nvte_convert_pointers_to_tensor(const int64_t *host_ptrs, int64_t *output, int64_t count,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_convert_pointers_to_tensor);
  int64_t offset = 0;
  while (offset < count) {
    const int64_t chunk = std::min(kMaxKernelAddresses, count - offset);
    HostPointersArgs args{};
    for (int64_t i = 0; i < chunk; ++i) {
      args.ptrs[i] = host_ptrs[offset + i];
    }
    constexpr int threads = kMaxKernelAddresses;
    write_pointers_kernel<<<1, threads, 0, stream>>>(args, output, chunk, offset);
    NVTE_CHECK_CUDA(cudaGetLastError());
    offset += chunk;
  }
}
